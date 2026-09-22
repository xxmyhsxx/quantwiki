#!/usr/bin/env python3
"""Small CUDA operator collection harness. No custom kernels or automatic installs."""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import random
import statistics
import subprocess
import shutil
import time

def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number

def summarize(values):
    if not values or not all(math.isfinite(v) and v > 0 for v in values):
        raise ValueError("nonpositive/nonfinite timing; increase repeats or investigate")
    xs = sorted(values)
    def q(p):
        x = (len(xs) - 1) * p
        lo = int(x); hi = min(lo + 1, len(xs) - 1)
        return xs[lo] + (xs[hi] - xs[lo]) * (x - lo)
    return {"median_ms": statistics.median(xs), "q25_ms": q(.25), "q75_ms": q(.75),
            "min_ms": min(xs), "max_ms": max(xs), "samples_ms": values,
            "statistic": "distribution of per-call averages of repeated-call batches"}

def validate_case(case, repeats):
    for key in ("calls", "validate", "metadata", "repeat_safe"):
        if key not in case:
            raise ValueError("factory missing " + key)
    if not case["calls"] or not all(callable(fn) for fn in case["calls"].values()):
        raise ValueError("calls must be a nonempty mapping of callables")
    if not callable(case["validate"]):
        raise ValueError("validate must be callable and raise on failure")
    if not case["repeat_safe"] and repeats != 1:
        raise ValueError("stateful case requires --repeats 1 and prepare outside timing")
    if not case["repeat_safe"] and not callable(case.get("prepare")):
        raise ValueError("stateful case requires prepare(op_name)")

def environment(torch):
    def version(name):
        try: return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: return None
    dev = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(dev)
    driver = None
    if shutil.which("nvidia-smi"):
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,uuid,driver_version,pstate,clocks.sm,clocks.mem,power.draw",
             "--format=csv"], capture_output=True, text=True, timeout=5)
        driver = result.stdout.strip() if result.returncode == 0 else result.stderr.strip()
    return {"python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "cuda_build": torch.version.cuda,
            "triton": version("triton"), "sglang": version("sglang"),
            "sgl_kernel": version("sgl-kernel"), "device_index": dev,
            "device_name": props.name, "capability": list(torch.cuda.get_device_capability(dev)),
            "total_memory": props.total_memory,
            "nvidia_smi_snapshot_all_devices": driver,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision()}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--factory", default="performance_cases:rmsnorm")
    p.add_argument("--config", default='{"m":16,"k":1024,"dtype":"float16"}',
                   help="JSON object passed to the factory")
    p.add_argument("--op", default="all", help="one key of calls, or all")
    p.add_argument("--mode", choices=["check", "bench", "trace", "torch-profiler"], default="check")
    p.add_argument("--timer", choices=["event", "wall"], default="event")
    p.add_argument("--warmup", type=positive, default=10)
    p.add_argument("--repeats", type=positive, default=20)
    p.add_argument("--samples", type=positive, default=30)
    p.add_argument("--trace-iterations", type=positive, default=5)
    p.add_argument("--output", type=Path, help="new JSON path; existing file is rejected")
    p.add_argument("--trace-output", type=Path, help="new Chrome trace path for torch-profiler")
    args = p.parse_args()
    config = json.loads(args.config)
    if not isinstance(config, dict): p.error("--config must be a JSON object")
    if args.output and args.output.exists(): p.error("--output already exists")
    if args.mode == "torch-profiler":
        if not args.trace_output: p.error("torch-profiler requires --trace-output")
        if args.trace_output.exists(): p.error("--trace-output already exists")
    try:
        import torch
    except ImportError:
        p.error("PyTorch is not installed; no GPU measurements produced")
    if not torch.cuda.is_available() or torch.version.cuda is None:
        p.error("NVIDIA CUDA device/build required; no GPU measurements produced")
    module, name = args.factory.split(":", 1)
    mod = importlib.import_module(module)
    factory = getattr(mod, name)
    with torch.inference_mode():
        case = factory(config)
        validate_case(case, args.repeats)
        names = list(case["calls"]) if args.op == "all" else [args.op]
        if any(name not in case["calls"] for name in names): p.error("unknown --op")
        prepare = case.get("prepare", lambda name: None)
        report = {"factory": args.factory, "config": config, "mode": args.mode,
                  "case": case["metadata"], "environment": environment(torch),
                  "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "factory_sha256": hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest(),
                  "timings": {}, "gpu_executed": True,
                  "warmup_calls_per_op": args.warmup, "samples": args.samples,
                  "repeats_per_sample": args.repeats,
                  "cache_policy": "same allocations repeatedly used; no explicit cache flush; not guaranteed hot"}
        report["validation_before"] = case["validate"]()
        torch.cuda.synchronize()
        if args.mode != "check":
            for name in names:
                for _ in range(args.warmup):
                    prepare(name)
                    case["calls"][name]()
            torch.cuda.synchronize()
        if args.mode == "bench":
            samples = {name: [] for name in names}
            # Create/initialize events before collecting samples.
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(); end.record(); end.synchronize()
            order_rng = random.Random(22)
            for _ in range(args.samples):
                order = names.copy(); order_rng.shuffle(order)
                for name in order:
                    prepare(name)
                    torch.cuda.synchronize()
                    if args.timer == "event":
                        start.record()
                        for _ in range(args.repeats): case["calls"][name]()
                        end.record(); end.synchronize()
                        elapsed = start.elapsed_time(end)
                    else:
                        before = time.perf_counter()
                        for _ in range(args.repeats): case["calls"][name]()
                        torch.cuda.synchronize()
                        elapsed = (time.perf_counter() - before) * 1000
                    samples[name].append(elapsed / args.repeats)
            report["timings"] = {name: summarize(values) for name, values in samples.items()}
            report["timer"] = args.timer
            report["timing_scope"] = (
                "current-stream interval incl. kernels, stream idle gaps and needed copies; not all host cost"
                if args.timer == "event" else "host submission plus completion of the synchronized device")
        elif args.mode in ("trace", "torch-profiler"):
            def workload():
                for _ in range(args.trace_iterations):
                    for name in names:
                        prepare(name)
                        torch.cuda.nvtx.range_push("perf_target")
                        try:
                            with torch.profiler.record_function("op:" + name):
                                case["calls"][name]()
                        finally: torch.cuda.nvtx.range_pop()
                torch.cuda.synchronize()
            if args.mode == "torch-profiler":
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=True, with_stack=False, profile_memory=False
                ) as prof:
                    workload()
                prof.export_chrome_trace(str(args.trace_output))
                report["trace_path"] = str(args.trace_output)
            else:
                workload()
            report["timing_scope"] = "diagnostic capture only; benchmark numbers intentionally absent"
        report["validation_after"] = case["validate"]()
        torch.cuda.synchronize()
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x") as f: f.write(text)
    print(text, end="")

if __name__ == "__main__":
    main()
