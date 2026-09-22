#!/usr/bin/env python3
"""CPU-only checks of the teaching harness and analysis budgets, never a GPU benchmark."""
import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import struct
import tempfile

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("operator_harness", HERE / "benchmark_operator.py")
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)
checks = []

def add(name, detail):
    checks.append({"name": name, "status": "passed", "detail": detail})

def expect_error(fn):
    try: fn()
    except (ValueError, TypeError): return
    raise AssertionError("expected rejection")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = harness.summarize([4., 1., 3., 2.])
    assert result["median_ms"] == 2.5
    assert result["q25_ms"] == 1.75 and result["q75_ms"] == 3.25
    assert result["samples_ms"] == [4., 1., 3., 2.]
    add("sample_statistics", "Known synthetic values; raw order retained; not GPU timings")
    below_half = .5 - 2**-25  # largest FP32 value below 0.5
    fp32_sum = struct.unpack('f', struct.pack('f', below_half + .5))[0]
    assert math.floor(fp32_sum) == 1 and math.floor(below_half + .5) == 0
    add("round_away_reference_near_tie", "FP64 addition preserves the FP32 scaled value below a tie")
    for bad in [[], [0.], [-1.], [math.nan], [math.inf]]:
        expect_error(lambda bad=bad: harness.summarize(bad))
    add("reject_invalid_timings", "Empty, zero, negative and nonfinite inputs rejected")

    case = {"calls": {"op": lambda: None}, "validate": lambda: None,
            "metadata": {}, "repeat_safe": False}
    expect_error(lambda: harness.validate_case(case, 20))
    expect_error(lambda: harness.validate_case(case, 1))
    case["prepare"] = lambda name: None
    harness.validate_case(case, 1)
    case["repeat_safe"] = True
    harness.validate_case(case, 20)
    add("stateful_repetition_contract", "Stateful batches rejected; explicit prepare and repeats=1 required")

    for filename in ["benchmark_operator.py", "performance_cases.py", "check_analysis.py"]:
        ast.parse((HERE / filename).read_text())
    help_run = subprocess.run([sys.executable, str(HERE / "benchmark_operator.py"), "--help"],
                              capture_output=True, text=True)
    assert help_run.returncode == 0 and "--trace-output" in help_run.stdout
    add("syntax_and_cli", "All Python sources parsed; help works without importing CUDA or framework factories")

    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        # Simulate an importable PyTorch without CUDA: tests the refusal path, not any tensor API.
        (folder / "torch.py").write_text(
            "class cuda:\n    @staticmethod\n    def is_available(): return False\n"
            "class version:\n    cuda = None\n")
        output = folder / "result.json"
        env = dict(os.environ, PYTHONPATH=tmp)
        proc = subprocess.run([sys.executable, str(HERE / "benchmark_operator.py"),
                               "--mode", "bench", "--output", str(output)],
                              capture_output=True, text=True, env=env)
        assert proc.returncode == 2 and "no GPU measurements produced" in proc.stderr
        assert not output.exists()
        output.write_text("keep previous result")
        proc = subprocess.run([sys.executable, str(HERE / "benchmark_operator.py"),
                               "--output", str(output)], capture_output=True, text=True, env=env)
        assert proc.returncode == 2 and output.read_text() == "keep previous result"
    add("no_cuda_and_result_preservation", "Subprocess with CUDA-unavailable stub produces no timing; old output preserved")

    m, k, b = 16, 1024, 2
    input_bytes, output_bytes, shared_weight = m*k*b, m*k*b, k*b
    rms_min = 2*m*k*b+k*b
    assert rms_min == input_bytes+output_bytes+shared_weight == 67584
    logical_weight_per_row = m*k*b
    assert input_bytes*2 + output_bytes + logical_weight_per_row > rms_min
    add("rmsnorm_byte_accounting", {"ideal_unique_bytes": rms_min,
                                   "scope": "algorithmic lower-bound model, not measured HBM traffic"})

    m, n, k, g = 16, 1024, 1024, 128
    packed = (k*n)//2
    scales = 2*(k//g)*n
    total = 2*m*k+packed+2*m*n+scales
    assert packed == 524288 and scales == 16384 and total == 606208
    quant_written = m*k+4*m
    quant_roundtrip = 2*quant_written
    assert quant_written == 16448 and quant_roundtrip == 32896
    add("quantized_gemm_byte_accounting", {"hypothetical_w4_unique_bytes": total,
         "w8_activation_intermediate_write_bytes": quant_written,
         "w8_intermediate_write_plus_read_bytes": quant_roundtrip,
         "scope": "excludes unspecified metadata/padding/cache effects; not measured DRAM traffic"})

    # Abstract interval units, not a claim about any real trace.
    intervals = [(0, 10), (5, 15)]
    sum_work = sum(b-a for a,b in intervals)
    union = max(b for a,b in intervals)-min(a for a,b in intervals)
    assert sum_work == 20 and union == 15
    whole_before = 1+8+1
    whole_after = 1+4+1
    assert whole_before / whole_after < 2
    add("overlap_and_end_to_end_counterexamples",
        {"abstract_sum_vs_union": [sum_work, union], "synthetic_stage_speedup": whole_before/whole_after})

    report = {"date": "2026-09-22", "evidence": "CPU-only checks; no CUDA/NVIDIA tool execution",
              "python": sys.version.split()[0], "passed": len(checks), "checks": checks,
              "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in [HERE/"benchmark_operator.py", HERE/"performance_cases.py", HERE/"check_analysis.py"]}}
    content = json.dumps(report, indent=2, allow_nan=False)+"\n"
    if args.output:
        with args.output.open("x") as out: out.write(content)
    print(content, end="")

if __name__ == "__main__":
    main()
