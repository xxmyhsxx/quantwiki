"""Derive the fixed report from completed raw records; does not run a model."""
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def read(name):
    return json.loads((RESULTS / f"{name}.json").read_text())


def metrics(row):
    samples = row["samples"]
    return {key: {"median": statistics.median(s[key] for s in samples),
                  "mean": statistics.mean(s[key] for s in samples),
                  "sample_stdev": statistics.stdev(s[key] for s in samples),
                  "min": min(s[key] for s in samples), "max": max(s[key] for s in samples)}
            for key in ["prefill_ms", "decode_ms_per_step", "decode_tokens_per_second"]}


if __name__ == "__main__":
    q = {name: read(f"quality-{name}") for name in ["fp16", "fake", "packed"]}
    assert len({(d["blocks"], d["predicted_tokens"], d["context_length"]) for d in q.values()}) == 1
    pack, alignment, transform = read("packing"), read("backend-alignment"), read("transform-equivalence")
    assert pack["quantized_linear_layers"] == 224
    assert all(row["code_mismatches"] == 0 for row in pack["layers"])
    assert alignment["strict_checkpoint_reload"]
    assert all(row["relative_rmse"] < alignment["relative_rmse_tolerance"]
               for row in alignment["metrics"].values())
    assert all(row["relative_rmse"] < transform["relative_rmse_tolerance"]
               for row in transform["metrics"].values())
    kernels = read("executed-kernels")["cuda_kernel_names"]
    assert any("gemv_kernel" in name for name in kernels)
    assert any("gemm_w4a16" in name for name in kernels)
    before = {line for line in (RESULTS / "environment-before.txt").read_text().splitlines() if line}
    after = {line for line in (RESULTS / "environment-after.txt").read_text().splitlines() if line}
    assert after - before == {"zstandard==0.25.0"} and not before - after
    fp16 = read("benchmark-fp16-final")["measurements"]
    packed = read("benchmark-packed")["measurements"]
    cases = []
    for a, b in zip(fp16, packed, strict=True):
        assert (a["batch"], a["prompt_tokens"]) == (b["batch"], b["prompt_tokens"])
        assert (a["decode_steps"], a["warmup"], a["repeats"]) == (b["decode_steps"], b["warmup"], b["repeats"])
        assert a["gpu_before"].split(",")[0] == b["gpu_before"].split(",")[0]
        am, bm = metrics(a), metrics(b)
        cases.append({"batch": a["batch"], "prompt_tokens": a["prompt_tokens"],
                      "fp16": am, "packed": bm,
                      "prefill_latency_ratio_packed_over_fp16": bm["prefill_ms"]["median"] / am["prefill_ms"]["median"],
                      "decode_throughput_ratio_packed_over_fp16": bm["decode_tokens_per_second"]["median"] / am["decode_tokens_per_second"]["median"],
                      "peak_allocated_GiB": {"fp16": a["max_allocated_bytes"] / 2**30,
                                              "packed": b["max_allocated_bytes"] / 2**30},
                      "peak_reserved_GiB": {"fp16": a["max_reserved_bytes"] / 2**30,
                                             "packed": b["max_reserved_bytes"] / 2**30},
                      "telemetry": {"fp16_before": a["gpu_before"], "fp16_after": a["gpu_after"],
                                    "packed_before": b["gpu_before"], "packed_after": b["gpu_after"]}})
    data = read("inputs")
    fp16_bytes = sum(v["bytes"] for name, v in data["model_files"].items()
                     if name.endswith(".safetensors"))
    record = {"date": "2026-10-03", "status": "completed",
              "model": "Llama-2-7B", "scheme": "AWQ W4A16 asymmetric g128",
              "backend": "archived WQLinear CUDA GEMV/GEMM + Transformers SDPA",
              "calibration": {key: data[key] for key in ["calibration_blocks", "calibration_tokens",
                                                         "calibration_token_sha256"]},
              "evaluation": {"blocks": q["fp16"]["blocks"],
                             "predicted_tokens": q["fp16"]["predicted_tokens"],
                             "context_length": q["fp16"]["context_length"],
                             "token_sha256": data["evaluation_token_sha256"]},
              "ppl": {name: d["ppl"] for name, d in q.items()},
              "packed_ppl_increase_over_fp16_percent": 100 * (q["packed"]["ppl"] / q["fp16"]["ppl"] - 1),
              "packed_ppl_increase_over_simulation_percent": 100 * (q["packed"]["ppl"] / q["fake"]["ppl"] - 1),
              "numerics": {"transform": transform, "backend_alignment": alignment},
              "packing": {"layers": pack["quantized_linear_layers"], "codes_checked": pack["codes_checked"],
                          "checkpoint_bytes": pack["checkpoint_bytes"],
                          "checkpoint_sha256": pack["checkpoint_sha256"],
                          "float_safetensors_bytes": fp16_bytes,
                          "checkpoint_size_ratio_float_over_packed": fp16_bytes / pack["checkpoint_bytes"]},
              "performance": cases,
              "dependency_delta": {"added": sorted(after - before), "removed": sorted(before - after)},
              "limitations": ["one model, seed and calibration corpus", "language-model PPL and fixed-input logits",
                              "cached teacher-forced forward timing, not serving latency or output quality",
                              "FP16, g128, single A100; no BF16, TP, MoE or KV quantization",
                              "not bit-exact and not a reproduction of the full TinyChat system",
                              "thermal state differs; telemetry retained, timing repeats are not independent runs"]}
    (RESULTS / "summary.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    files = [p for p in RESULTS.iterdir() if p.is_file() and p.name != "record-manifest.json"]
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    (RESULTS / "record-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": record["status"], "ppl": record["ppl"],
                      "packed_ppl_increase_percent": record["packed_ppl_increase_over_fp16_percent"],
                      "checkpoint_ratio": record["packing"]["checkpoint_size_ratio_float_over_packed"],
                      "case_ratios": [{"batch": r["batch"], "input": r["prompt_tokens"],
                                       "prefill_ratio": r["prefill_latency_ratio_packed_over_fp16"],
                                       "decode_ratio": r["decode_throughput_ratio_packed_over_fp16"]}
                                      for r in cases]}, indent=2))
