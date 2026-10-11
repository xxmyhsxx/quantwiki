"""One fixed-model experiment, using archived AWQ code and existing local data."""
import argparse
import gc
import hashlib
import json
import math
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
MODEL = "/data/xiao/models/Llama-2-7b"
PILE = "/data/xiao/datasets/pile-val/val.jsonl.zst"
TEST = "/data/xiao/datasets/wikitext/wikitext-2-raw-v1/test-00000-of-00001.parquet"
QCONFIG = {"zero_point": True, "q_group_size": 128}
torch.set_num_threads(8)
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
torch.backends.cuda.matmul.allow_tf32 = False


def save(name, data):
    (ROOT / "results" / f"{name}.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print("SAVED", name, flush=True)


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def prepare():
    from datasets import load_dataset
    import pyarrow.parquet as pq
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    dataset = load_dataset("json", data_files=PILE, split="train", streaming=True,
                           cache_dir=str(ROOT / "cache"))
    dataset = dataset.map(lambda row, idx: {"original_row": idx}, with_indices=True)
    dataset = dataset.shuffle(seed=42, buffer_size=10000)
    tokens, documents = [], []
    for row in dataset:
        text = row["text"].strip()
        if not text:
            continue
        ids = tokenizer.encode(text)
        if len(ids) > 512:
            continue
        tokens.extend(ids)
        documents.append({"row": row["original_row"], "tokens": len(ids),
                          "text_sha256": hashlib.sha256(text.encode()).hexdigest()})
        if len(documents) == 128:
            break
    blocks = torch.tensor(tokens[:len(tokens) // 512 * 512]).reshape(-1, 512)
    assert len(documents) == 128 and len(blocks) > 0
    text = "\n\n".join(pq.read_table(TEST)["text"].to_pylist())
    ids = torch.tensor(tokenizer.encode(text))
    evaluation = ids[:len(ids) // 2048 * 2048].reshape(-1, 2048)
    torch.save({"calibration": blocks, "evaluation": evaluation,
                "test_tokens": ids}, ROOT / "inputs.pt")
    model_files = [p for p in Path(MODEL).iterdir()
                   if p.suffix in [".json", ".model", ".safetensors"]]
    save("inputs", {"seed": 42, "model": MODEL,
                    "model_files": {p.name: {"bytes": p.stat().st_size,
                                             "sha256": file_hash(p)} for p in sorted(model_files)},
                    "pile_file_sha256": file_hash(PILE),
                    "calibration_documents": documents,
                    "calibration_blocks": len(blocks),
                    "calibration_tokens": blocks.numel(),
                    "calibration_token_sha256": tensor_hash(blocks),
                    "calibration_dropped_tokens": len(tokens) - blocks.numel(),
                    "test_file_sha256": file_hash(TEST),
                    "test_tokens": len(ids), "evaluation_blocks": len(evaluation),
                    "evaluation_predicted_tokens": len(evaluation) * 2047,
                    "evaluation_token_sha256": tensor_hash(evaluation),
                    "block_length": 2048, "calibration_shuffle_buffer": 10000,
                    "python": platform.python_version(), "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,uuid,driver_version",
                                                     "--format=csv,noheader"], text=True)})


def load_float():
    model, info = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, attn_implementation="sdpa",
        local_files_only=True, use_safetensors=True, output_loading_info=True)
    save(f"load-float-{args.stage}", {k: sorted(v) if isinstance(v, set) else v
                                     for k, v in info.items()})
    assert not info.get("missing_keys") and not info.get("mismatched_keys"), info
    return model.eval()


def inputs():
    return torch.load(ROOT / "inputs.pt", weights_only=True)


def differences(actual, reference):
    x, y = actual.float(), reference.float()
    error = x - y
    return {"rmse": error.square().mean().sqrt().item(),
            "relative_rmse": (error.square().mean() / y.square().mean()).sqrt().item(),
            "max_absolute": error.abs().max().item(),
            "top1_agreement": (x.argmax(-1) == y.argmax(-1)).float().mean().item()}


@torch.inference_mode()
def probes(model, variant):
    data = inputs()["test_tokens"]
    prefills, decodes = [], []
    for offset in [0, 4096, 8192]:
        seq = data[offset:offset + 256].unsqueeze(0).cuda()
        prefills.append(model(seq, use_cache=False).logits.cpu())
        out = model(seq[:, :128], use_cache=True, logits_to_keep=1)
        cache = out.past_key_values
        logits = []
        for step in range(32):
            out = model(seq[:, 128 + step:129 + step], past_key_values=cache,
                        use_cache=True, logits_to_keep=1)
            cache = out.past_key_values
            logits.append(out.logits.cpu())
        decodes.append(torch.cat(logits, dim=1))
    values = {"prefill": torch.cat(prefills), "decode": torch.cat(decodes)}
    torch.save(values, ROOT / f"probes-{variant}.pt")
    return values


@torch.inference_mode()
def quality(model, variant):
    blocks = inputs()["evaluation"]
    losses = []
    started = time.perf_counter()
    for index, block in enumerate(blocks):
        block = block.unsqueeze(0).cuda()
        logits = model(block, use_cache=False).logits
        loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                               block[:, 1:].reshape(-1), reduction="sum").item()
        losses.append(loss)
        if index % 20 == 0:
            print("QUALITY", variant, index, "/", len(blocks), flush=True)
    result = {"model": MODEL, "variant": variant, "context_length": 2048,
              "blocks": len(blocks), "predicted_tokens": len(blocks) * 2047,
              "nll_sum_per_block": losses,
              "mean_nll": sum(losses) / (len(blocks) * 2047),
              "ppl": math.exp(sum(losses) / (len(blocks) * 2047)),
              "seconds": time.perf_counter() - started}
    save(f"quality-{variant}", result)
    return result


@torch.inference_mode()
def benchmark(model, variant):
    data = inputs()["test_tokens"]
    measurements = []
    for batch in [1, 4]:
        for length in [512, 2048]:
            gpu_before = subprocess.check_output([
                "nvidia-smi", "--id=0",
                "--query-gpu=uuid,temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu",
                "--format=csv,noheader"], text=True).strip()
            prompt = data[:length].unsqueeze(0).repeat(batch, 1).cuda()
            future = data[length:length + 64].unsqueeze(0).repeat(batch, 1).cuda()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            samples = []
            for iteration in range(8):
                torch.cuda.synchronize()
                started = time.perf_counter()
                out = model(prompt, use_cache=True, logits_to_keep=1)
                cache = out.past_key_values
                torch.cuda.synchronize()
                prefill = time.perf_counter() - started
                started = time.perf_counter()
                for step in range(64):
                    out = model(future[:, step:step + 1].contiguous(),
                                past_key_values=cache, use_cache=True, logits_to_keep=1)
                    cache = out.past_key_values
                torch.cuda.synchronize()
                decode = time.perf_counter() - started
                if iteration >= 3:
                    samples.append({"prefill_ms": 1000 * prefill,
                                    "decode_ms": 1000 * decode,
                                    "decode_ms_per_step": 1000 * decode / 64,
                                    "decode_tokens_per_second": batch * 64 / decode})
                del out, cache
            measurements.append({"batch": batch, "prompt_tokens": length,
                                 "decode_steps": 64, "warmup": 3, "repeats": 5,
                                 "samples": samples,
                                 "max_allocated_bytes": torch.cuda.max_memory_allocated(),
                                 "max_reserved_bytes": torch.cuda.max_memory_reserved(),
                                 "gpu_before": gpu_before,
                                 "gpu_after": subprocess.check_output([
                                     "nvidia-smi", "--id=0",
                                     "--query-gpu=uuid,temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu",
                                     "--format=csv,noheader"], text=True).strip()})
            print("BENCHMARK", variant, batch, length, samples, flush=True)
    save(f"benchmark-{variant}", {"variant": variant, "mode": "teacher-forced cached forward",
                                  "attention": "sdpa", "cuda_graphs": False,
                                  "torch_threads": 8, "measurements": measurements})


def baseline():
    model = load_float().cuda()
    probes(model, "fp16")
    quality(model, "fp16")
    benchmark(model, "fp16-initial")


def performance_fp16():
    benchmark(load_float().cuda(), "fp16-final")


def initialize():
    from awq_runtime import initialize as init
    return init(ROOT)


@torch.inference_mode()
def kernel_smoke():
    initialize()
    from awq.quantize.quantizer import pseudo_quantize_tensor
    from awq.quantize.qmodule import WQLinear
    linear = torch.nn.Linear(4096, 4096, bias=False, device="cuda", dtype=torch.float16)
    linear.weight.data, scales, zeros = pseudo_quantize_tensor(
        linear.weight, 4, q_group_size=128, get_scale_zp=True)
    quantized = WQLinear.from_linear(linear, 4, 128, False, scales, zeros)
    expected = torch.round((linear.weight + (zeros * scales).repeat_interleave(128, dim=1))
                           / scales.repeat_interleave(128, dim=1)).int().cpu()
    assert torch.equal(unpack(quantized.qweight, 4096, 4096), expected)
    report = []
    for tokens in [1, 4, 7, 8, 32, 512, 2048]:
        x = torch.randn(1, tokens, 4096, device="cuda", dtype=torch.float16)
        error = differences(quantized(x), F.linear(x, linear.weight))
        torch.cuda.synchronize()
        assert error["relative_rmse"] < 0.005
        report.append({"tokens": tokens, **error})
    save("kernel-smoke", {"code_roundtrip_exact": True, "shape": [4096, 4096],
                          "metrics": report, "relative_rmse_tolerance": 0.005})


def calibrate():
    initialize()
    from awq.quantize import pre_quant
    from awq.utils import calib_data
    blocks = inputs()["calibration"]
    calib_data.get_calib_dataset = lambda **kwargs: [row.unsqueeze(0) for row in blocks]
    model = load_float()
    model.config.use_cache = False
    started = time.perf_counter()
    result = pre_quant.run_awq(model, None, 4, QCONFIG, n_samples=128, seqlen=512)
    torch.save(result, ROOT / "awq-search.pt")
    save("calibration", {"seconds": time.perf_counter() - started,
                         "scale_entries": len(result["scale"]),
                         "clip_entries": len(result["clip"]),
                         "search_sha256": file_hash(ROOT / "awq-search.pt"),
                         "calibration_blocks": len(blocks), "q_config": QCONFIG})


def simulate():
    initialize()
    from awq.quantize.auto_scale import apply_scale
    from awq.quantize.auto_clip import apply_clip
    from awq.quantize.quantizer import pseudo_quantize_model_weight
    result = torch.load(ROOT / "awq-search.pt", weights_only=True)
    model = load_float()
    apply_scale(model, result["scale"])
    model.cuda()
    transformed = probes(model, "scale-only")
    reference = torch.load(ROOT / "probes-fp16.pt", weights_only=True)
    metrics = {key: differences(transformed[key], reference[key]) for key in reference}
    save("transform-equivalence", {"metrics": metrics, "relative_rmse_tolerance": 0.005})
    assert all(m["relative_rmse"] < 0.005 for m in metrics.values()), metrics
    model.cpu()
    apply_clip(model, result["clip"])
    pseudo_quantize_model_weight(model, 4, QCONFIG)
    model.cuda()
    simulated = probes(model, "fake")
    save("simulation-vs-float", {key: differences(simulated[key], reference[key]) for key in reference})
    quality(model, "fake")


def unpack(weight, out_features, in_features):
    # Reverse the archived pack_intweight's row interleave and two permutations.
    packed = weight.detach().cpu().numpy().astype(np.uint16).reshape(out_features // 4,
                                                                      in_features // 64, 64)
    codes = np.stack([(packed >> (4 * i)) & 15 for i in range(4)], axis=-1)
    codes = codes.reshape(out_features // 4, in_features // 64, 4, 64)
    codes = codes.transpose(0, 2, 1, 3).reshape(out_features, in_features)
    codes = codes.reshape(out_features, in_features // 32, 4, 2, 4)
    codes = codes.transpose(0, 1, 2, 4, 3).reshape(out_features, in_features // 32, 32)
    codes = codes.reshape(out_features, in_features // 32, 4, 4, 2)
    codes = codes.transpose(0, 1, 3, 2, 4).reshape(out_features, in_features)
    return torch.from_numpy(codes.astype(np.int32))


def export():
    initialize()
    from awq.quantize.pre_quant import apply_awq
    from awq.quantize.quantizer import real_quantize_model_weight
    from awq.quantize.qmodule import WQLinear
    model = load_float()
    apply_awq(model, torch.load(ROOT / "awq-search.pt", weights_only=True))
    names = [name for name, module in model.named_modules()
             if isinstance(module, torch.nn.Linear) and name != "lm_head"]
    original = WQLinear.from_linear.__func__
    reports = []

    @torch.no_grad()
    def checked(cls, linear, w_bit, group_size, init_only=False, scales=None, zeros=None):
        quantized = original(cls, linear, w_bit, group_size, init_only, scales, zeros)
        expected = torch.round((linear.weight + (zeros * scales).repeat_interleave(128, dim=1))
                               / scales.repeat_interleave(128, dim=1)).int().cpu()
        recovered = unpack(quantized.qweight, linear.out_features, linear.in_features)
        assert torch.equal(expected, recovered), names[len(reports)]
        row = {"name": names[len(reports)], "shape": list(linear.weight.shape),
               "codes": expected.numel(), "code_mismatches": 0,
               "code_sha256": tensor_hash(expected),
               "packed_sha256": tensor_hash(quantized.qweight), "kernel": []}
        # Exercise representative Llama weight shapes across GEMV/GEMM boundary.
        if len(reports) < 7:
            for tokens in [1, 4, 8, 32]:
                x = torch.randn(1, tokens, linear.in_features, device="cuda", dtype=torch.float16)
                ref = F.linear(x, linear.weight)
                got = quantized(x)
                torch.cuda.synchronize()
                error = differences(got, ref)
                assert error["relative_rmse"] < 0.005, (row["name"], tokens, error)
                row["kernel"].append({"tokens": tokens, **error})
        reports.append(row)
        return quantized

    WQLinear.from_linear = classmethod(checked)
    started = time.perf_counter()
    real_quantize_model_weight(model, 4, QCONFIG)
    WQLinear.from_linear = classmethod(original)
    assert len(reports) == len(names) == 224
    state = model.state_dict()
    torch.save(state, ROOT / "llama-2-7b-awq-g128-v2.pt")
    AutoConfig.from_pretrained(MODEL, local_files_only=True).save_pretrained(ROOT / "checkpoint-config")
    save("packing", {"q_config": QCONFIG, "quantized_linear_layers": len(reports),
                     "codes_checked": sum(row["codes"] for row in reports),
                     "layers": reports, "export_seconds": time.perf_counter() - started,
                     "checkpoint_bytes": (ROOT / "llama-2-7b-awq-g128-v2.pt").stat().st_size,
                     "checkpoint_sha256": file_hash(ROOT / "llama-2-7b-awq-g128-v2.pt"),
                     "tensor_bytes": sum(t.numel() * t.element_size() for t in state.values())})


def packed():
    engine = initialize()
    from accelerate import init_empty_weights
    from awq.quantize.quantizer import real_quantize_model_weight
    from awq.quantize.qmodule import WQLinear
    config = AutoConfig.from_pretrained(ROOT / "checkpoint-config", local_files_only=True)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, dtype=torch.float16,
                                                attn_implementation="sdpa")
        real_quantize_model_weight(model, 4, QCONFIG, init_only=True)
    state = torch.load(ROOT / "llama-2-7b-awq-g128-v2.pt", weights_only=True, mmap=True)
    loaded = model.load_state_dict(state, strict=True, assign=True)
    assert not loaded.missing_keys and not loaded.unexpected_keys
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in state.items())
    assert sum(isinstance(m, WQLinear) for m in model.modules()) == 224
    del state
    gc.collect()
    model = model.eval().cuda()
    calls = {"gemv": 0, "gemm": 0}
    original_gemv, original_gemm = engine.gemv_forward_cuda_new, engine.gemm_forward_cuda_new

    def gemv(*args):
        calls["gemv"] += 1
        return original_gemv(*args)

    def gemm(*args):
        calls["gemm"] += 1
        return original_gemm(*args)

    engine.gemv_forward_cuda_new, engine.gemm_forward_cuda_new = gemv, gemm
    actual = probes(model, "packed")
    engine.gemv_forward_cuda_new, engine.gemm_forward_cuda_new = original_gemv, original_gemm
    reference = torch.load(ROOT / "probes-fake.pt", weights_only=True)
    metrics = {key: differences(actual[key], reference[key]) for key in reference}
    save("backend-alignment", {"strict_checkpoint_reload": True, "dispatch_calls": calls,
                               "metrics": metrics, "relative_rmse_tolerance": 0.02})
    assert all(m["relative_rmse"] < 0.02 for m in metrics.values()), metrics
    quality(model, "packed")
    benchmark(model, "packed")
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as profiler:
        with torch.inference_mode():
            prompt = inputs()["test_tokens"][:128].unsqueeze(0).cuda()
            out = model(prompt, use_cache=True, logits_to_keep=1)
            model(prompt[:, :1], past_key_values=out.past_key_values, use_cache=True,
                  logits_to_keep=1)
            torch.cuda.synchronize()
    kernels = sorted({event.name for event in profiler.events()
                      if event.device_type == torch.autograd.DeviceType.CUDA})
    save("executed-kernels", {"cuda_kernel_names": kernels})
    profiler.export_chrome_trace(str(ROOT / "results/packed-trace.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "kernel_smoke", "baseline", "calibrate", "simulate", "export", "packed", "performance_fp16"])
    args = parser.parse_args()
    print("STAGE", args.stage, "START", time.strftime("%Y-%m-%dT%H:%M:%S%z"), flush=True)
    globals()[args.stage]()
    print("STAGE", args.stage, "COMPLETE", flush=True)
