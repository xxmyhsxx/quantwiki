"""Example factories. CUDA execution is required; no custom kernel is implemented here."""
def options(config):
    import torch
    m, k = int(config.get("m", 16)), int(config.get("k", 1024))
    n = int(config.get("n", k))
    if min(m, n, k) <= 0: raise ValueError("m/n/k must be positive")
    dtype_name = config.get("dtype", "float16")
    if dtype_name not in ("float16", "bfloat16"): raise ValueError("dtype must be float16 or bfloat16")
    dtype = getattr(torch, dtype_name)
    seed = int(config.get("seed", 22))
    torch.manual_seed(seed)
    return torch, m, n, k, dtype, seed

def rmsnorm(config):
    torch, m, _, k, dtype, seed = options(config)
    eps = float(config.get("eps", 1e-6))
    if eps <= 0: raise ValueError("eps must be positive")
    x = torch.randn((m, k), device="cuda", dtype=dtype)
    w = torch.randn(k, device="cuda", dtype=dtype)
    # Freeze the high precision reference outside all measured regions.
    xd, wd = x.cpu().double(), w.cpu().double()
    expected = (xd * torch.rsqrt(xd.square().mean(-1, keepdim=True) + eps) * wd)
    rtol = float(config.get("rtol", 1e-2 if dtype == torch.float16 else 3e-2))
    atol = float(config.get("atol", 1e-3 if dtype == torch.float16 else 1e-2))
    def eager():
        xf = x.float()
        return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps) * w.float()).to(dtype)
    def native():
        return torch.nn.functional.rms_norm(x, (k,), w, eps=eps)
    calls = {"eager_fp32": eager, "torch_rmsnorm": native}
    def validate():
        checks = {}
        for name, fn in calls.items():
            actual = fn().cpu().double()
            if not torch.isfinite(actual).all(): raise AssertionError(name + " has nonfinite output")
            torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
            checks[name] = {"max_abs_error": (actual - expected).abs().max().item()}
        return {"status": "passed", "reference": "CPU float64 from frozen input", "checks": checks}
    return {"calls": calls, "validate": validate, "repeat_safe": True,
            "metadata": {"operator": "RMSNorm without residual", "shape": [m, k], "dtype": str(dtype),
                         "strides": list(x.stride()), "epsilon": eps, "seed": seed,
                         "rtol": rtol, "atol": atol, "allocations": "wrapper outputs/intermediates included",
                         "candidate_identity": "installed PyTorch F.rms_norm; fusion not assumed",
                         "streams": "current stream only", "custom_kernel_executed": False}}

def sglang_w8a8(config):
    torch, m, n, k, dtype, seed = options(config)
    if k % 16 or n % 8: raise ValueError("SGLang path requires K%16=0 and N%8=0")
    from sglang.srt.layers.quantization.int8_kernel import per_token_quant_int8
    from sgl_kernel import int8_scaled_mm
    x = torch.randn((m, k), device="cuda", dtype=dtype)
    # Weight is already quantized and transposed outside the measured call.
    weight = torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8).t()
    sw = torch.full((n, 1), .01, device="cuda", dtype=torch.float32)
    qx, sx = per_token_quant_int8(x)
    rtol = float(config.get("rtol", 1e-2 if dtype == torch.float16 else 3e-2))
    atol = float(config.get("atol", 1e-3 if dtype == torch.float16 else 1e-2))
    xc = x.cpu().float()
    absmax = xc.abs().amax(-1, keepdim=True).clamp_min(1e-10)
    ref_s = absmax / 127
    scaled = xc * (127 / absmax)
    # Add 0.5 in FP64: FP32 addition can turn a value just below a tie into a tie.
    scaled64 = scaled.double()
    ref_q = (scaled64.sign() * torch.floor(scaled64.abs() + .5)).to(torch.int8)
    # FP64 exactly accumulates these bounded integer products for practical K.
    if k * 128 * 128 >= 2**53: raise ValueError("K exceeds exact integer FP64 reference range")
    dots = ref_q.double() @ weight.cpu().double()
    expected = dots * ref_s.double() * sw.cpu().double().reshape(1, n)
    def quant():
        return per_token_quant_int8(x)
    def gemm():
        return int8_scaled_mm(qx, weight, sx, sw, out_dtype=dtype, bias=None)
    def linear():
        aq, scale = quant()
        return int8_scaled_mm(aq, weight, scale, sw, out_dtype=dtype, bias=None)
    def validate():
        aq, scale = quant()
        torch.testing.assert_close(aq.cpu(), ref_q, rtol=0, atol=0)
        torch.testing.assert_close(scale.cpu(), ref_s, rtol=1e-6, atol=1e-12)
        errors = {}
        for name, fn in {"gemm": gemm, "linear": linear}.items():
            actual = fn().cpu().double()
            if not torch.isfinite(actual).all(): raise AssertionError(name + " has nonfinite output")
            torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
            errors[name] = (actual - expected).abs().max().item()
        return {"status": "passed", "activation_codes_exact": True,
                "reference": "CPU float64 dot of integer codes, then row/column scales",
                "max_abs_error": errors, "scope": "quantized representation, not original float model quality"}
    return {"calls": {"quant": quant, "gemm": gemm, "linear": linear},
            "validate": validate, "repeat_safe": True,
            "metadata": {"operator": "SGLang symmetric W8A8 without bias",
                         "shape_MNK": [m, n, k], "dtype": str(dtype), "seed": seed,
                         "weight_strides": list(weight.stride()), "rtol": rtol, "atol": atol,
                         "activations": "dynamic per-token", "weights": "per-output-channel",
                         "allocations": "real wrappers incl. output/scales/workspace allocations",
                         "offline_weight_quantization_and_transpose": "excluded",
                         "gemm_inputs": "fixed prequantized activation and scales",
                         "linear_scope": "activation quant plus scaled GEMM in one call",
                         "streams": "current stream only",
                         "source_contract": "sglang 2f730e299f3b574e3bee2c6ef9669fa2a5b26dbc; installed version must be checked"}}
