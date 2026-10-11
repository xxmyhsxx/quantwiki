"""Build/load the archived AWQ kernels and adapt one HF5 return-value boundary."""
import sys
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


def initialize(root):
    root = Path(root)
    source = root / "upstream/awq/kernels/csrc"
    build = root / "build"
    build.mkdir(exist_ok=True)
    engine = load(
        name="awq_inference_engine",
        sources=[str(root / "binding.cpp"),
                 str(source / "quantization_new/gemm/gemm_cuda.cu"),
                 str(source / "quantization_new/gemv/gemv_cuda.cu")],
        extra_include_paths=[str(source)],
        extra_cflags=["-O3", "-std=c++17", "-DENABLE_BF16"],
        extra_cuda_cflags=["-O3", "-std=c++17", "-DENABLE_BF16",
                          "-U__CUDA_NO_HALF_OPERATORS__",
                          "-U__CUDA_NO_HALF_CONVERSIONS__",
                          "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                          "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                          "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                          "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                          "--expt-relaxed-constexpr", "--expt-extended-lambda",
                          "--use_fast_math"],
        build_directory=str(build), verbose=True,
    )
    sys.modules["awq_inference_engine"] = engine
    sys.path.insert(0, str(root / "upstream"))
    # run_awq indexes layer output as a tuple in this fixed upstream version.
    # HF5 returns a Tensor. Patch only the experiment's imported function;
    # keep source archive and all search/quantization/packing math unchanged.
    from awq.quantize import pre_quant
    import inspect
    original = inspect.getsource(pre_quant.run_awq)
    old = "inps = layer(inps, **layer_kwargs)[0]"
    assert original.count(old) == 1
    adapted = original.replace(old, "inps = layer(inps, **layer_kwargs)")
    exec(compile(adapted, "<awq-hf5-return-adapter>", "exec"), pre_quant.__dict__)
    return engine
