---
title: CUDA 算子的编写、编译与 PyTorch 接入
type: recipe
tags:
  - kernels
  - cuda
  - deployment
sources:
  - raw/articles/2026-10-04/pytorch-cpp-cuda-operators
  - raw/articles/2026-09-21/cuda-simt-kernels
  - raw/repositories/2026-09-22/kernels
updated: 2026-10-04
---

# CUDA 算子的编写、编译与 PyTorch 接入

CUDA 设备函数只描述线程怎样计算。一个可调用的 PyTorch 算子还需要主机包装、编译与库加载、dispatcher 注册，以及与使用场景相符的元信息或梯度接口。本页用 $y=x+1$ 贯通这条链；教学代码由整理者按官方文档与固定模板改写，本次只做来源和代码审查，未编译或运行 GPU。

先掌握[张量布局与接口](../fundamentals/operators/tensor-layout-and-kernel-contracts.md)及[GPU 执行与存储](../fundamentals/hardware/gpu-execution-and-memory-hierarchy.md)。分块、归约、共享内存和 Tensor Core 的复杂性由已有[计算模式](../fundamentals/operators/gpu-kernel-computation-patterns.md)与[GEMM 实现](tensor-core-quantized-gemm.md)承接。

## 1. 先确定这个例子的接口

输入为 CUDA 上按默认布局连续的 FP32 tensor，任意 shape，`requires_grad=False`；输出是同 shape、dtype、device 的新连续 tensor，不修改或别名引用输入。空输入返回空输出。例子只实现 CUDA 推理接口，CPU、非连续输入和训练梯度不在支持范围内。

需要支持转置或切片时，应选择按 stride 寻址，或在公开包装中显式转为连续输入，并把复制算进完整调用成本。这里直接拒绝非连续输入，保持接口与线性寻址一致。

## 2. 设备函数与主机包装

将以下内容保存为 `add_one.cu`：

```cpp
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>

__global__ void add_one_kernel(
    const float* __restrict__ x, float* __restrict__ y, int64_t n) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t step = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; i < n; i += step) {
    y[i] = x[i] + 1.0f;
  }
}

at::Tensor add_one_cuda(const at::Tensor& x) {
  TORCH_CHECK(x.is_cuda(), "x must be CUDA");
  TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(!x.requires_grad(), "this example is inference-only");
  const c10::cuda::CUDAGuard guard(x.device());
  auto y = at::empty(x.sizes(), x.options());
  const int64_t n = x.numel();
  if (n == 0) return y;

  constexpr int threads = 256;
  const int blocks = static_cast<int>(
      std::min<int64_t>((n + threads - 1) / threads, 4096));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device());
  add_one_kernel<<<blocks, threads, 0, stream>>>(
      x.data_ptr<float>(), y.data_ptr<float>(), n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

TORCH_LIBRARY(quantwiki, m) {
  m.def("add_one(Tensor x) -> Tensor");
}

TORCH_LIBRARY_IMPL(quantwiki, CUDA, m) {
  m.impl("add_one", &add_one_cuda);
}
```

每个线程领取 $i=t+bT$，随后按整个 grid 的线程数跨步领取新元素。由整数商与余数可知，有效位置各有一个写入者，且读取后立即写自己的独立输出，所以这里不需要线程间同步。256 threads 和 4096 blocks 上限只是教学配置，不是性能最优结论；跨步循环保证超过一次 grid 工作量的元素仍被覆盖。

主机包装承担三个实际责任：按契约检查 tensor；用 device guard 选择输入所在设备；在该设备的当前 stream 启动。默认 stream 不能替代调用者的当前 stream，否则可能破坏与前后 PyTorch 操作的顺序。空输入在启动前返回，避免零个 block 的 launch。索引与元素数使用 64 位，不沿用初始化模板的 `int n`。

launch check 用于发现启动时可报告的 CUDA 错误；kernel 仍异步执行，它不等价于等待完成并核验全部设备访问。正常包装不插入全设备同步。实际开发的越界、竞争和数值验证见[正确性与测量](kernel-correctness-and-benchmarking.md)。（CUDA Writing SIMT Kernels 快照；HF kernels `kernel_cuda/kernel.cu`；PyTorch 自定义算子教程的 CUDA 注册节。）

## 3. 编译、加载，再从 dispatcher 调用

目标环境需要 CUDA 版 PyTorch、可用的 NVIDIA 驱动、与其构建要求相容的 CUDA toolkit/`nvcc`、主机 C++ 编译器和 Ninja。安装 PyTorch 不等于安装扩展编译工具链。`CUDA_HOME` 用来定位 toolkit；编译目标还要覆盖实际 GPU。比如只为 A100 构建时，可在启动 Python 前设置 `TORCH_CUDA_ARCH_LIST="8.0"`；增加其他 GPU 时重新选择目标，而不是把这个设置当作通用值。（`cpp_extension` 2.14：`CUDAExtension`、`load`、`load_inline`。）

在同目录保存并运行 `load_add_one.py`：

```python
from pathlib import Path
import torch
from torch.utils.cpp_extension import load

load(
    name="quantwiki_add_one",
    sources=[str(Path(__file__).with_name("add_one.cu"))],
    extra_cuda_cflags=["-O3"],
    is_python_module=False,
    verbose=True,
)

@torch.library.register_fake("quantwiki::add_one")
def add_one_fake(x):
    torch._check(x.device.type == "cuda")
    torch._check(x.dtype == torch.float32)
    torch._check(x.is_contiguous())
    torch._check(not x.requires_grad)
    return torch.empty_like(x, memory_format=torch.contiguous_format)

x = torch.randn(257, device="cuda", dtype=torch.float32)
y = torch.ops.quantwiki.add_one(x)
torch.testing.assert_close(y, x + 1, rtol=0, atol=0)
```

`load` 为源码生成 Ninja 构建，`.cu` 经 `nvcc` 编译并链接成动态库，再加载进当前进程。本例 `is_python_module=False` 且 `is_standalone=False`（默认），加载是副作用，调用入口在 `torch.ops`；不需要 `PYBIND11_MODULE`，也不能把返回值当作有 `add_one` 方法的 Python module。加载库时，`TORCH_LIBRARY` 定义 schema，`TORCH_LIBRARY_IMPL(..., CUDA, ...)` 绑定 CUDA 实现。

编译缓存减少重复构建，不证明不同 PyTorch、toolkit、编译器或架构之间的二进制兼容。本例使用普通 ATen/LibTorch API，面向当前环境编译；发布跨版本 wheel 时，应另按官方稳定 ABI 接口设计和验证，不能把普通 ATen 示例直接称为稳定 ABI 扩展。（PyTorch 官方文档 2026-10-04 快照，`load` 返回说明与 Custom C++ and CUDA Operators 的两种 API 分支。）

## 4. 可调用、可编译图使用、可训练是不同责任

| 接口 | 本例如何落实 | 不能据此推出什么 |
| --- | --- | --- |
| eager 调用 | 先加载库，再通过 schema 和 CUDA dispatch 调用 | 数学结果已经正确或性能已经改善 |
| 输出元信息 | 加载 schema 后注册 FakeTensor 函数，返回相同 shape/dtype/device 的连续输出 | 已执行真实 CUDA 代码或会自动与相邻 kernel 融合 |
| 输入修改与别名 | 本例新建输出，schema 无输入修改/输出别名声明 | 任意 `out` 或原地变体也可以复用同一 schema |
| autograd | 本例拒绝 `requires_grad=True`，没有训练接口 | 注册了 CUDA kernel 就拥有正确 backward |

FakeTensor 只计算元信息，不能读取真实输入数据或发起 CUDA launch；它必须与真实包装的输出布局及支持条件相符。若增加训练功能，需要通过 `torch.library.register_autograd` 登记 backward 和必要的上下文，并验证梯度；若改成修改传入的 `out`，schema 要用如 `Tensor(a!)` 的别名/修改标记明确声明。（官方教程的 FakeTensor、hybrid registration、autograd、mutable operators 各节。）

## 5. 将来执行这个例子时检查什么

先比较 `0、1、255、256、257` 与超过 `4096*256` 的元素数，覆盖空输入、尾部和跨步循环；再检查二维连续输入、输入未被修改、非连续/dtype/梯度输入按契约拒绝。多设备和非默认 stream 场景检查实际输入设备与执行顺序。

```python
# 接上前面的 load_add_one.py；这些是待执行的检查代码。
for n in (0, 1, 255, 256, 257, 4096 * 256 + 17):
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    torch.testing.assert_close(torch.ops.quantwiki.add_one(x), x + 1,
                               rtol=0, atol=0)
torch.library.opcheck(torch.ops.quantwiki.add_one.default, (x,))
compiled = torch.compile(torch.ops.quantwiki.add_one.default, fullgraph=True)
torch.testing.assert_close(compiled(x), x + 1, rtol=0, atol=0)
```

`opcheck` 检查注册与 PyTorch 子系统契约，不能替代数学正确性或梯度检查。需要优化时，再按[基线、计时和瓶颈分析](gpu-kernel-performance-analysis.md)比较完整调用；本页没有给出 GPU 验证或速度结论。

## 来源身份

- PyTorch `torch.utils.cpp_extension`（`https://docs.pytorch.org/docs/2.14/cpp_extension.html`），2.14 文档，获取于 2026-10-04T04:17:43+08:00；来源标识 `pytorch-cpp-cuda-operators-snapshot-2026-10-04`。核对编译、目标架构与 `load` 模式。
- Custom C++ and CUDA Operators（`https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html`），2026-10-04T04:17:49+08:00 快照，页面显示 Tutorials 2.14.0+cu130；同上来源标识。核对普通 ATen/稳定 ABI、schema、CUDA dispatch、FakeTensor、autograd 与 opcheck。
- Hugging Face kernels（`https://github.com/huggingface/kernels/tree/5c2cf07f7625e1b8c5fb60bcfb073cd055581cbc`），`5c2cf07f7625e1b8c5fb60bcfb073cd055581cbc`；`kernel-builder/src/init/templates/kernel_cuda/kernel.cu` 与 `torch-ext/torch_binding.cpp`。借鉴包装和注册关系，本页没有使用其完整构建系统。
- CUDA Programming Guide — Writing SIMT Kernels（`https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html`），2026-07-02T02:31:16+08:00 快照，`2-3-writing-simt-kernels-cuda-programming-guide-snapshot-2026-07-02`；线程索引、同步与 grid 跨步遍历。示例代码为整理者组合，不冒充上游逐字实现。
