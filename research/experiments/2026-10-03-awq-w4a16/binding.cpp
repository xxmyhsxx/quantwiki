#include <torch/extension.h>
#include "quantization_new/gemm/gemm_cuda.h"
#include "quantization_new/gemv/gemv_cuda.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_forward_cuda_new", &gemm_forward_cuda_new);
    m.def("gemv_forward_cuda_new", &gemv_forward_cuda_new);
}
