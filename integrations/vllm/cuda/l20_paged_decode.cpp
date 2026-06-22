#include <torch/extension.h>

torch::Tensor l20_paged_decode_cuda(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor block_table,
    torch::Tensor seq_lens);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("paged_decode", &l20_paged_decode_cuda);
}
