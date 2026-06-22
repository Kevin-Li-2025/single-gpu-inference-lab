#include <torch/extension.h>

torch::Tensor l20_paged_decode_cuda(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor block_table,
    torch::Tensor seq_lens);

torch::Tensor l20_paged_decode_split_cuda(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor block_table,
    torch::Tensor seq_lens,
    int64_t max_seq_len,
    int64_t split_size);

void l20_paged_decode_split_out_cuda(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor block_table,
    torch::Tensor seq_lens,
    torch::Tensor partial_output,
    torch::Tensor partial_max,
    torch::Tensor partial_sum,
    torch::Tensor output,
    int64_t max_seq_len,
    int64_t split_size);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("paged_decode", &l20_paged_decode_cuda);
  module.def("paged_decode_split", &l20_paged_decode_split_cuda);
  module.def("paged_decode_split_out", &l20_paged_decode_split_out_cuda);
}
