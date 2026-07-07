#include <torch/extension.h>
#include <torch/library.h>

torch::Tensor l20_sparse_repetition_penalty_out_cuda(
    torch::Tensor logits,
    torch::Tensor token_ids,
    torch::Tensor lengths,
    double repetition_penalty);

torch::Tensor l20_sparse_repetition_penalty_out_dispatch(
    torch::Tensor logits,
    torch::Tensor token_ids,
    torch::Tensor lengths,
    double repetition_penalty) {
  return l20_sparse_repetition_penalty_out_cuda(
      logits, token_ids, lengths, repetition_penalty);
}

TORCH_LIBRARY_FRAGMENT(l20_stack, module) {
  module.def(
      "sparse_repetition_penalty_out("
      "Tensor(a!) logits, Tensor token_ids, Tensor lengths, "
      "float repetition_penalty) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(l20_stack, CUDA, module) {
  module.impl(
      "sparse_repetition_penalty_out",
      &l20_sparse_repetition_penalty_out_dispatch);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "sparse_repetition_penalty_out",
      &l20_sparse_repetition_penalty_out_cuda);
}
