#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace {

__device__ __forceinline__ float apply_repetition_penalty(
    float value,
    float penalty) {
  return value > 0.0f ? value / penalty : value * penalty;
}

__global__ void sparse_repetition_penalty_kernel(
    float* logits,
    const int64_t* token_ids,
    const int64_t* lengths,
    int64_t batch,
    int64_t max_tokens,
    int64_t vocab,
    float repetition_penalty) {
  const int64_t total = batch * max_tokens;
  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t row = index / max_tokens;
    const int64_t col = index - row * max_tokens;
    if (col >= lengths[row]) {
      continue;
    }

    const int64_t token = token_ids[index];
    if (token < 0 || token >= vocab) {
      continue;
    }

    float* value = logits + row * vocab + token;
    *value = apply_repetition_penalty(*value, repetition_penalty);
  }
}

}  // namespace

torch::Tensor l20_sparse_repetition_penalty_out_cuda(
    torch::Tensor logits,
    torch::Tensor token_ids,
    torch::Tensor lengths,
    double repetition_penalty) {
  // Contract: every active token_ids row prefix is deduplicated. The vLLM
  // processor constructs this representation before dispatch; direct callers
  // must preserve it to avoid repeated/racing writes to one logit.
  TORCH_CHECK(logits.is_cuda(), "logits must be a CUDA tensor");
  TORCH_CHECK(token_ids.is_cuda(), "token_ids must be a CUDA tensor");
  TORCH_CHECK(lengths.is_cuda(), "lengths must be a CUDA tensor");
  TORCH_CHECK(
      token_ids.device() == logits.device(),
      "token_ids must be on the same CUDA device as logits");
  TORCH_CHECK(
      lengths.device() == logits.device(),
      "lengths must be on the same CUDA device as logits");
  TORCH_CHECK(logits.dim() == 2, "logits must be [batch, vocab]");
  TORCH_CHECK(token_ids.dim() == 2, "token_ids must be [batch, max_tokens]");
  TORCH_CHECK(lengths.dim() == 1, "lengths must be [batch]");
  TORCH_CHECK(
      token_ids.size(0) == logits.size(0),
      "token_ids batch must match logits batch");
  TORCH_CHECK(
      lengths.size(0) == logits.size(0),
      "lengths batch must match logits batch");
  TORCH_CHECK(logits.is_contiguous(), "logits must be contiguous");
  TORCH_CHECK(token_ids.is_contiguous(), "token_ids must be contiguous");
  TORCH_CHECK(lengths.is_contiguous(), "lengths must be contiguous");
  TORCH_CHECK(
      logits.scalar_type() == at::kFloat,
      "sparse_repetition_penalty_out currently requires float32 logits");
  TORCH_CHECK(
      token_ids.scalar_type() == at::kLong,
      "token_ids must have dtype torch.int64");
  TORCH_CHECK(
      lengths.scalar_type() == at::kLong,
      "lengths must have dtype torch.int64");
  TORCH_CHECK(
      std::isfinite(repetition_penalty) && repetition_penalty > 0.0,
      "repetition_penalty must be finite and positive");

  const int64_t batch = logits.size(0);
  const int64_t vocab = logits.size(1);
  const int64_t max_tokens = token_ids.size(1);
  if (batch == 0 || vocab == 0 || max_tokens == 0) {
    return logits;
  }

  c10::cuda::CUDAGuard device_guard(logits.device());
  const int threads = 256;
  const int64_t total = batch * max_tokens;
  const int blocks = std::min<int64_t>(
      4096,
      (total + threads - 1) / threads);

  sparse_repetition_penalty_kernel<<<
      blocks,
      threads,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      logits.data_ptr<float>(),
      token_ids.data_ptr<int64_t>(),
      lengths.data_ptr<int64_t>(),
      batch,
      max_tokens,
      vocab,
      static_cast<float>(repetition_penalty));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return logits;
}
