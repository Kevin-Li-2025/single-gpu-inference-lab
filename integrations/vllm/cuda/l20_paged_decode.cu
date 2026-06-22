#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cmath>

namespace {

__inline__ __device__ float warp_sum(float value) {
  for (int offset = 16; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__global__ void paged_decode_kernel(
    const half* query,
    const half* key_cache,
    const half* value_cache,
    const int* block_table,
    const int* seq_lens,
    half* output,
    int num_q_heads,
    int num_kv_heads,
    int page_size,
    int max_pages) {
  const int batch = blockIdx.y;
  const int q_head = blockIdx.x;
  const int kv_head = q_head / (num_q_heads / num_kv_heads);
  const int dim = threadIdx.x;
  const int lane = dim & 31;
  const int warp = dim >> 5;
  __shared__ float warp_sums[4];
  __shared__ float score_shared;
  __shared__ float alpha_shared;
  __shared__ float beta_shared;
  __shared__ float running_max_shared;
  __shared__ float running_sum_shared;

  const float q = __half2float(
      query[(batch * num_q_heads + q_head) * 128 + dim]);
  float accumulator = 0.0f;
  const int seq_len = seq_lens[batch];
  if (dim == 0) {
    running_max_shared = -INFINITY;
    running_sum_shared = 0.0f;
  }
  __syncthreads();

  for (int token = 0; token < seq_len; ++token) {
    const int logical_page = token / page_size;
    const int page_offset = token - logical_page * page_size;
    const int physical_page = block_table[batch * max_pages + logical_page];
    const int cache_offset =
        ((physical_page * page_size + page_offset) * num_kv_heads + kv_head) *
            128 +
        dim;
    float dot = q * __half2float(key_cache[cache_offset]);
    dot = warp_sum(dot);
    if (lane == 0) {
      warp_sums[warp] = dot;
    }
    __syncthreads();
    if (warp == 0) {
      float block_sum = lane < 4 ? warp_sums[lane] : 0.0f;
      block_sum = warp_sum(block_sum);
      if (lane == 0) {
        score_shared = block_sum * 0.08838834764831845f;
      }
    }
    __syncthreads();
    if (dim == 0) {
      const float next_max = fmaxf(running_max_shared, score_shared);
      alpha_shared = expf(running_max_shared - next_max);
      beta_shared = expf(score_shared - next_max);
      running_sum_shared =
          running_sum_shared * alpha_shared + beta_shared;
      running_max_shared = next_max;
    }
    __syncthreads();
    accumulator = accumulator * alpha_shared +
                  beta_shared * __half2float(value_cache[cache_offset]);
    __syncthreads();
  }
  output[(batch * num_q_heads + q_head) * 128 + dim] =
      __float2half(accumulator / running_sum_shared);
}

}  // namespace

torch::Tensor l20_paged_decode_cuda(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor block_table,
    torch::Tensor seq_lens) {
  TORCH_CHECK(query.is_cuda(), "query must be CUDA");
  TORCH_CHECK(query.scalar_type() == torch::kFloat16, "FP16 only");
  TORCH_CHECK(query.dim() == 3 && query.size(2) == 128, "Q must be [B,H,128]");
  TORCH_CHECK(key_cache.dim() == 4 && key_cache.size(3) == 128, "NHD cache only");
  TORCH_CHECK(key_cache.sizes() == value_cache.sizes(), "K/V cache mismatch");
  TORCH_CHECK(block_table.scalar_type() == torch::kInt32, "int32 block table");
  TORCH_CHECK(seq_lens.scalar_type() == torch::kInt32, "int32 sequence lengths");
  const at::cuda::CUDAGuard guard(query.device());
  auto output = torch::empty_like(query);
  const dim3 grid(query.size(1), query.size(0));
  paged_decode_kernel<<<grid, 128, 0, at::cuda::getDefaultCUDAStream()>>>(
      reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
      block_table.data_ptr<int>(),
      seq_lens.data_ptr<int>(),
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      query.size(1),
      key_cache.size(2),
      key_cache.size(1),
      block_table.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
