#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cmath>
#include <cstdint>
#include <limits>

namespace {

constexpr int kHeadDim = 128;
constexpr int kPageSize = 16;
constexpr int kMaxSplits = 64;

void check_cuda_contiguous(
    const torch::Tensor& tensor,
    const c10::Device& device,
    const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(
      tensor.device() == device,
      name,
      " must be on the same CUDA device as query");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_query_cache_seq_inputs(
    const torch::Tensor& query,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    const torch::Tensor& seq_lens,
    at::ScalarType cache_dtype) {
  TORCH_CHECK(query.is_cuda(), "query must be a CUDA tensor");
  TORCH_CHECK(query.is_contiguous(), "query must be contiguous");
  TORCH_CHECK(query.scalar_type() == at::kHalf, "query must be float16");
  TORCH_CHECK(
      query.dim() == 3 && query.size(0) > 0 && query.size(1) > 0 &&
          query.size(2) == kHeadDim,
      "query must have shape [batch, num_q_heads, 128] with nonzero batch/heads");

  const auto device = query.device();
  check_cuda_contiguous(key_cache, device, "key_cache");
  check_cuda_contiguous(value_cache, device, "value_cache");
  check_cuda_contiguous(seq_lens, device, "seq_lens");

  TORCH_CHECK(
      key_cache.scalar_type() == cache_dtype &&
          value_cache.scalar_type() == cache_dtype,
      "key_cache and value_cache have the wrong dtype");
  TORCH_CHECK(
      key_cache.dim() == 4 && key_cache.size(0) > 0 &&
          key_cache.size(1) == kPageSize && key_cache.size(2) > 0 &&
          key_cache.size(3) == kHeadDim,
      "K/V cache must have shape [num_pages, 16, num_kv_heads, 128]");
  TORCH_CHECK(key_cache.sizes() == value_cache.sizes(), "K/V cache mismatch");
  TORCH_CHECK(
      query.size(1) % key_cache.size(2) == 0,
      "num_q_heads must be divisible by num_kv_heads");
  TORCH_CHECK(
      query.size(0) <= std::numeric_limits<int>::max() &&
          query.size(1) <= std::numeric_limits<int>::max() &&
          key_cache.size(0) <= std::numeric_limits<int>::max() &&
          key_cache.size(2) <= std::numeric_limits<int>::max(),
      "batch, head counts, and cache page count must fit the CUDA kernel "
      "integer parameters");
  TORCH_CHECK(
      seq_lens.scalar_type() == at::kInt &&
          seq_lens.dim() == 1 &&
          seq_lens.size(0) == query.size(0),
      "seq_lens must be int32 [batch]");
}

void check_paged_decode_inputs(
    const torch::Tensor& query,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    const torch::Tensor& block_table,
    const torch::Tensor& seq_lens,
    at::ScalarType cache_dtype) {
  check_query_cache_seq_inputs(
      query,
      key_cache,
      value_cache,
      seq_lens,
      cache_dtype);
  check_cuda_contiguous(block_table, query.device(), "block_table");
  TORCH_CHECK(
      block_table.scalar_type() == at::kInt &&
          block_table.dim() == 2 &&
          block_table.size(0) == query.size(0) &&
          block_table.size(1) > 0,
      "block_table must be int32 [batch, max_pages]");
  TORCH_CHECK(
      block_table.size(1) <=
          std::numeric_limits<int>::max() / kPageSize,
      "block_table capacity exceeds the CUDA kernel integer range");
}

int check_split_config(int64_t max_seq_len, int64_t split_size) {
  TORCH_CHECK(max_seq_len > 0, "max_seq_len must be positive");
  TORCH_CHECK(
      max_seq_len <= std::numeric_limits<int>::max(),
      "max_seq_len exceeds the CUDA kernel integer range");
  TORCH_CHECK(
      split_size >= 64 && split_size <= 1024 &&
          split_size % kPageSize == 0,
      "split_size must be a multiple of 16 from 64 through 1024");
  const int64_t num_splits =
      (max_seq_len + split_size - 1) / split_size;
  TORCH_CHECK(
      num_splits >= 1 && num_splits <= kMaxSplits,
      "num_splits must be in [1, 64]");
  return static_cast<int>(num_splits);
}

void check_split_workspaces(
    const torch::Tensor& query,
    const torch::Tensor& partial_output,
    const torch::Tensor& partial_max,
    const torch::Tensor& partial_sum,
    const torch::Tensor& output,
    int num_splits) {
  const auto device = query.device();
  const int64_t partial_rows =
      query.size(0) * query.size(1) * num_splits;
  check_cuda_contiguous(partial_output, device, "partial_output");
  check_cuda_contiguous(partial_max, device, "partial_max");
  check_cuda_contiguous(partial_sum, device, "partial_sum");
  check_cuda_contiguous(output, device, "output");
  TORCH_CHECK(
      partial_output.scalar_type() == at::kHalf &&
          partial_output.numel() >= partial_rows * kHeadDim,
      "partial_output must be a contiguous float16 workspace with at least "
      "batch*num_q_heads*num_splits*128 elements");
  TORCH_CHECK(
      partial_max.scalar_type() == at::kFloat &&
          partial_max.numel() >= partial_rows,
      "partial_max must be a contiguous float32 workspace with at least "
      "batch*num_q_heads*num_splits elements");
  TORCH_CHECK(
      partial_sum.scalar_type() == at::kFloat &&
          partial_sum.numel() >= partial_rows,
      "partial_sum must be a contiguous float32 workspace with at least "
      "batch*num_q_heads*num_splits elements");
  TORCH_CHECK(
      output.scalar_type() == at::kHalf &&
          output.sizes() == query.sizes(),
      "output must be float16 with the same shape as query");
}

__inline__ __device__ float warp_sum(float value) {
  for (int offset = 16; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__inline__ __device__ float fp8_e4m3fn_to_float(uint8_t value) {
  const __half_raw half_value = __nv_cvt_fp8_to_halfraw(
      static_cast<__nv_fp8_storage_t>(value),
      __NV_E4M3);
  return __half2float(static_cast<half>(half_value));
}

__inline__ __device__ float2 fp8x2_e4m3fn_to_float2(const uint8_t* values) {
  const auto packed = static_cast<__nv_fp8x2_storage_t>(
      static_cast<uint16_t>(values[0]) |
      (static_cast<uint16_t>(values[1]) << 8));
  const __half2_raw half_pair = __nv_cvt_fp8x2_to_halfraw2(packed, __NV_E4M3);
  return __half22float2(static_cast<half2>(half_pair));
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
    int num_pages,
    int page_size,
    int max_pages) {
  const int batch = blockIdx.y;
  const int q_head = blockIdx.x;
  const int kv_head = q_head / (num_q_heads / num_kv_heads);
  const int thread = threadIdx.x;
  const int lane = thread & 31;
  const int warp = thread >> 5;
  __shared__ float scores[16];
  __shared__ float probabilities[16];
  __shared__ float alpha_shared;
  __shared__ float running_max_shared;
  __shared__ float running_sum_shared;
  __shared__ int physical_page_shared;

  const int64_t q_base =
      (static_cast<int64_t>(batch) * num_q_heads + q_head) * kHeadDim;
  const int pair0 = lane * 2;
  const int pair1 = pair0 + 64;
  const half2 q01 = *reinterpret_cast<const half2*>(query + q_base + pair0);
  const half2 q23 = *reinterpret_cast<const half2*>(query + q_base + pair1);
  const float2 q01f = __half22float2(q01);
  const float2 q23f = __half22float2(q23);
  float2 accumulator = make_float2(0.0f, 0.0f);
  const int seq_len = max(0, min(seq_lens[batch], max_pages * page_size));
  if (seq_len == 0) {
    if (thread < 64) {
      *reinterpret_cast<half2*>(
          output + q_base + thread * 2) =
          __floats2half2_rn(0.0f, 0.0f);
    }
    return;
  }
  if (thread == 0) {
    running_max_shared = -INFINITY;
    running_sum_shared = 0.0f;
  }
  __syncthreads();

  for (int tile_start = 0; tile_start < seq_len; tile_start += 16) {
    if (thread == 0) {
      const int logical_page = tile_start / page_size;
      const int physical_page =
          block_table[
              static_cast<int64_t>(batch) * max_pages + logical_page];
      physical_page_shared =
          physical_page >= 0 && physical_page < num_pages ? physical_page : -1;
    }
    __syncthreads();
    if (physical_page_shared < 0) {
      continue;
    }
#pragma unroll
    for (int warp_token = 0; warp_token < 2; ++warp_token) {
      const int token_index = warp + warp_token * 8;
      const int token = tile_start + token_index;
      float dot = 0.0f;
      if (token < seq_len) {
        const int page_offset = token - tile_start;
        const int64_t cache_base =
            ((static_cast<int64_t>(physical_page_shared) * page_size +
              page_offset) *
                 num_kv_heads +
             kv_head) *
            kHeadDim;
        const half2 k01 =
            *reinterpret_cast<const half2*>(key_cache + cache_base + pair0);
        const half2 k23 =
            *reinterpret_cast<const half2*>(key_cache + cache_base + pair1);
        const float2 k01f = __half22float2(k01);
        const float2 k23f = __half22float2(k23);
        dot = q01f.x * k01f.x + q01f.y * k01f.y +
              q23f.x * k23f.x + q23f.y * k23f.y;
      }
      dot = warp_sum(dot);
      if (lane == 0) {
        scores[token_index] = token < seq_len
            ? dot * 0.08838834764831845f
            : -INFINITY;
      }
    }
    __syncthreads();
    if (thread == 0) {
      float tile_max = scores[0];
#pragma unroll
      for (int index = 1; index < 16; ++index) {
        tile_max = fmaxf(tile_max, scores[index]);
      }
      const float next_max = fmaxf(running_max_shared, tile_max);
      alpha_shared = expf(running_max_shared - next_max);
      float tile_sum = 0.0f;
#pragma unroll
      for (int index = 0; index < 16; ++index) {
        probabilities[index] = expf(scores[index] - next_max);
        tile_sum += probabilities[index];
      }
      running_sum_shared = running_sum_shared * alpha_shared + tile_sum;
      running_max_shared = next_max;
    }
    __syncthreads();
    if (thread < 64) {
      accumulator.x *= alpha_shared;
      accumulator.y *= alpha_shared;
#pragma unroll
      for (int index = 0; index < 16; ++index) {
        const int value_token = tile_start + index;
        if (value_token < seq_len) {
          const int64_t cache_offset =
              ((static_cast<int64_t>(physical_page_shared) * page_size +
                index) *
                   num_kv_heads +
               kv_head) *
                  kHeadDim +
              thread * 2;
          const half2 value =
              *reinterpret_cast<const half2*>(value_cache + cache_offset);
          const float2 value_float = __half22float2(value);
          accumulator.x += probabilities[index] * value_float.x;
          accumulator.y += probabilities[index] * value_float.y;
        }
      }
    }
    __syncthreads();
  }
  if (thread < 64) {
    const float inverse_sum =
        running_sum_shared > 0.0f ? 1.0f / running_sum_shared : 0.0f;
    const half2 result = __floats2half2_rn(
        accumulator.x * inverse_sum,
        accumulator.y * inverse_sum);
    *reinterpret_cast<half2*>(
        output + q_base + thread * 2) = result;
  }
}

__global__ void paged_decode_partial_kernel(
    const half* query,
    const half* key_cache,
    const half* value_cache,
    const int* block_table,
    const int* seq_lens,
    half* partial_output,
    float* partial_max,
    float* partial_sum,
    int num_q_heads,
    int num_kv_heads,
    int num_pages,
    int page_size,
    int max_pages,
    int num_splits,
    int split_size,
    int max_seq_len,
    const int* page_indptr,
    bool use_indptr,
    int page_indices_count) {
  const int q_head = blockIdx.x;
  const int split = blockIdx.y;
  const int batch = blockIdx.z;
  const int kv_head = q_head / (num_q_heads / num_kv_heads);
  const int thread = threadIdx.x;
  const int lane = thread & 31;
  const int warp = thread >> 5;
  const int split_start = split * split_size;
  const int page_start = use_indptr
      ? max(0, min(page_indptr[batch], page_indices_count))
      : 0;
  const int page_end = use_indptr
      ? max(page_start, min(page_indptr[batch + 1], page_indices_count))
      : max_pages;
  const int sequence_capacity = (page_end - page_start) * page_size;
  const int safe_seq_len =
      max(0, min(seq_lens[batch], min(max_seq_len, sequence_capacity)));
  const int split_end = min(split_start + split_size, safe_seq_len);
  __shared__ float scores[16];
  __shared__ float probabilities[16];
  __shared__ float alpha_shared;
  __shared__ float running_max_shared;
  __shared__ float running_sum_shared;
  __shared__ int physical_page_shared;

  const int64_t q_base =
      (static_cast<int64_t>(batch) * num_q_heads + q_head) * kHeadDim;
  const int pair0 = lane * 2;
  const int pair1 = pair0 + 64;
  const float2 q01f = __half22float2(
      *reinterpret_cast<const half2*>(query + q_base + pair0));
  const float2 q23f = __half22float2(
      *reinterpret_cast<const half2*>(query + q_base + pair1));
  float2 accumulator = make_float2(0.0f, 0.0f);
  if (thread == 0) {
    running_max_shared = -INFINITY;
    running_sum_shared = 0.0f;
  }
  __syncthreads();

  for (int tile_start = split_start; tile_start < split_end; tile_start += 16) {
    if (thread == 0) {
      const int logical_page = tile_start / page_size;
      const int64_t page_index = use_indptr
          ? static_cast<int64_t>(page_start) + logical_page
          : static_cast<int64_t>(batch) * max_pages + logical_page;
      const int physical_page = block_table[page_index];
      physical_page_shared =
          physical_page >= 0 && physical_page < num_pages ? physical_page : -1;
    }
    __syncthreads();
    if (physical_page_shared < 0) {
      continue;
    }
#pragma unroll
    for (int warp_token = 0; warp_token < 2; ++warp_token) {
      const int token_index = warp + warp_token * 8;
      const int token = tile_start + token_index;
      float dot = 0.0f;
      if (token < split_end) {
        const int page_offset = token - tile_start;
        const int64_t cache_base =
            ((static_cast<int64_t>(physical_page_shared) * page_size +
              page_offset) *
                 num_kv_heads +
             kv_head) *
            kHeadDim;
        const float2 k01f = __half22float2(
            *reinterpret_cast<const half2*>(key_cache + cache_base + pair0));
        const float2 k23f = __half22float2(
            *reinterpret_cast<const half2*>(key_cache + cache_base + pair1));
        dot = q01f.x * k01f.x + q01f.y * k01f.y +
              q23f.x * k23f.x + q23f.y * k23f.y;
      }
      dot = warp_sum(dot);
      if (lane == 0) {
        scores[token_index] = token < split_end
            ? dot * 0.08838834764831845f
            : -INFINITY;
      }
    }
    __syncthreads();
    if (thread == 0) {
      float tile_max = scores[0];
#pragma unroll
      for (int index = 1; index < 16; ++index) {
        tile_max = fmaxf(tile_max, scores[index]);
      }
      const float next_max = fmaxf(running_max_shared, tile_max);
      alpha_shared = expf(running_max_shared - next_max);
      float tile_sum = 0.0f;
#pragma unroll
      for (int index = 0; index < 16; ++index) {
        probabilities[index] = expf(scores[index] - next_max);
        tile_sum += probabilities[index];
      }
      running_sum_shared = running_sum_shared * alpha_shared + tile_sum;
      running_max_shared = next_max;
    }
    __syncthreads();
    if (thread < 64) {
      accumulator.x *= alpha_shared;
      accumulator.y *= alpha_shared;
#pragma unroll
      for (int index = 0; index < 16; ++index) {
        const int token = tile_start + index;
        if (token < split_end) {
          const int64_t cache_offset =
              ((static_cast<int64_t>(physical_page_shared) * page_size +
                index) *
                   num_kv_heads +
               kv_head) *
                  kHeadDim +
              thread * 2;
          const float2 value = __half22float2(
              *reinterpret_cast<const half2*>(value_cache + cache_offset));
          accumulator.x += probabilities[index] * value.x;
          accumulator.y += probabilities[index] * value.y;
        }
      }
    }
    __syncthreads();
  }

  const int64_t partial_index =
      ((static_cast<int64_t>(batch) * num_q_heads + q_head) * num_splits +
       split);
  if (thread < 64) {
    *reinterpret_cast<half2*>(
        partial_output + partial_index * kHeadDim + thread * 2) =
        __floats2half2_rn(accumulator.x, accumulator.y);
  }
  if (thread == 0) {
    partial_max[partial_index] = running_max_shared;
    partial_sum[partial_index] = running_sum_shared;
  }
}

__global__ void paged_decode_fp8_e4m3_partial_kernel(
    const half* query,
    const uint8_t* key_cache,
    const uint8_t* value_cache,
    const int* block_table,
    const int* seq_lens,
    half* partial_output,
    float* partial_max,
    float* partial_sum,
    float k_scale,
    float v_scale,
    int num_q_heads,
    int num_kv_heads,
    int num_pages,
    int page_size,
    int max_pages,
    int num_splits,
    int split_size,
    int max_seq_len) {
  const int q_head = blockIdx.x;
  const int split = blockIdx.y;
  const int batch = blockIdx.z;
  const int kv_head = q_head / (num_q_heads / num_kv_heads);
  const int thread = threadIdx.x;
  const int lane = thread & 31;
  const int warp = thread >> 5;
  const int split_start = split * split_size;
  const int sequence_capacity = max_pages * page_size;
  const int safe_seq_len =
      max(0, min(seq_lens[batch], min(max_seq_len, sequence_capacity)));
  const int split_end = min(split_start + split_size, safe_seq_len);
  __shared__ float scores[16];
  __shared__ float probabilities[16];
  __shared__ float alpha_shared;
  __shared__ float running_max_shared;
  __shared__ float running_sum_shared;
  __shared__ int physical_page_shared;

  const int64_t q_base =
      (static_cast<int64_t>(batch) * num_q_heads + q_head) * kHeadDim;
  const int pair0 = lane * 2;
  const int pair1 = pair0 + 64;
  const float2 q01f = __half22float2(
      *reinterpret_cast<const half2*>(query + q_base + pair0));
  const float2 q23f = __half22float2(
      *reinterpret_cast<const half2*>(query + q_base + pair1));
  float2 accumulator = make_float2(0.0f, 0.0f);
  if (thread == 0) {
    running_max_shared = -INFINITY;
    running_sum_shared = 0.0f;
  }
  __syncthreads();

  for (int tile_start = split_start; tile_start < split_end; tile_start += 16) {
    if (thread == 0) {
      const int logical_page = tile_start / page_size;
      const int physical_page =
          block_table[
              static_cast<int64_t>(batch) * max_pages + logical_page];
      physical_page_shared =
          physical_page >= 0 && physical_page < num_pages ? physical_page : -1;
    }
    __syncthreads();
    if (physical_page_shared < 0) {
      continue;
    }
#pragma unroll
    for (int warp_token = 0; warp_token < 2; ++warp_token) {
      const int token_index = warp + warp_token * 8;
      const int token = tile_start + token_index;
      float dot = 0.0f;
      if (token < split_end) {
        const int page_offset = token - tile_start;
        const int64_t cache_base =
            ((static_cast<int64_t>(physical_page_shared) * page_size +
              page_offset) *
                 num_kv_heads +
             kv_head) *
            kHeadDim;
        const float2 k01 =
            fp8x2_e4m3fn_to_float2(key_cache + cache_base + pair0);
        const float2 k23 =
            fp8x2_e4m3fn_to_float2(key_cache + cache_base + pair1);
        dot = (q01f.x * k01.x + q01f.y * k01.y +
               q23f.x * k23.x + q23f.y * k23.y) *
              k_scale;
      }
      dot = warp_sum(dot);
      if (lane == 0) {
        scores[token_index] = token < split_end
            ? dot * 0.08838834764831845f
            : -INFINITY;
      }
    }
    __syncthreads();
    if (thread == 0) {
      float tile_max = scores[0];
#pragma unroll
      for (int index = 1; index < 16; ++index) {
        tile_max = fmaxf(tile_max, scores[index]);
      }
      const float next_max = fmaxf(running_max_shared, tile_max);
      alpha_shared = expf(running_max_shared - next_max);
      float tile_sum = 0.0f;
#pragma unroll
      for (int index = 0; index < 16; ++index) {
        probabilities[index] = expf(scores[index] - next_max);
        tile_sum += probabilities[index];
      }
      running_sum_shared = running_sum_shared * alpha_shared + tile_sum;
      running_max_shared = next_max;
    }
    __syncthreads();
    if (thread < 64) {
      accumulator.x *= alpha_shared;
      accumulator.y *= alpha_shared;
#pragma unroll
      for (int index = 0; index < 16; ++index) {
        const int token = tile_start + index;
        if (token < split_end) {
          const int64_t cache_offset =
              ((static_cast<int64_t>(physical_page_shared) * page_size +
                index) *
                   num_kv_heads +
               kv_head) *
                  kHeadDim +
              thread * 2;
          const float2 value =
              fp8x2_e4m3fn_to_float2(value_cache + cache_offset);
          accumulator.x += probabilities[index] * value.x * v_scale;
          accumulator.y += probabilities[index] * value.y * v_scale;
        }
      }
    }
    __syncthreads();
  }

  const int64_t partial_index =
      ((static_cast<int64_t>(batch) * num_q_heads + q_head) * num_splits +
       split);
  if (thread < 64) {
    *reinterpret_cast<half2*>(
        partial_output + partial_index * kHeadDim + thread * 2) =
        __floats2half2_rn(accumulator.x, accumulator.y);
  }
  if (thread == 0) {
    partial_max[partial_index] = running_max_shared;
    partial_sum[partial_index] = running_sum_shared;
  }
}

__global__ void paged_decode_merge_kernel(
    const half* partial_output,
    const float* partial_max,
    const float* partial_sum,
    half* output,
    int num_q_heads,
    int num_splits) {
  const int q_head = blockIdx.x;
  const int batch = blockIdx.y;
  const int pair = threadIdx.x;
  const int64_t base =
      (static_cast<int64_t>(batch) * num_q_heads + q_head) * num_splits;
  __shared__ float denominator;
  __shared__ float corrections[64];
  if (pair == 0) {
    float max_value = partial_max[base];
    for (int split = 1; split < num_splits; ++split) {
      max_value = fmaxf(max_value, partial_max[base + split]);
    }
    float sum = 0.0f;
    if (isfinite(max_value)) {
      for (int split = 0; split < num_splits; ++split) {
        corrections[split] = expf(partial_max[base + split] - max_value);
        sum += partial_sum[base + split] * corrections[split];
      }
    } else {
      for (int split = 0; split < num_splits; ++split) {
        corrections[split] = 0.0f;
      }
    }
    denominator = sum > 0.0f ? sum : 1.0f;
  }
  __syncthreads();
  float2 numerator = make_float2(0.0f, 0.0f);
  for (int split = 0; split < num_splits; ++split) {
    const float2 partial = __half22float2(
        *reinterpret_cast<const half2*>(
            partial_output + (base + split) * kHeadDim + pair * 2));
    numerator.x += partial.x * corrections[split];
    numerator.y += partial.y * corrections[split];
  }
  *reinterpret_cast<half2*>(
      output +
      (static_cast<int64_t>(batch) * num_q_heads + q_head) * kHeadDim +
      pair * 2) =
      __floats2half2_rn(numerator.x / denominator, numerator.y / denominator);
}

}  // namespace

torch::Tensor l20_paged_decode_cuda(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor block_table,
    torch::Tensor seq_lens) {
  check_paged_decode_inputs(
      query,
      key_cache,
      value_cache,
      block_table,
      seq_lens,
      at::kHalf);
  const at::cuda::CUDAGuard guard(query.device());
  auto output = torch::empty_like(query);
  const dim3 grid(query.size(1), query.size(0));
  paged_decode_kernel<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
      block_table.data_ptr<int>(),
      seq_lens.data_ptr<int>(),
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      query.size(1),
      key_cache.size(2),
      key_cache.size(0),
      key_cache.size(1),
      block_table.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

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

torch::Tensor l20_paged_decode_split_cuda(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor block_table,
    torch::Tensor seq_lens,
    int64_t max_seq_len,
    int64_t split_size) {
  check_paged_decode_inputs(
      query,
      key_cache,
      value_cache,
      block_table,
      seq_lens,
      at::kHalf);
  TORCH_CHECK(
      max_seq_len <= block_table.size(1) * kPageSize,
      "max_seq_len exceeds block_table capacity");
  const int num_splits = check_split_config(max_seq_len, split_size);
  auto partial_output = torch::empty(
      {query.size(0), query.size(1), num_splits, kHeadDim},
      query.options());
  auto float_options = query.options().dtype(torch::kFloat32);
  auto partial_max =
      torch::empty({query.size(0), query.size(1), num_splits}, float_options);
  auto partial_sum = torch::empty_like(partial_max);
  auto output = torch::empty_like(query);
  l20_paged_decode_split_out_cuda(
      query,
      key_cache,
      value_cache,
      block_table,
      seq_lens,
      partial_output,
      partial_max,
      partial_sum,
      output,
      max_seq_len,
      split_size);
  return output;
}

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
    int64_t split_size) {
  check_paged_decode_inputs(
      query,
      key_cache,
      value_cache,
      block_table,
      seq_lens,
      at::kHalf);
  TORCH_CHECK(
      max_seq_len <= block_table.size(1) * kPageSize,
      "max_seq_len exceeds block_table capacity");
  const int num_splits = check_split_config(max_seq_len, split_size);
  check_split_workspaces(
      query,
      partial_output,
      partial_max,
      partial_sum,
      output,
      num_splits);
  const at::cuda::CUDAGuard guard(query.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 partial_grid(query.size(1), num_splits, query.size(0));
  paged_decode_partial_kernel<<<partial_grid, 256, 0, stream>>>(
      reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
      block_table.data_ptr<int>(),
      seq_lens.data_ptr<int>(),
      reinterpret_cast<half*>(partial_output.data_ptr<at::Half>()),
      partial_max.data_ptr<float>(),
      partial_sum.data_ptr<float>(),
      query.size(1),
      key_cache.size(2),
      key_cache.size(0),
      key_cache.size(1),
      block_table.size(1),
      num_splits,
      split_size,
      static_cast<int>(max_seq_len),
      nullptr,
      false,
      0);
  paged_decode_merge_kernel<<<
      dim3(query.size(1), query.size(0)),
      kHeadDim / 2,
      0,
      stream>>>(
      reinterpret_cast<const half*>(partial_output.data_ptr<at::Half>()),
      partial_max.data_ptr<float>(),
      partial_sum.data_ptr<float>(),
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      query.size(1),
      num_splits);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void l20_paged_decode_split_indices_out_cuda(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor page_indptr,
    torch::Tensor page_indices,
    torch::Tensor seq_lens,
    torch::Tensor partial_output,
    torch::Tensor partial_max,
    torch::Tensor partial_sum,
    torch::Tensor output,
    int64_t max_seq_len,
    int64_t split_size) {
  check_query_cache_seq_inputs(
      query,
      key_cache,
      value_cache,
      seq_lens,
      at::kHalf);
  check_cuda_contiguous(page_indptr, query.device(), "page_indptr");
  check_cuda_contiguous(page_indices, query.device(), "page_indices");
  TORCH_CHECK(
      page_indptr.scalar_type() == at::kInt &&
          page_indptr.dim() == 1 &&
          page_indptr.size(0) == query.size(0) + 1,
      "page_indptr must be int32 [batch + 1]");
  TORCH_CHECK(
      page_indices.scalar_type() == at::kInt && page_indices.dim() == 1,
      "page_indices must be a one-dimensional int32 tensor");
  TORCH_CHECK(
      page_indices.numel() <=
          std::numeric_limits<int>::max() / kPageSize,
      "page_indices capacity exceeds the CUDA kernel integer range");
  const int num_splits = check_split_config(max_seq_len, split_size);
  check_split_workspaces(
      query,
      partial_output,
      partial_max,
      partial_sum,
      output,
      num_splits);
  const at::cuda::CUDAGuard guard(query.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  paged_decode_partial_kernel<<<
      dim3(query.size(1), num_splits, query.size(0)),
      256,
      0,
      stream>>>(
      reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
      page_indices.data_ptr<int>(),
      seq_lens.data_ptr<int>(),
      reinterpret_cast<half*>(partial_output.data_ptr<at::Half>()),
      partial_max.data_ptr<float>(),
      partial_sum.data_ptr<float>(),
      query.size(1),
      key_cache.size(2),
      key_cache.size(0),
      key_cache.size(1),
      0,
      num_splits,
      split_size,
      static_cast<int>(max_seq_len),
      page_indptr.data_ptr<int>(),
      true,
      static_cast<int>(page_indices.numel()));
  paged_decode_merge_kernel<<<
      dim3(query.size(1), query.size(0)),
      kHeadDim / 2,
      0,
      stream>>>(
      reinterpret_cast<const half*>(partial_output.data_ptr<at::Half>()),
      partial_max.data_ptr<float>(),
      partial_sum.data_ptr<float>(),
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      query.size(1),
      num_splits);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void l20_paged_decode_fp8_e4m3_split_out_cuda(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor block_table,
    torch::Tensor seq_lens,
    torch::Tensor partial_output,
    torch::Tensor partial_max,
    torch::Tensor partial_sum,
    torch::Tensor output,
    double k_scale,
    double v_scale,
    int64_t max_seq_len,
    int64_t split_size) {
  check_paged_decode_inputs(
      query,
      key_cache,
      value_cache,
      block_table,
      seq_lens,
      at::kFloat8_e4m3fn);
  TORCH_CHECK(
      max_seq_len <= block_table.size(1) * kPageSize,
      "max_seq_len exceeds block_table capacity");
  TORCH_CHECK(
      std::isfinite(k_scale) && k_scale > 0.0 &&
          std::isfinite(v_scale) && v_scale > 0.0,
      "k_scale and v_scale must be finite and positive");
  const int num_splits = check_split_config(max_seq_len, split_size);
  check_split_workspaces(
      query,
      partial_output,
      partial_max,
      partial_sum,
      output,
      num_splits);

  const at::cuda::CUDAGuard guard(query.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  paged_decode_fp8_e4m3_partial_kernel<<<
      dim3(query.size(1), num_splits, query.size(0)),
      256,
      0,
      stream>>>(
      reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
      reinterpret_cast<const uint8_t*>(key_cache.data_ptr()),
      reinterpret_cast<const uint8_t*>(value_cache.data_ptr()),
      block_table.data_ptr<int>(),
      seq_lens.data_ptr<int>(),
      reinterpret_cast<half*>(partial_output.data_ptr<at::Half>()),
      partial_max.data_ptr<float>(),
      partial_sum.data_ptr<float>(),
      static_cast<float>(k_scale),
      static_cast<float>(v_scale),
      query.size(1),
      key_cache.size(2),
      key_cache.size(0),
      key_cache.size(1),
      block_table.size(1),
      num_splits,
      split_size,
      static_cast<int>(max_seq_len));
  paged_decode_merge_kernel<<<
      dim3(query.size(1), query.size(0)),
      kHeadDim / 2,
      0,
      stream>>>(
      reinterpret_cast<const half*>(partial_output.data_ptr<at::Half>()),
      partial_max.data_ptr<float>(),
      partial_sum.data_ptr<float>(),
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      query.size(1),
      num_splits);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
