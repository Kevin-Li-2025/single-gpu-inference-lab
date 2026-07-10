// Real-GGUF Q4_K x Q8_K kernel validation for Apple M4.
//
// This program parses a GGUF v2/v3 file without a model framework, mmaps one
// real Q4_K tensor, and compares a self-written NEON dot kernel against the
// exported llama.cpp kernel on identical bytes.

#include <algorithm>
#include <arm_neon.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "m4_gguf_q4k.hpp"

namespace {

using Clock = std::chrono::steady_clock;
using m4q4k::BlockQ4K;
using m4q4k::BlockQ8K;
using m4q4k::GgufIndex;
using m4q4k::MappedFile;
using m4q4k::TensorInfo;
using m4q4k::fp16_to_fp32;
using m4q4k::kGgmlTypeQ4K;
using m4q4k::kQkK;
using m4q4k::kScaleBytes;
using m4q4k::parse_gguf;

struct Config {
  std::string model;
  std::string tensor = "blk.0.ffn_down.weight";
  std::string ggml_cpu_library;
  int warmup = 5;
  int iterations = 20;
  int cache_flush_mib = 64;
  int row_limit = 0;
  std::uint64_t seed = 7;
  bool list = false;
};

struct XorShift64 {
  std::uint64_t state;
  explicit XorShift64(std::uint64_t seed) : state(seed ? seed : 1) {}
  std::uint32_t next() {
    std::uint64_t x = state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    state = x;
    return static_cast<std::uint32_t>(x >> 32);
  }
  float uniform() {
    return static_cast<float>(next()) / 2147483647.5f - 1.0f;
  }
};

void quantize_q8k(const float* input, int count, BlockQ8K* output) {
  if (count % kQkK != 0) {
    throw std::runtime_error("Q8_K input size must be divisible by 256");
  }
  for (int block = 0; block < count / kQkK; ++block) {
    const float* source = input + block * kQkK;
    BlockQ8K& target = output[block];
    float max_value = 0.0f;
    float max_abs = 0.0f;
    for (int i = 0; i < kQkK; ++i) {
      const float magnitude = std::fabs(source[i]);
      if (magnitude > max_abs) {
        max_abs = magnitude;
        max_value = source[i];
      }
    }
    if (max_abs == 0.0f) {
      std::memset(&target, 0, sizeof(target));
      continue;
    }
    const float inverse_scale = -127.0f / max_value;
    for (int i = 0; i < kQkK; ++i) {
      const int value = static_cast<int>(std::nearbyint(source[i] * inverse_scale));
      target.qs[i] = static_cast<std::int8_t>(std::min(127, value));
    }
    for (int group = 0; group < kQkK / 16; ++group) {
      int sum = 0;
      for (int i = 0; i < 16; ++i) {
        sum += target.qs[group * 16 + i];
      }
      target.bsums[group] = static_cast<std::int16_t>(sum);
    }
    target.d = 1.0f / inverse_scale;
  }
}

void decode_scale_min(const std::uint8_t* packed, int group, std::uint8_t& scale,
                      std::uint8_t& minimum) {
  if (group < 4) {
    scale = packed[group] & 63;
    minimum = packed[group + 4] & 63;
  } else {
    scale = (packed[group + 4] & 0x0f) | ((packed[group - 4] >> 6) << 4);
    minimum = (packed[group + 4] >> 4) | ((packed[group] >> 6) << 4);
  }
}

float dot_q4k_q8k_scalar(int count, const BlockQ4K* weights, const BlockQ8K* input) {
  float total = 0.0f;
  for (int block = 0; block < count / kQkK; ++block) {
    const BlockQ4K& w = weights[block];
    const BlockQ8K& x = input[block];
    const float d = fp16_to_fp32(w.d) * x.d;
    const float dmin = fp16_to_fp32(w.dmin) * x.d;
    int weighted_dot = 0;
    int weighted_min = 0;
    for (int group = 0; group < 8; ++group) {
      std::uint8_t scale = 0;
      std::uint8_t minimum = 0;
      decode_scale_min(w.scales, group, scale, minimum);
      const int chunk = group / 2;
      const bool high = (group & 1) != 0;
      int dot = 0;
      for (int i = 0; i < 32; ++i) {
        const std::uint8_t packed = w.qs[chunk * 32 + i];
        const int q = high ? packed >> 4 : packed & 0x0f;
        dot += q * x.qs[group * 32 + i];
      }
      weighted_dot += static_cast<int>(scale) * dot;
      weighted_min += static_cast<int>(minimum) *
                      (x.bsums[group * 2] + x.bsums[group * 2 + 1]);
    }
    total += d * weighted_dot - dmin * weighted_min;
  }
  return total;
}

float dot_q4k_q8k_m4(int count, const BlockQ4K* weights, const BlockQ8K* input) {
  const uint8x16_t mask = vdupq_n_u8(0x0f);
  constexpr std::uint32_t mask1 = 0x3f3f3f3f;
  constexpr std::uint32_t mask2 = 0x0f0f0f0f;
  constexpr std::uint32_t mask3 = 0x03030303;
  float total = 0.0f;
  for (int block = 0; block < count / kQkK; ++block) {
    const BlockQ4K& w = weights[block];
    const BlockQ8K& x = input[block];
    std::uint32_t decoded[4] = {};
    std::memcpy(decoded, w.scales, kScaleBytes);
    const std::uint32_t packed_mins = decoded[1] & mask1;
    decoded[3] = ((decoded[2] >> 4) & mask2) |
                 (((decoded[1] >> 6) & mask3) << 4);
    decoded[1] = (decoded[2] & mask2) | (((decoded[0] >> 6) & mask3) << 4);
    decoded[2] = packed_mins;
    decoded[0] &= mask1;
    const auto* scales = reinterpret_cast<const std::uint8_t*>(&decoded[0]);
    const auto* mins = reinterpret_cast<const std::uint8_t*>(&decoded[2]);

    const int16x8_t q8_sums = vpaddq_s16(vld1q_s16(x.bsums),
                                          vld1q_s16(x.bsums + 8));
    const int16x8_t min_values = vreinterpretq_s16_u16(
        vmovl_u8(vld1_u8(mins)));
    const int32x4_t min_products = vaddq_s32(
        vmull_s16(vget_low_s16(q8_sums), vget_low_s16(min_values)),
        vmull_s16(vget_high_s16(q8_sums), vget_high_s16(min_values)));
    const int weighted_min = vaddvq_s32(min_products);
    int weighted_low = 0;
    int weighted_high = 0;
    for (int chunk = 0; chunk < 4; ++chunk) {
      const uint8x16x2_t packed = vld1q_u8_x2(w.qs + chunk * 32);
      const int8x16_t low0 = vreinterpretq_s8_u8(vandq_u8(packed.val[0], mask));
      const int8x16_t low1 = vreinterpretq_s8_u8(vandq_u8(packed.val[1], mask));
      const int8x16_t high0 = vreinterpretq_s8_u8(vshrq_n_u8(packed.val[0], 4));
      const int8x16_t high1 = vreinterpretq_s8_u8(vshrq_n_u8(packed.val[1], 4));
      const std::int8_t* q8 = x.qs + chunk * 64;
      int32x4_t low_dot = vdotq_s32(vdupq_n_s32(0), low0, vld1q_s8(q8));
      low_dot = vdotq_s32(low_dot, low1, vld1q_s8(q8 + 16));
      int32x4_t high_dot = vdotq_s32(vdupq_n_s32(0), high0, vld1q_s8(q8 + 32));
      high_dot = vdotq_s32(high_dot, high1, vld1q_s8(q8 + 48));
      weighted_low += vaddvq_s32(low_dot) * scales[chunk * 2];
      weighted_high += vaddvq_s32(high_dot) * scales[chunk * 2 + 1];
    }
    total += fp16_to_fp32(w.d) * x.d * (weighted_low + weighted_high) -
             fp16_to_fp32(w.dmin) * x.d * weighted_min;
  }
  return total;
}

void matvec_q4k_q8k_m4(int count, const BlockQ4K* weights,
                       std::size_t row_stride_bytes, int rows,
                       const BlockQ8K* input, float* output) {
  for (int row = 0; row < rows; ++row) {
    const auto* row_weights = reinterpret_cast<const BlockQ4K*>(
        reinterpret_cast<const std::uint8_t*>(weights) +
        static_cast<std::size_t>(row) * row_stride_bytes);
    output[row] = dot_q4k_q8k_m4(count, row_weights, input);
  }
}

using LlamaDot = void (*)(int, float*, std::size_t, const void*, std::size_t,
                          const void*, std::size_t, int);

class LlamaKernel {
 public:
  explicit LlamaKernel(const std::string& library) {
    handle_ = dlopen(library.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle_) {
      throw std::runtime_error(std::string("cannot load ggml-cpu: ") + dlerror());
    }
    dot_ = reinterpret_cast<LlamaDot>(dlsym(handle_, "ggml_vec_dot_q4_K_q8_K"));
    if (!dot_) {
      throw std::runtime_error("ggml_vec_dot_q4_K_q8_K is not exported");
    }
  }
  ~LlamaKernel() {
    if (handle_) {
      dlclose(handle_);
    }
  }
  float dot(int count, const BlockQ4K* weights, const BlockQ8K* input) const {
    float result = 0.0f;
    dot_(count, &result, 0, weights, 0, input, 0, 1);
    return result;
  }

 private:
  void* handle_ = nullptr;
  LlamaDot dot_ = nullptr;
};

int positive_int(const char* text, const std::string& name) {
  char* end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (!end || *end != '\0' || value <= 0 ||
      value > static_cast<long>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("invalid positive integer for " + name);
  }
  return static_cast<int>(value);
}

Config parse_args(int argc, char** argv) {
  Config cfg;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto value = [&]() -> const char* {
      if (++i >= argc) {
        throw std::invalid_argument("missing value for " + arg);
      }
      return argv[i];
    };
    if (arg == "--model") {
      cfg.model = value();
    } else if (arg == "--tensor") {
      cfg.tensor = value();
    } else if (arg == "--ggml-cpu-library") {
      cfg.ggml_cpu_library = value();
    } else if (arg == "--warmup") {
      cfg.warmup = positive_int(value(), arg);
    } else if (arg == "--iterations") {
      cfg.iterations = positive_int(value(), arg);
    } else if (arg == "--cache-flush-mib") {
      cfg.cache_flush_mib = positive_int(value(), arg);
    } else if (arg == "--row-limit") {
      cfg.row_limit = positive_int(value(), arg);
    } else if (arg == "--seed") {
      cfg.seed = static_cast<std::uint64_t>(positive_int(value(), arg));
    } else if (arg == "--list") {
      cfg.list = true;
    } else {
      throw std::invalid_argument("unknown argument: " + arg);
    }
  }
  if (cfg.model.empty()) {
    throw std::invalid_argument("--model is required");
  }
  if (!cfg.list && cfg.ggml_cpu_library.empty()) {
    throw std::invalid_argument("--ggml-cpu-library is required for validation");
  }
  return cfg;
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2;
  return values.size() % 2 ? values[middle]
                           : (values[middle - 1] + values[middle]) * 0.5;
}

std::uint64_t flush_cache(std::vector<std::uint8_t>& cache) {
  std::uint64_t sum = 0;
  for (std::size_t i = 0; i < cache.size(); i += 64) {
    cache[i] = static_cast<std::uint8_t>(cache[i] + 1);
    sum += cache[i];
  }
  return sum;
}

template <typename Function>
double benchmark(Function&& function, int warmup, int iterations,
                 std::vector<std::uint8_t>& cache, double& checksum) {
  for (int i = 0; i < warmup; ++i) {
    checksum += function();
  }
  std::vector<double> samples;
  samples.reserve(iterations);
  volatile std::uint64_t cache_sink = 0;
  for (int i = 0; i < iterations; ++i) {
    cache_sink = flush_cache(cache);
    const auto start = Clock::now();
    checksum += function();
    const auto end = Clock::now();
    samples.push_back(std::chrono::duration<double, std::micro>(end - start).count());
  }
  (void)cache_sink;
  return median(std::move(samples));
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Config cfg = parse_args(argc, argv);
    const MappedFile file(cfg.model);
    const GgufIndex index = parse_gguf(file);

    if (cfg.list) {
      std::cout << "{\n  \"version\": " << index.version << ",\n"
                << "  \"alignment\": " << index.alignment << ",\n"
                << "  \"tensor_count\": " << index.tensors.size() << ",\n"
                << "  \"q4_k_tensors\": [\n";
      bool first = true;
      for (const TensorInfo& tensor : index.tensors) {
        if (tensor.type != kGgmlTypeQ4K) {
          continue;
        }
        if (!first) {
          std::cout << ",\n";
        }
        first = false;
        std::cout << "    {\"name\": \"" << tensor.name << "\", \"dims\": [";
        for (std::size_t d = 0; d < tensor.dims.size(); ++d) {
          std::cout << (d ? ", " : "") << tensor.dims[d];
        }
        std::cout << "]}";
      }
      std::cout << "\n  ]\n}\n";
      return 0;
    }

    const auto found = std::find_if(index.tensors.begin(), index.tensors.end(),
                                    [&](const TensorInfo& tensor) {
                                      return tensor.name == cfg.tensor;
                                    });
    if (found == index.tensors.end()) {
      throw std::runtime_error("tensor not found: " + cfg.tensor);
    }
    if (found->type != kGgmlTypeQ4K || found->dims.size() != 2) {
      throw std::runtime_error("target must be a rank-2 Q4_K tensor");
    }
    const int cols = static_cast<int>(found->dims[0]);
    const int available_rows = static_cast<int>(found->dims[1]);
    const int rows = cfg.row_limit > 0 ? std::min(cfg.row_limit, available_rows)
                                       : available_rows;
    if (cols % kQkK != 0) {
      throw std::runtime_error("Q4_K tensor width is not divisible by 256");
    }
    const std::size_t row_bytes = static_cast<std::size_t>(cols / kQkK) *
                                  sizeof(BlockQ4K);
    const std::size_t tensor_bytes = row_bytes * available_rows;
    const std::size_t absolute_offset = index.data_offset + found->offset;
    if (absolute_offset > file.size() || tensor_bytes > file.size() - absolute_offset) {
      throw std::runtime_error("tensor bytes exceed mapped GGUF file");
    }
    const auto* weights = reinterpret_cast<const BlockQ4K*>(file.data() + absolute_offset);

    XorShift64 rng(cfg.seed);
    std::vector<float> activation(cols);
    for (float& value : activation) {
      value = rng.uniform();
    }
    std::vector<BlockQ8K> quantized(cols / kQkK);
    quantize_q8k(activation.data(), cols, quantized.data());
    const LlamaKernel llama(cfg.ggml_cpu_library);

    std::vector<float> scalar_output(rows);
    std::vector<float> llama_output(rows);
    std::vector<float> m4_output(rows);
    float max_abs_diff_llama = 0.0f;
    float max_abs_diff_scalar = 0.0f;
    for (int row = 0; row < rows; ++row) {
      const auto* row_weights = reinterpret_cast<const BlockQ4K*>(
          reinterpret_cast<const std::uint8_t*>(weights) + row * row_bytes);
      scalar_output[row] = dot_q4k_q8k_scalar(cols, row_weights, quantized.data());
      llama_output[row] = llama.dot(cols, row_weights, quantized.data());
    }
    matvec_q4k_q8k_m4(cols, weights, row_bytes, rows, quantized.data(),
                      m4_output.data());
    for (int row = 0; row < rows; ++row) {
      max_abs_diff_llama =
          std::max(max_abs_diff_llama, std::fabs(m4_output[row] - llama_output[row]));
      max_abs_diff_scalar =
          std::max(max_abs_diff_scalar, std::fabs(m4_output[row] - scalar_output[row]));
    }

    std::vector<std::uint8_t> cache(
        static_cast<std::size_t>(cfg.cache_flush_mib) * 1024 * 1024);
    double llama_checksum = 0.0;
    double m4_checksum = 0.0;
    auto run_llama = [&]() {
      double sum = 0.0;
      for (int row = 0; row < rows; ++row) {
        const auto* row_weights = reinterpret_cast<const BlockQ4K*>(
            reinterpret_cast<const std::uint8_t*>(weights) + row * row_bytes);
        sum += llama.dot(cols, row_weights, quantized.data());
      }
      return sum;
    };
    auto run_m4 = [&]() {
      matvec_q4k_q8k_m4(cols, weights, row_bytes, rows, quantized.data(),
                        m4_output.data());
      double sum = 0.0;
      for (float value : m4_output) {
        sum += value;
      }
      return sum;
    };
    const double llama_us = benchmark(run_llama, cfg.warmup, cfg.iterations, cache,
                                      llama_checksum);
    const double m4_us =
        benchmark(run_m4, cfg.warmup, cfg.iterations, cache, m4_checksum);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "{\n"
              << "  \"schema_version\": 1,\n"
              << "  \"implementation\": \"cpp/m4_q4k_gguf.cpp\",\n"
              << "  \"mode\": \"real_gguf_q4_k_tensor_kernel_ab\",\n"
              << "  \"model_filename\": \""
              << cfg.model.substr(cfg.model.find_last_of('/') + 1) << "\",\n"
              << "  \"gguf_version\": " << index.version << ",\n"
              << "  \"tensor\": \"" << found->name << "\",\n"
              << "  \"ggml_type\": \"Q4_K\",\n"
              << "  \"cols\": " << cols << ",\n"
              << "  \"rows\": " << rows << ",\n"
              << "  \"tensor_offset_bytes\": " << absolute_offset << ",\n"
              << "  \"tensor_bytes\": " << tensor_bytes << ",\n"
              << "  \"cache_flush_mib\": " << cfg.cache_flush_mib << ",\n"
              << "  \"iterations\": " << cfg.iterations << ",\n"
              << "  \"max_abs_diff_vs_llama\": " << max_abs_diff_llama << ",\n"
              << "  \"max_abs_diff_vs_scalar\": " << max_abs_diff_scalar << ",\n"
              << "  \"correct\": "
              << (max_abs_diff_llama <= 1e-4f ? "true" : "false") << ",\n"
              << "  \"llama_cpp_median_us\": " << llama_us << ",\n"
              << "  \"m4_custom_median_us\": " << m4_us << ",\n"
              << "  \"speedup_vs_llama_cpp_kernel\": " << llama_us / m4_us << ",\n"
              << "  \"llama_checksum\": " << llama_checksum << ",\n"
              << "  \"m4_checksum\": " << m4_checksum << "\n"
              << "}\n";
    return max_abs_diff_llama <= 1e-4f ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << "\n";
    return 2;
  }
}
