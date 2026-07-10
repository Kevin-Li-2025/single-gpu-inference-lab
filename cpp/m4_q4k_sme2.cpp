#include <arm_neon.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "m4_gguf_q4k.hpp"

extern "C" {
#include "kai/kai_common.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32p_f32_neon.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon.h"
}

namespace {

using Clock = std::chrono::steady_clock;
using m4q4k::BlockQ4K;
using m4q4k::BlockQ8K;
using m4q4k::GgufIndex;
using m4q4k::MappedFile;
using m4q4k::TensorInfo;

constexpr std::size_t kBlockLength = 32;
constexpr std::size_t kQ4SourceBytes = kBlockLength / 2 + sizeof(std::uint16_t);

struct Config {
  std::string model;
  std::string tensor = "blk.0.ffn_up.weight";
  int row_limit = 0;
  int warmup = 3;
  int iterations = 10;
  int cache_flush_mib = 64;
  std::uint64_t seed = 20260710;
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

int parse_positive(const char* text, const std::string& name) {
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
    } else if (arg == "--row-limit") {
      cfg.row_limit = parse_positive(value(), arg);
    } else if (arg == "--warmup") {
      cfg.warmup = parse_positive(value(), arg);
    } else if (arg == "--iterations") {
      cfg.iterations = parse_positive(value(), arg);
    } else if (arg == "--cache-flush-mib") {
      cfg.cache_flush_mib = parse_positive(value(), arg);
    } else if (arg == "--seed") {
      cfg.seed = static_cast<std::uint64_t>(parse_positive(value(), arg));
    } else {
      throw std::invalid_argument("unknown argument: " + arg);
    }
  }
  if (cfg.model.empty()) {
    throw std::invalid_argument("--model is required");
  }
  return cfg;
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2;
  return values.size() % 2 != 0
             ? values[middle]
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
    samples.push_back(
        std::chrono::duration<double, std::micro>(end - start).count());
  }
  (void)cache_sink;
  return median(std::move(samples));
}

void quantize_q8k(const float* input, int count, BlockQ8K* output) {
  for (int block = 0; block < count / m4q4k::kQkK; ++block) {
    const float* source = input + block * m4q4k::kQkK;
    BlockQ8K& target = output[block];
    float max_value = 0.0f;
    float max_abs = 0.0f;
    for (int i = 0; i < m4q4k::kQkK; ++i) {
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
    for (int i = 0; i < m4q4k::kQkK; ++i) {
      const int value =
          static_cast<int>(std::nearbyint(source[i] * inverse_scale));
      target.qs[i] = static_cast<std::int8_t>(std::min(127, value));
    }
    for (int group = 0; group < m4q4k::kQkK / 16; ++group) {
      int sum = 0;
      for (int i = 0; i < 16; ++i) {
        sum += target.qs[group * 16 + i];
      }
      target.bsums[group] = static_cast<std::int16_t>(sum);
    }
    target.d = 1.0f / inverse_scale;
  }
}

float dot_q4k_q8k_m4(int count, const BlockQ4K* weights,
                     const BlockQ8K* input) {
  const uint8x16_t mask = vdupq_n_u8(0x0f);
  float total = 0.0f;
  for (int block = 0; block < count / m4q4k::kQkK; ++block) {
    const BlockQ4K& w = weights[block];
    const BlockQ8K& x = input[block];
    int weighted_dot = 0;
    int weighted_min = 0;
    for (int group = 0; group < 8; ++group) {
      std::uint8_t scale = 0;
      std::uint8_t minimum = 0;
      m4q4k::decode_scale_min(w.scales, group, scale, minimum);
      const int chunk = group / 2;
      const bool high = (group & 1) != 0;
      const std::uint8_t* packed = w.qs + chunk * 32;
      const std::int8_t* q8 = x.qs + group * 32;
      int32x4_t dot = vdupq_n_s32(0);
      for (int i = 0; i < 32; i += 16) {
        const uint8x16_t q4 = vld1q_u8(packed + i);
        const int8x16_t values = vreinterpretq_s8_u8(
            high ? vshrq_n_u8(q4, 4) : vandq_u8(q4, mask));
        dot = vdotq_s32(dot, values, vld1q_s8(q8 + i));
      }
      weighted_dot += static_cast<int>(scale) * vaddvq_s32(dot);
      weighted_min += static_cast<int>(minimum) *
                      (x.bsums[group * 2] + x.bsums[group * 2 + 1]);
    }
    total += m4q4k::fp16_to_fp32(w.d) * x.d * weighted_dot -
             m4q4k::fp16_to_fp32(w.dmin) * x.d * weighted_min;
  }
  return total;
}

void matvec_q4k_q8k_m4(int cols, const BlockQ4K* weights,
                       std::size_t row_bytes, int rows,
                       const BlockQ8K* input, float* output) {
  const auto* bytes = reinterpret_cast<const std::uint8_t*>(weights);
  for (int row = 0; row < rows; ++row) {
    output[row] = dot_q4k_q8k_m4(
        cols, reinterpret_cast<const BlockQ4K*>(bytes + row * row_bytes),
        input);
  }
}

struct Sme2Weights {
  std::vector<std::uint8_t> source;
  std::vector<std::uint8_t> packed;
  std::vector<float> correction;
  std::size_t groups = 0;
  std::size_t nr = 0;
  std::size_t kr = 0;
  std::size_t sr = 0;
};

Sme2Weights repack_q4k(const BlockQ4K* weights, std::size_t row_bytes,
                       int rows, int cols) {
  Sme2Weights result;
  result.groups = static_cast<std::size_t>(cols) / kBlockLength;
  result.nr =
      kai_get_nr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot();
  result.kr =
      kai_get_kr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot();
  result.sr =
      kai_get_sr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot();
  result.source.resize(static_cast<std::size_t>(rows) * result.groups *
                       kQ4SourceBytes);
  result.correction.resize(static_cast<std::size_t>(rows) * result.groups);

  const auto* weight_bytes = reinterpret_cast<const std::uint8_t*>(weights);
  for (int row = 0; row < rows; ++row) {
    const auto* row_weights = reinterpret_cast<const BlockQ4K*>(
        weight_bytes + static_cast<std::size_t>(row) * row_bytes);
    for (int block = 0; block < cols / m4q4k::kQkK; ++block) {
      const BlockQ4K& source_block = row_weights[block];
      const float d = m4q4k::fp16_to_fp32(source_block.d);
      const float dmin = m4q4k::fp16_to_fp32(source_block.dmin);
      for (int group = 0; group < 8; ++group) {
        std::uint8_t scale = 0;
        std::uint8_t minimum = 0;
        m4q4k::decode_scale_min(source_block.scales, group, scale, minimum);
        const std::size_t group_index =
            static_cast<std::size_t>(block) * 8 + group;
        std::uint8_t* destination =
            result.source.data() +
            (static_cast<std::size_t>(row) * result.groups + group_index) *
                kQ4SourceBytes;
        const std::uint16_t scale_f16 =
            m4q4k::fp32_to_fp16(d * static_cast<float>(scale));
        std::memcpy(destination, &scale_f16, sizeof(scale_f16));
        for (int i = 0; i < 16; ++i) {
          const std::uint8_t low = m4q4k::q4_k_value(source_block, group, i);
          const std::uint8_t high =
              m4q4k::q4_k_value(source_block, group, i + 16);
          destination[sizeof(scale_f16) + i] = low | (high << 4);
        }
        const float rounded_scale = m4q4k::fp16_to_fp32(scale_f16);
        result.correction[static_cast<std::size_t>(row) * result.groups +
                          group_index] =
            8.0f * rounded_scale - dmin * static_cast<float>(minimum);
      }
    }
  }

  const std::size_t packed_size =
      kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon(
          rows, cols, result.nr, result.kr, kBlockLength);
  result.packed.resize(packed_size);
  const kai_rhs_pack_qs4cxs1s0_param params{1, 8};
  kai_run_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon(
      1, rows, cols, result.nr, result.kr, result.sr, kBlockLength,
      result.source.data(), nullptr, result.packed.data(), 0, &params);
  return result;
}

void qsi8_block_sums(const std::uint8_t* packed_lhs, std::size_t groups,
                     std::vector<float>& sums) {
  const auto* values = reinterpret_cast<const std::int8_t*>(packed_lhs);
  const auto* scales = reinterpret_cast<const std::uint16_t*>(
      packed_lhs + groups * kBlockLength);
  for (std::size_t group = 0; group < groups; ++group) {
    const std::int8_t* block = values + group * kBlockLength;
    const int sum = vaddlvq_s8(vld1q_s8(block)) +
                    vaddlvq_s8(vld1q_s8(block + 16));
    sums[group] = static_cast<float>(sum) * m4q4k::fp16_to_fp32(scales[group]);
  }
}

void apply_correction(const std::vector<float>& coefficients,
                      const std::vector<float>& sums, int rows,
                      std::vector<float>& output) {
  const std::size_t groups = sums.size();
  for (int row = 0; row < rows; ++row) {
    const float* coeff = coefficients.data() + static_cast<std::size_t>(row) * groups;
    float32x4_t total4 = vdupq_n_f32(0.0f);
    std::size_t group = 0;
    for (; group + 4 <= groups; group += 4) {
      total4 = vfmaq_f32(total4, vld1q_f32(coeff + group),
                         vld1q_f32(sums.data() + group));
    }
    float total = vaddvq_f32(total4);
    for (; group < groups; ++group) {
      total += coeff[group] * sums[group];
    }
    output[row] += total;
  }
}

void run_sme2(const std::vector<float>& activation, const Sme2Weights& weights,
              int rows, int cols, std::vector<std::uint8_t>& packed_lhs,
              std::vector<float>& block_sums, std::vector<float>& output) {
  const std::size_t mr =
      kai_get_mr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot();
  kai_run_lhs_quant_pack_qsi8d32p_f32_neon(
      1, cols, kBlockLength, mr, weights.kr, weights.sr, 0,
      activation.data(), static_cast<std::size_t>(cols) * sizeof(float),
      packed_lhs.data());
  qsi8_block_sums(packed_lhs.data(), weights.groups, block_sums);
  kai_run_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot(
      1, rows, cols, kBlockLength, packed_lhs.data(), weights.packed.data(),
      output.data(), static_cast<std::size_t>(rows) * sizeof(float),
      sizeof(float), std::numeric_limits<float>::lowest(),
      std::numeric_limits<float>::max());
  apply_correction(weights.correction, block_sums, rows, output);
}

void reference_qsi8(const BlockQ4K* weights, std::size_t row_bytes, int rows,
                    int cols, const std::uint8_t* packed_lhs,
                    std::vector<float>& output) {
  const std::size_t groups = static_cast<std::size_t>(cols) / kBlockLength;
  const auto* q8 = reinterpret_cast<const std::int8_t*>(packed_lhs);
  const auto* lhs_scales = reinterpret_cast<const std::uint16_t*>(
      packed_lhs + groups * kBlockLength);
  const auto* bytes = reinterpret_cast<const std::uint8_t*>(weights);
  for (int row = 0; row < rows; ++row) {
    const auto* row_weights = reinterpret_cast<const BlockQ4K*>(
        bytes + static_cast<std::size_t>(row) * row_bytes);
    double total = 0.0;
    for (int block = 0; block < cols / m4q4k::kQkK; ++block) {
      const BlockQ4K& weight = row_weights[block];
      const float dmin = m4q4k::fp16_to_fp32(weight.dmin);
      for (int group = 0; group < 8; ++group) {
        std::uint8_t scale = 0;
        std::uint8_t minimum = 0;
        m4q4k::decode_scale_min(weight.scales, group, scale, minimum);
        const std::size_t group_index =
            static_cast<std::size_t>(block) * 8 + group;
        const float weight_scale = m4q4k::fp16_to_fp32(
            m4q4k::fp32_to_fp16(m4q4k::fp16_to_fp32(weight.d) * scale));
        const float input_scale = m4q4k::fp16_to_fp32(lhs_scales[group_index]);
        for (int i = 0; i < 32; ++i) {
          const float dequant_weight =
              weight_scale * m4q4k::q4_k_value(weight, group, i) -
              dmin * minimum;
          total += dequant_weight * q8[group_index * 32 + i] * input_scale;
        }
      }
    }
    output[row] = static_cast<float>(total);
  }
}

void reference_fp32(const BlockQ4K* weights, std::size_t row_bytes, int rows,
                    int cols, const std::vector<float>& activation,
                    std::vector<float>& output) {
  const auto* bytes = reinterpret_cast<const std::uint8_t*>(weights);
  for (int row = 0; row < rows; ++row) {
    const auto* row_weights = reinterpret_cast<const BlockQ4K*>(
        bytes + static_cast<std::size_t>(row) * row_bytes);
    double total = 0.0;
    for (int block = 0; block < cols / m4q4k::kQkK; ++block) {
      const BlockQ4K& weight = row_weights[block];
      const float d = m4q4k::fp16_to_fp32(weight.d);
      const float dmin = m4q4k::fp16_to_fp32(weight.dmin);
      for (int group = 0; group < 8; ++group) {
        std::uint8_t scale = 0;
        std::uint8_t minimum = 0;
        m4q4k::decode_scale_min(weight.scales, group, scale, minimum);
        for (int i = 0; i < 32; ++i) {
          const int column = block * m4q4k::kQkK + group * 32 + i;
          const float dequant_weight =
              d * scale * m4q4k::q4_k_value(weight, group, i) - dmin * minimum;
          total += dequant_weight * activation[column];
        }
      }
    }
    output[row] = static_cast<float>(total);
  }
}

struct ErrorStats {
  float max_abs = 0.0f;
  double rmse = 0.0;
  double normalized_rmse = 0.0;
};

ErrorStats error_stats(const std::vector<float>& actual,
                       const std::vector<float>& reference) {
  double square_error = 0.0;
  double square_reference = 0.0;
  float max_abs = 0.0f;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const double error = static_cast<double>(actual[i]) - reference[i];
    max_abs = std::max(max_abs, static_cast<float>(std::fabs(error)));
    square_error += error * error;
    square_reference += static_cast<double>(reference[i]) * reference[i];
  }
  ErrorStats result;
  result.max_abs = max_abs;
  result.rmse = std::sqrt(square_error / actual.size());
  result.normalized_rmse =
      square_reference == 0.0 ? 0.0 : std::sqrt(square_error / square_reference);
  return result;
}

double checksum(const std::vector<float>& values) {
  double result = 0.0;
  for (float value : values) {
    result += value;
  }
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Config cfg = parse_args(argc, argv);
    const MappedFile file(cfg.model);
    const GgufIndex index = m4q4k::parse_gguf(file);
    const auto found = std::find_if(
        index.tensors.begin(), index.tensors.end(),
        [&](const TensorInfo& tensor) { return tensor.name == cfg.tensor; });
    if (found == index.tensors.end()) {
      throw std::runtime_error("tensor not found: " + cfg.tensor);
    }
    if (found->type != m4q4k::kGgmlTypeQ4K || found->dims.size() != 2) {
      throw std::runtime_error("target must be a rank-2 Q4_K tensor");
    }
    const int cols = static_cast<int>(found->dims[0]);
    const int available_rows = static_cast<int>(found->dims[1]);
    const int rows = cfg.row_limit > 0 ? std::min(cfg.row_limit, available_rows)
                                       : available_rows;
    if (cols % m4q4k::kQkK != 0 || cols % kBlockLength != 0) {
      throw std::runtime_error("Q4_K width is incompatible with block-32 SME2");
    }
    const std::size_t row_bytes =
        static_cast<std::size_t>(cols / m4q4k::kQkK) * sizeof(BlockQ4K);
    const std::size_t tensor_bytes = row_bytes * available_rows;
    const std::size_t absolute_offset = index.data_offset + found->offset;
    if (absolute_offset > file.size() ||
        tensor_bytes > file.size() - absolute_offset) {
      throw std::runtime_error("tensor bytes exceed mapped GGUF file");
    }
    const auto* weights =
        reinterpret_cast<const BlockQ4K*>(file.data() + absolute_offset);

    XorShift64 rng(cfg.seed);
    std::vector<float> activation(cols);
    for (float& value : activation) {
      value = rng.uniform();
    }

    const Sme2Weights sme2_weights =
        repack_q4k(weights, row_bytes, rows, cols);
    const std::size_t mr =
        kai_get_mr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot();
    const std::size_t lhs_size =
        kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32p_f32_neon(
            1, cols, kBlockLength, mr, sme2_weights.kr, sme2_weights.sr);
    std::vector<std::uint8_t> packed_lhs(lhs_size);
    std::vector<float> block_sums(sme2_weights.groups);
    std::vector<float> sme2_output(rows);
    run_sme2(activation, sme2_weights, rows, cols, packed_lhs, block_sums,
             sme2_output);

    std::vector<float> qsi8_reference(rows);
    std::vector<float> fp32_reference(rows);
    reference_qsi8(weights, row_bytes, rows, cols, packed_lhs.data(),
                   qsi8_reference);
    reference_fp32(weights, row_bytes, rows, cols, activation, fp32_reference);
    const ErrorStats qsi8_error = error_stats(sme2_output, qsi8_reference);
    const ErrorStats fp32_error = error_stats(sme2_output, fp32_reference);

    std::vector<BlockQ8K> q8k(static_cast<std::size_t>(cols) / m4q4k::kQkK);
    std::vector<float> neon_output(rows);
    quantize_q8k(activation.data(), cols, q8k.data());
    matvec_q4k_q8k_m4(cols, weights, row_bytes, rows, q8k.data(),
                      neon_output.data());
    const ErrorStats neon_fp32_error = error_stats(neon_output, fp32_reference);

    std::vector<std::uint8_t> cache(
        static_cast<std::size_t>(cfg.cache_flush_mib) * 1024 * 1024);
    double sme2_checksum = 0.0;
    double neon_checksum = 0.0;
    auto run_sme2_step = [&]() {
      run_sme2(activation, sme2_weights, rows, cols, packed_lhs, block_sums,
               sme2_output);
      return checksum(sme2_output);
    };
    auto run_neon_step = [&]() {
      quantize_q8k(activation.data(), cols, q8k.data());
      matvec_q4k_q8k_m4(cols, weights, row_bytes, rows, q8k.data(),
                        neon_output.data());
      return checksum(neon_output);
    };
    const double sme2_us = benchmark(run_sme2_step, cfg.warmup, cfg.iterations,
                                     cache, sme2_checksum);
    const double neon_us = benchmark(run_neon_step, cfg.warmup, cfg.iterations,
                                     cache, neon_checksum);
    const double speedup = neon_us / sme2_us;
    const bool mapping_correct = qsi8_error.normalized_rmse <= 1e-5;
    const bool quality_gate = fp32_error.normalized_rmse <= 0.01;
    const bool performance_gate = speedup >= 1.05;

    std::cout << std::fixed << std::setprecision(8)
              << "{\n"
              << "  \"schema_version\": 1,\n"
              << "  \"implementation\": \"cpp/m4_q4k_sme2.cpp\",\n"
              << "  \"mode\": \"real_gguf_q4_k_affine_sme2_gate\",\n"
              << "  \"model_filename\": \""
              << cfg.model.substr(cfg.model.find_last_of('/') + 1) << "\",\n"
              << "  \"tensor\": \"" << found->name << "\",\n"
              << "  \"cols\": " << cols << ",\n"
              << "  \"rows\": " << rows << ",\n"
              << "  \"tensor_offset_bytes\": " << absolute_offset << ",\n"
              << "  \"source_tensor_bytes\": " << row_bytes * rows << ",\n"
              << "  \"packed_rhs_bytes\": " << sme2_weights.packed.size() << ",\n"
              << "  \"correction_bytes\": "
              << sme2_weights.correction.size() * sizeof(float) << ",\n"
              << "  \"weight_transform\": \"lossless_q4_nibbles_plus_affine_correction\",\n"
              << "  \"rhs_scale_storage\": \"fp16_per_32_values\",\n"
              << "  \"activation_storage\": \"qsi8_fp16_scale_per_32_values\",\n"
              << "  \"qsi8_reference_max_abs_diff\": " << qsi8_error.max_abs
              << ",\n"
              << "  \"qsi8_reference_normalized_rmse\": "
              << qsi8_error.normalized_rmse << ",\n"
              << "  \"fp32_activation_max_abs_diff\": " << fp32_error.max_abs
              << ",\n"
              << "  \"fp32_activation_normalized_rmse\": "
              << fp32_error.normalized_rmse << ",\n"
              << "  \"neon_fp32_activation_normalized_rmse\": "
              << neon_fp32_error.normalized_rmse << ",\n"
              << "  \"cache_flush_mib\": " << cfg.cache_flush_mib << ",\n"
              << "  \"iterations\": " << cfg.iterations << ",\n"
              << "  \"neon_q4k_q8k_median_us\": " << neon_us << ",\n"
              << "  \"sme2_affine_median_us\": " << sme2_us << ",\n"
              << "  \"speedup_vs_custom_neon\": " << speedup << ",\n"
              << "  \"mapping_correct\": "
              << (mapping_correct ? "true" : "false") << ",\n"
              << "  \"quality_gate_pass\": "
              << (quality_gate ? "true" : "false") << ",\n"
              << "  \"performance_gate_pass\": "
              << (performance_gate ? "true" : "false") << ",\n"
              << "  \"full_decode_integration_gate_pass\": "
              << (mapping_correct && quality_gate && performance_gate ? "true"
                                                                       : "false")
              << ",\n"
              << "  \"sme2_checksum\": " << sme2_checksum << ",\n"
              << "  \"neon_checksum\": " << neon_checksum << "\n"
              << "}\n";
    std::cout.flush();
    return mapping_correct && quality_gate ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << "\n";
    return 2;
  }
}
