// Apple M4-specific Q4 x Q8 matrix-vector kernel benchmark.
//
// This is a model-shaped primitive, not an end-to-end model runner. Weights are
// stored as signed int4 blocks and activations are dynamically quantized to int8.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#if defined(__APPLE__)
#include <pthread/qos.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;
constexpr int kBlock = 32;
volatile std::uint64_t g_cache_sink = 0;

struct Config {
  int rows = 4864;
  int cols = 896;
  int threads = 4;
  int warmup = 8;
  int iterations = 40;
  int cache_flush_mib = 64;
  std::uint64_t seed = 7;
};

struct BlockQ4 {
  float scale;
  std::uint8_t values[kBlock / 2];
};

struct BlockQ8 {
  float scale;
  std::int8_t values[kBlock];
};

static_assert(sizeof(BlockQ4) == 20, "unexpected Q4 block padding");
static_assert(sizeof(BlockQ8) == 36, "unexpected Q8 block padding");

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

  float uniform(float magnitude) {
    const float unit = static_cast<float>(next()) / 4294967295.0f;
    return (unit * 2.0f - 1.0f) * magnitude;
  }
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
    if (arg == "--rows") {
      cfg.rows = positive_int(value(), arg);
    } else if (arg == "--cols") {
      cfg.cols = positive_int(value(), arg);
    } else if (arg == "--threads") {
      cfg.threads = positive_int(value(), arg);
    } else if (arg == "--warmup") {
      cfg.warmup = positive_int(value(), arg);
    } else if (arg == "--iterations") {
      cfg.iterations = positive_int(value(), arg);
    } else if (arg == "--cache-flush-mib") {
      cfg.cache_flush_mib = positive_int(value(), arg);
    } else if (arg == "--seed") {
      cfg.seed = static_cast<std::uint64_t>(positive_int(value(), arg));
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "usage: " << argv[0]
                << " [--rows N] [--cols N] [--threads N] [--warmup N]"
                   " [--iterations N] [--cache-flush-mib N] [--seed N]\n";
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown argument: " + arg);
    }
  }
  if (cfg.cols % kBlock != 0) {
    throw std::invalid_argument("--cols must be divisible by 32");
  }
  cfg.threads = std::min(cfg.threads, cfg.rows);
  return cfg;
}

void raise_thread_qos() {
#if defined(__APPLE__)
  (void)pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
}

std::vector<BlockQ4> make_weights(const Config& cfg, XorShift64& rng) {
  const int blocks_per_row = cfg.cols / kBlock;
  std::vector<BlockQ4> weights(static_cast<std::size_t>(cfg.rows) * blocks_per_row);
  for (BlockQ4& block : weights) {
    block.scale = 0.006f + std::fabs(rng.uniform(0.004f));
    for (std::uint8_t& packed : block.values) {
      const std::uint8_t low = static_cast<std::uint8_t>(rng.next() & 0x0f);
      const std::uint8_t high = static_cast<std::uint8_t>(rng.next() & 0x0f);
      packed = static_cast<std::uint8_t>(low | (high << 4));
    }
  }
  return weights;
}

void quantize_activation(const std::vector<float>& input, std::vector<BlockQ8>& output) {
  for (std::size_t b = 0; b < output.size(); ++b) {
    const float* source = input.data() + b * kBlock;
    float max_abs = 0.0f;
#if defined(__aarch64__)
    float32x4_t maximum = vdupq_n_f32(0.0f);
    for (int i = 0; i < kBlock; i += 4) {
      maximum = vmaxq_f32(maximum, vabsq_f32(vld1q_f32(source + i)));
    }
    max_abs = vmaxvq_f32(maximum);
#else
    for (int i = 0; i < kBlock; ++i) {
      max_abs = std::max(max_abs, std::fabs(source[i]));
    }
#endif
    BlockQ8& block = output[b];
    block.scale = max_abs > 0.0f ? max_abs / 127.0f : 0.0f;
    const float inverse = block.scale > 0.0f ? 1.0f / block.scale : 0.0f;
    for (int i = 0; i < kBlock; ++i) {
      const int quantized = static_cast<int>(std::nearbyint(source[i] * inverse));
      block.values[i] = static_cast<std::int8_t>(std::clamp(quantized, -127, 127));
    }
  }
}

inline std::int32_t dot_scalar(const BlockQ4& weight, const BlockQ8& input) {
  std::int32_t sum = 0;
  for (int i = 0; i < kBlock / 2; ++i) {
    const std::uint8_t packed = weight.values[i];
    sum += (static_cast<int>(packed & 0x0f) - 8) * input.values[i];
    sum += (static_cast<int>(packed >> 4) - 8) * input.values[i + 16];
  }
  return sum;
}

inline std::int32_t dot_m4(const BlockQ4& weight, const BlockQ8& input) {
#if defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)
  const uint8x16_t packed = vld1q_u8(weight.values);
  const uint8x16_t mask = vdupq_n_u8(0x0f);
  const uint8x16_t offset = vdupq_n_u8(8);
  const int8x16_t low = vreinterpretq_s8_u8(vsubq_u8(vandq_u8(packed, mask), offset));
  const int8x16_t high = vreinterpretq_s8_u8(vsubq_u8(vshrq_n_u8(packed, 4), offset));
  int32x4_t sums = vdupq_n_s32(0);
  sums = vdotq_s32(sums, low, vld1q_s8(input.values));
  sums = vdotq_s32(sums, high, vld1q_s8(input.values + 16));
  return vaddvq_s32(sums);
#else
  return dot_scalar(weight, input);
#endif
}

using DotKernel = std::int32_t (*)(const BlockQ4&, const BlockQ8&);

void matvec_range(const BlockQ4* weights, const BlockQ8* input, float* output,
                  int blocks_per_row, int first_row, int last_row, DotKernel dot) {
  for (int row = first_row; row < last_row; ++row) {
    const BlockQ4* row_weights = weights + static_cast<std::size_t>(row) * blocks_per_row;
    float sum = 0.0f;
    for (int block = 0; block < blocks_per_row; ++block) {
      sum += static_cast<float>(dot(row_weights[block], input[block])) *
             row_weights[block].scale * input[block].scale;
    }
    output[row] = sum;
  }
}

class PersistentMatvec {
 public:
  explicit PersistentMatvec(int threads) : threads_(threads) {
    workers_.reserve(std::max(0, threads - 1));
    for (int id = 1; id < threads_; ++id) {
      workers_.emplace_back([this, id] { worker_loop(id); });
    }
  }

  ~PersistentMatvec() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
      ++generation_;
    }
    ready_.notify_all();
    for (std::thread& worker : workers_) {
      worker.join();
    }
  }

  PersistentMatvec(const PersistentMatvec&) = delete;
  PersistentMatvec& operator=(const PersistentMatvec&) = delete;

  void run(const BlockQ4* weights, const BlockQ8* input, float* output, int rows,
           int blocks_per_row, DotKernel dot) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      weights_ = weights;
      input_ = input;
      output_ = output;
      rows_ = rows;
      blocks_per_row_ = blocks_per_row;
      dot_ = dot;
      pending_ = threads_ - 1;
      ++generation_;
    }
    ready_.notify_all();
    run_partition(0);
    if (threads_ > 1) {
      std::unique_lock<std::mutex> lock(mutex_);
      done_.wait(lock, [this] { return pending_ == 0; });
    }
  }

 private:
  void run_partition(int id) {
    const int first = rows_ * id / threads_;
    const int last = rows_ * (id + 1) / threads_;
    matvec_range(weights_, input_, output_, blocks_per_row_, first, last, dot_);
  }

  void worker_loop(int id) {
    raise_thread_qos();
    std::uint64_t seen = 0;
    while (true) {
      {
        std::unique_lock<std::mutex> lock(mutex_);
        ready_.wait(lock, [this, seen] { return generation_ != seen; });
        if (stop_) {
          return;
        }
        seen = generation_;
      }
      run_partition(id);
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (--pending_ == 0) {
          done_.notify_one();
        }
      }
    }
  }

  int threads_;
  std::vector<std::thread> workers_;
  std::mutex mutex_;
  std::condition_variable ready_;
  std::condition_variable done_;
  bool stop_ = false;
  std::uint64_t generation_ = 0;
  int pending_ = 0;
  const BlockQ4* weights_ = nullptr;
  const BlockQ8* input_ = nullptr;
  float* output_ = nullptr;
  int rows_ = 0;
  int blocks_per_row_ = 0;
  DotKernel dot_ = nullptr;
};

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2;
  if (values.size() % 2 == 0) {
    return (values[middle - 1] + values[middle]) * 0.5;
  }
  return values[middle];
}

struct Measurement {
  double median_us;
  double min_us;
  double checksum;
};

std::uint64_t flush_cache(std::vector<std::uint8_t>& cache) {
  std::uint64_t sum = 0;
  for (std::size_t i = 0; i < cache.size(); i += 64) {
    cache[i] = static_cast<std::uint8_t>(cache[i] + 1);
    sum += cache[i];
  }
  return sum;
}

double output_checksum(const std::vector<float>& output) {
  double checksum = 0.0;
  for (std::size_t i = 0; i < output.size(); ++i) {
    checksum += static_cast<double>(output[i]) * static_cast<double>((i % 17) + 1);
  }
  return checksum;
}

Measurement measure(PersistentMatvec& runner, const std::vector<BlockQ4>& weights,
                    const std::vector<BlockQ8>& input, std::vector<float>& output,
                    const Config& cfg, DotKernel dot, int warmup, int iterations,
                    std::vector<std::uint8_t>& cache) {
  const int blocks_per_row = cfg.cols / kBlock;
  for (int i = 0; i < warmup; ++i) {
    runner.run(weights.data(), input.data(), output.data(), cfg.rows, blocks_per_row, dot);
  }
  std::vector<double> samples;
  samples.reserve(iterations);
  std::uint64_t cache_checksum = 0;
  for (int i = 0; i < iterations; ++i) {
    cache_checksum += flush_cache(cache);
    const auto start = Clock::now();
    runner.run(weights.data(), input.data(), output.data(), cfg.rows, blocks_per_row, dot);
    const auto end = Clock::now();
    samples.push_back(std::chrono::duration<double, std::micro>(end - start).count());
  }
  g_cache_sink = cache_checksum;
  return {median(samples), *std::min_element(samples.begin(), samples.end()),
          output_checksum(output)};
}

Measurement measure_end_to_end(PersistentMatvec& runner,
                               const std::vector<BlockQ4>& weights,
                               const std::vector<float>& activation,
                               std::vector<BlockQ8>& quantized,
                               std::vector<float>& output, const Config& cfg) {
  const int blocks_per_row = cfg.cols / kBlock;
  std::vector<double> samples;
  samples.reserve(cfg.iterations);
  std::vector<std::uint8_t> cache(
      static_cast<std::size_t>(cfg.cache_flush_mib) * 1024 * 1024);
  std::uint64_t cache_checksum = 0;
  for (int i = 0; i < cfg.warmup + cfg.iterations; ++i) {
    cache_checksum += flush_cache(cache);
    const auto start = Clock::now();
    quantize_activation(activation, quantized);
    runner.run(weights.data(), quantized.data(), output.data(), cfg.rows, blocks_per_row,
               dot_m4);
    const auto end = Clock::now();
    if (i >= cfg.warmup) {
      samples.push_back(std::chrono::duration<double, std::micro>(end - start).count());
    }
  }
  g_cache_sink = cache_checksum;
  return {median(samples), *std::min_element(samples.begin(), samples.end()),
          output_checksum(output)};
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Config cfg = parse_args(argc, argv);
    raise_thread_qos();
    XorShift64 rng(cfg.seed);
    std::vector<BlockQ4> weights = make_weights(cfg, rng);
    std::vector<float> activation(cfg.cols);
    for (float& value : activation) {
      value = rng.uniform(1.0f);
    }
    std::vector<BlockQ8> quantized(cfg.cols / kBlock);
    quantize_activation(activation, quantized);
    std::vector<float> scalar_output(cfg.rows);
    std::vector<float> scalar_mt_output(cfg.rows);
    std::vector<float> neon_st_output(cfg.rows);
    std::vector<float> m4_output(cfg.rows);
    std::vector<std::uint8_t> cache(
        static_cast<std::size_t>(cfg.cache_flush_mib) * 1024 * 1024);

    // Persistent-worker wakeup dominates narrow Q/K/V/O projections. The
    // threshold keeps those on one performance core while parallelizing FFN-sized work.
    const int work_blocks = cfg.rows * (cfg.cols / kBlock);
    const int selected_threads = work_blocks >= 65536 ? cfg.threads : 1;

    PersistentMatvec scalar_runner(1);
    PersistentMatvec scalar_mt_runner(selected_threads);
    PersistentMatvec neon_st_runner(1);
    PersistentMatvec m4_runner(selected_threads);
    const Measurement scalar = measure(scalar_runner, weights, quantized, scalar_output, cfg,
                                       dot_scalar, cfg.warmup, cfg.iterations, cache);
    const Measurement scalar_mt = measure(scalar_mt_runner, weights, quantized,
                                          scalar_mt_output, cfg, dot_scalar, cfg.warmup,
                                          cfg.iterations, cache);
    const Measurement neon_st = measure(neon_st_runner, weights, quantized, neon_st_output,
                                        cfg, dot_m4, cfg.warmup, cfg.iterations, cache);
    const Measurement m4 = measure(m4_runner, weights, quantized, m4_output, cfg, dot_m4,
                                   cfg.warmup, cfg.iterations, cache);
    const Measurement end_to_end = measure_end_to_end(
        m4_runner, weights, activation, quantized, m4_output, cfg);

    float max_abs_diff = 0.0f;
    for (int row = 0; row < cfg.rows; ++row) {
      max_abs_diff = std::max(max_abs_diff, std::fabs(scalar_output[row] - m4_output[row]));
    }
    const double weight_mib = static_cast<double>(weights.size() * sizeof(BlockQ4)) /
                              (1024.0 * 1024.0);
    const double gib_per_s = (weight_mib / 1024.0) / (m4.median_us / 1.0e6);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "{\n"
              << "  \"schema_version\": 1,\n"
              << "  \"implementation\": \"cpp/m4_q4_matvec.cpp\",\n"
              << "  \"mode\": \"model_shaped_q4_q8_matvec_microbenchmark\",\n"
              << "  \"architecture\": \"Q4 weights + dynamic Q8 activations\",\n"
              << "  \"rows\": " << cfg.rows << ",\n"
              << "  \"cols\": " << cfg.cols << ",\n"
              << "  \"requested_threads\": " << cfg.threads << ",\n"
              << "  \"selected_threads\": " << selected_threads << ",\n"
              << "  \"dispatch_work_blocks\": " << work_blocks << ",\n"
              << "  \"dispatch_threshold_blocks\": 65536,\n"
              << "  \"iterations\": " << cfg.iterations << ",\n"
              << "  \"cache_flush_mib\": " << cfg.cache_flush_mib << ",\n"
              << "  \"weight_mib\": " << weight_mib << ",\n"
#if defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)
              << "  \"neon_dotprod_compiled\": true,\n"
#else
              << "  \"neon_dotprod_compiled\": false,\n"
#endif
              << "  \"scalar_median_us\": " << scalar.median_us << ",\n"
              << "  \"scalar_mt_median_us\": " << scalar_mt.median_us << ",\n"
              << "  \"neon_single_thread_median_us\": " << neon_st.median_us << ",\n"
              << "  \"m4_median_us\": " << m4.median_us << ",\n"
              << "  \"m4_min_us\": " << m4.min_us << ",\n"
              << "  \"m4_quantize_and_matvec_median_us\": " << end_to_end.median_us << ",\n"
              << "  \"speedup_vs_scalar\": " << scalar.median_us / m4.median_us << ",\n"
              << "  \"speedup_vs_scalar_same_threads\": "
              << scalar_mt.median_us / m4.median_us << ",\n"
              << "  \"parallel_speedup_neon\": " << neon_st.median_us / m4.median_us << ",\n"
              << "  \"dynamic_quantization_overhead_pct\": "
              << (end_to_end.median_us / m4.median_us - 1.0) * 100.0 << ",\n"
              << "  \"effective_weight_bandwidth_gib_s\": " << gib_per_s << ",\n"
              << "  \"max_abs_diff\": " << max_abs_diff << ",\n"
              << "  \"correct\": " << (max_abs_diff <= 1e-5f ? "true" : "false") << ",\n"
              << "  \"scalar_checksum\": " << scalar.checksum << ",\n"
              << "  \"m4_checksum\": " << m4.checksum << "\n"
              << "}\n";
    return max_abs_diff <= 1e-5f ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << "\n";
    return 2;
  }
}
