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
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
constexpr std::uint32_t kGgufMagic = 0x46554747;
constexpr std::uint32_t kGgmlTypeQ4K = 12;
constexpr int kQkK = 256;
constexpr int kScaleBytes = 12;

struct BlockQ4K {
  std::uint16_t d;
  std::uint16_t dmin;
  std::uint8_t scales[kScaleBytes];
  std::uint8_t qs[kQkK / 2];
};

struct BlockQ8K {
  float d;
  std::int8_t qs[kQkK];
  std::int16_t bsums[kQkK / 16];
};

static_assert(sizeof(BlockQ4K) == 144, "Q4_K ABI mismatch");
static_assert(sizeof(BlockQ8K) == 292, "Q8_K ABI mismatch");

struct TensorInfo {
  std::string name;
  std::vector<std::uint64_t> dims;
  std::uint32_t type = 0;
  std::uint64_t offset = 0;
};

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

class MappedFile {
 public:
  explicit MappedFile(const std::string& path) {
    fd_ = open(path.c_str(), O_RDONLY);
    if (fd_ < 0) {
      throw std::runtime_error("cannot open model: " + path);
    }
    struct stat st {};
    if (fstat(fd_, &st) != 0 || st.st_size <= 0) {
      close(fd_);
      throw std::runtime_error("cannot stat model: " + path);
    }
    size_ = static_cast<std::size_t>(st.st_size);
    data_ = static_cast<const std::uint8_t*>(
        mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0));
    if (data_ == MAP_FAILED) {
      data_ = nullptr;
      close(fd_);
      throw std::runtime_error("cannot mmap model: " + path);
    }
  }

  ~MappedFile() {
    if (data_) {
      munmap(const_cast<std::uint8_t*>(data_), size_);
    }
    if (fd_ >= 0) {
      close(fd_);
    }
  }

  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;

  const std::uint8_t* data() const { return data_; }
  std::size_t size() const { return size_; }

 private:
  int fd_ = -1;
  const std::uint8_t* data_ = nullptr;
  std::size_t size_ = 0;
};

class Reader {
 public:
  Reader(const std::uint8_t* data, std::size_t size) : data_(data), size_(size) {}

  template <typename T>
  T read() {
    require(sizeof(T));
    T value;
    std::memcpy(&value, data_ + offset_, sizeof(T));
    offset_ += sizeof(T);
    return value;
  }

  std::string read_string() {
    const std::uint64_t length = read<std::uint64_t>();
    if (length > (1u << 24)) {
      throw std::runtime_error("unreasonable GGUF string length");
    }
    require(static_cast<std::size_t>(length));
    std::string result(reinterpret_cast<const char*>(data_ + offset_),
                       static_cast<std::size_t>(length));
    offset_ += static_cast<std::size_t>(length);
    return result;
  }

  void skip(std::size_t bytes) {
    require(bytes);
    offset_ += bytes;
  }

  std::size_t offset() const { return offset_; }

 private:
  void require(std::size_t bytes) const {
    if (bytes > size_ - std::min(size_, offset_)) {
      throw std::runtime_error("truncated GGUF file");
    }
  }

  const std::uint8_t* data_;
  std::size_t size_;
  std::size_t offset_ = 0;
};

std::size_t scalar_size(std::uint32_t type) {
  switch (type) {
    case 0:
    case 1:
    case 7:
      return 1;
    case 2:
    case 3:
      return 2;
    case 4:
    case 5:
    case 6:
      return 4;
    case 10:
    case 11:
    case 12:
      return 8;
    default:
      throw std::runtime_error("unsupported GGUF scalar metadata type " +
                               std::to_string(type));
  }
}

void skip_value(Reader& reader, std::uint32_t type) {
  if (type == 8) {
    (void)reader.read_string();
    return;
  }
  if (type == 9) {
    const std::uint32_t element_type = reader.read<std::uint32_t>();
    const std::uint64_t count = reader.read<std::uint64_t>();
    if (count > (1ull << 34)) {
      throw std::runtime_error("unreasonable GGUF array length");
    }
    if (element_type == 8 || element_type == 9) {
      for (std::uint64_t i = 0; i < count; ++i) {
        skip_value(reader, element_type);
      }
    } else {
      const std::size_t width = scalar_size(element_type);
      if (count > std::numeric_limits<std::size_t>::max() / width) {
        throw std::runtime_error("GGUF array size overflow");
      }
      reader.skip(static_cast<std::size_t>(count) * width);
    }
    return;
  }
  reader.skip(scalar_size(type));
}

struct GgufIndex {
  std::uint32_t version = 0;
  std::uint32_t alignment = 32;
  std::size_t data_offset = 0;
  std::vector<TensorInfo> tensors;
};

GgufIndex parse_gguf(const MappedFile& file) {
  Reader reader(file.data(), file.size());
  if (reader.read<std::uint32_t>() != kGgufMagic) {
    throw std::runtime_error("missing GGUF magic");
  }
  GgufIndex index;
  index.version = reader.read<std::uint32_t>();
  if (index.version < 2 || index.version > 3) {
    throw std::runtime_error("unsupported GGUF version " +
                             std::to_string(index.version));
  }
  const std::uint64_t tensor_count = reader.read<std::uint64_t>();
  const std::uint64_t kv_count = reader.read<std::uint64_t>();
  if (tensor_count > 1000000 || kv_count > 1000000) {
    throw std::runtime_error("unreasonable GGUF table count");
  }

  for (std::uint64_t i = 0; i < kv_count; ++i) {
    const std::string key = reader.read_string();
    const std::uint32_t type = reader.read<std::uint32_t>();
    if (key == "general.alignment" && type == 4) {
      index.alignment = reader.read<std::uint32_t>();
    } else {
      skip_value(reader, type);
    }
  }
  if (index.alignment == 0 || (index.alignment & (index.alignment - 1)) != 0) {
    throw std::runtime_error("invalid GGUF alignment");
  }

  index.tensors.reserve(static_cast<std::size_t>(tensor_count));
  for (std::uint64_t i = 0; i < tensor_count; ++i) {
    TensorInfo tensor;
    tensor.name = reader.read_string();
    const std::uint32_t dims = reader.read<std::uint32_t>();
    if (dims == 0 || dims > 4) {
      throw std::runtime_error("unsupported tensor rank");
    }
    tensor.dims.reserve(dims);
    for (std::uint32_t d = 0; d < dims; ++d) {
      tensor.dims.push_back(reader.read<std::uint64_t>());
    }
    tensor.type = reader.read<std::uint32_t>();
    tensor.offset = reader.read<std::uint64_t>();
    index.tensors.push_back(std::move(tensor));
  }
  index.data_offset =
      (reader.offset() + index.alignment - 1) & ~(index.alignment - 1);
  if (index.data_offset >= file.size()) {
    throw std::runtime_error("invalid GGUF tensor data offset");
  }
  return index;
}

float fp16_to_fp32(std::uint16_t bits) {
  __fp16 value;
  std::memcpy(&value, &bits, sizeof(bits));
  return static_cast<float>(value);
}

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
