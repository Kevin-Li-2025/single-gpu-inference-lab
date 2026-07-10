#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace m4q4k {

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
      fd_ = -1;
      throw std::runtime_error("cannot stat model: " + path);
    }
    size_ = static_cast<std::size_t>(st.st_size);
    data_ = static_cast<const std::uint8_t*>(
        mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0));
    if (data_ == MAP_FAILED) {
      data_ = nullptr;
      close(fd_);
      fd_ = -1;
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
    if (offset_ > size_ || bytes > size_ - offset_) {
      throw std::runtime_error("truncated GGUF file");
    }
  }

  const std::uint8_t* data_;
  std::size_t size_;
  std::size_t offset_ = 0;
};

inline std::size_t scalar_size(std::uint32_t type) {
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

inline void skip_value(Reader& reader, std::uint32_t type) {
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

inline GgufIndex parse_gguf(const MappedFile& file) {
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

inline float fp16_to_fp32(std::uint16_t bits) {
  __fp16 value;
  std::memcpy(&value, &bits, sizeof(bits));
  return static_cast<float>(value);
}

inline std::uint16_t fp32_to_fp16(float value) {
  const __fp16 half = static_cast<__fp16>(value);
  std::uint16_t bits;
  std::memcpy(&bits, &half, sizeof(bits));
  return bits;
}

inline void decode_scale_min(const std::uint8_t* packed, int group,
                             std::uint8_t& scale, std::uint8_t& minimum) {
  if (group < 4) {
    scale = packed[group] & 63;
    minimum = packed[group + 4] & 63;
  } else {
    scale = (packed[group + 4] & 0x0f) | ((packed[group - 4] >> 6) << 4);
    minimum = (packed[group + 4] >> 4) | ((packed[group] >> 6) << 4);
  }
}

inline std::uint8_t q4_k_value(const BlockQ4K& block, int group, int index) {
  const std::uint8_t packed = block.qs[(group / 2) * 32 + index];
  return (group & 1) != 0 ? packed >> 4 : packed & 0x0f;
}

}  // namespace m4q4k
