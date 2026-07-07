// Self-contained CPU tiny-transformer benchmark.
//
// This is intentionally not a model loader. It exercises the decode path with
// deterministic synthetic weights so CPU-side bottlenecks can be measured before
// adding tokenizer, GGUF, quantization, or SIMD-specific kernels.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Config {
  int layers = 2;
  int dim = 64;
  int heads = 4;
  int vocab = 1024;
  int prompt_tokens = 32;
  int decode_tokens = 16;
  int ffn_mult = 4;
  int tile = 32;
  std::uint64_t seed = 1;
  std::string matmul = "tiled";
};

struct XorShift64 {
  std::uint64_t state;

  explicit XorShift64(std::uint64_t seed) : state(seed ? seed : 1) {}

  std::uint32_t next_u32() {
    std::uint64_t x = state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    state = x;
    return static_cast<std::uint32_t>(x >> 32);
  }

  float uniform(float scale) {
    const float unit = static_cast<float>(next_u32()) / 4294967295.0f;
    return (2.0f * unit - 1.0f) * scale;
  }
};

struct Layer {
  std::vector<float> rms_att;
  std::vector<float> rms_ffn;
  std::vector<float> wq;
  std::vector<float> wk;
  std::vector<float> wv;
  std::vector<float> wo;
  std::vector<float> w1;
  std::vector<float> w2;
  std::vector<float> k_cache;
  std::vector<float> v_cache;
};

struct Model {
  Config cfg;
  int head_dim = 0;
  int ffn_dim = 0;
  int max_seq = 0;
  std::vector<float> token_embedding;
  std::vector<float> final_norm;
  std::vector<float> lm_head;
  std::vector<Layer> layers;
};

struct ForwardResult {
  int token = 0;
  double logits_checksum = 0.0;
};

void usage(const char* argv0) {
  std::cerr << "usage: " << argv0
            << " [--layers N] [--dim N] [--heads N] [--vocab N]\n"
            << "       [--prompt N] [--decode N] [--ffn-mult N]\n"
            << "       [--matmul naive|tiled] [--tile N] [--seed N]\n";
}

int parse_int(const char* value, const std::string& name) {
  char* end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (!end || *end != '\0' || parsed <= 0 ||
      parsed > static_cast<long>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("invalid positive integer for " + name);
  }
  return static_cast<int>(parsed);
}

std::uint64_t parse_u64(const char* value, const std::string& name) {
  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(value, &end, 10);
  if (!end || *end != '\0') {
    throw std::invalid_argument("invalid unsigned integer for " + name);
  }
  return static_cast<std::uint64_t>(parsed);
}

Config parse_args(int argc, char** argv) {
  Config cfg;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto require_value = [&](const std::string& name) -> const char* {
      if (i + 1 >= argc) {
        throw std::invalid_argument("missing value for " + name);
      }
      return argv[++i];
    };
    if (arg == "--help" || arg == "-h") {
      usage(argv[0]);
      std::exit(0);
    } else if (arg == "--layers") {
      cfg.layers = parse_int(require_value(arg), arg);
    } else if (arg == "--dim") {
      cfg.dim = parse_int(require_value(arg), arg);
    } else if (arg == "--heads") {
      cfg.heads = parse_int(require_value(arg), arg);
    } else if (arg == "--vocab") {
      cfg.vocab = parse_int(require_value(arg), arg);
    } else if (arg == "--prompt") {
      cfg.prompt_tokens = parse_int(require_value(arg), arg);
    } else if (arg == "--decode") {
      cfg.decode_tokens = parse_int(require_value(arg), arg);
    } else if (arg == "--ffn-mult") {
      cfg.ffn_mult = parse_int(require_value(arg), arg);
    } else if (arg == "--tile") {
      cfg.tile = parse_int(require_value(arg), arg);
    } else if (arg == "--seed") {
      cfg.seed = parse_u64(require_value(arg), arg);
    } else if (arg == "--matmul") {
      cfg.matmul = require_value(arg);
    } else {
      throw std::invalid_argument("unknown argument: " + arg);
    }
  }
  if (cfg.dim % cfg.heads != 0) {
    throw std::invalid_argument("--dim must be divisible by --heads");
  }
  if ((cfg.dim / cfg.heads) % 2 != 0) {
    throw std::invalid_argument("head dimension must be even for RoPE");
  }
  if (cfg.matmul != "naive" && cfg.matmul != "tiled") {
    throw std::invalid_argument("--matmul must be 'naive' or 'tiled'");
  }
  return cfg;
}

void fill_random(std::vector<float>& values, XorShift64& rng, float scale) {
  for (float& value : values) {
    value = rng.uniform(scale);
  }
}

std::vector<float> random_vector(std::size_t size, XorShift64& rng, float scale) {
  std::vector<float> values(size);
  fill_random(values, rng, scale);
  return values;
}

Model build_model(const Config& cfg) {
  Model model;
  model.cfg = cfg;
  model.head_dim = cfg.dim / cfg.heads;
  model.ffn_dim = cfg.ffn_mult * cfg.dim;
  model.max_seq = cfg.prompt_tokens + cfg.decode_tokens + 1;

  XorShift64 rng(cfg.seed);
  const float embed_scale = 0.05f;
  const float mat_scale = 1.0f / std::sqrt(static_cast<float>(cfg.dim));

  model.token_embedding = random_vector(
      static_cast<std::size_t>(cfg.vocab) * cfg.dim, rng, embed_scale);
  model.final_norm.assign(cfg.dim, 1.0f);
  model.lm_head = random_vector(
      static_cast<std::size_t>(cfg.vocab) * cfg.dim, rng, mat_scale);

  model.layers.reserve(cfg.layers);
  for (int layer_idx = 0; layer_idx < cfg.layers; ++layer_idx) {
    Layer layer;
    layer.rms_att.assign(cfg.dim, 1.0f);
    layer.rms_ffn.assign(cfg.dim, 1.0f);
    const std::size_t d2 = static_cast<std::size_t>(cfg.dim) * cfg.dim;
    layer.wq = random_vector(d2, rng, mat_scale);
    layer.wk = random_vector(d2, rng, mat_scale);
    layer.wv = random_vector(d2, rng, mat_scale);
    layer.wo = random_vector(d2, rng, mat_scale);
    layer.w1 = random_vector(
        static_cast<std::size_t>(model.ffn_dim) * cfg.dim, rng, mat_scale);
    layer.w2 = random_vector(
        static_cast<std::size_t>(cfg.dim) * model.ffn_dim, rng, mat_scale);
    layer.k_cache.assign(static_cast<std::size_t>(model.max_seq) * cfg.dim, 0.0f);
    layer.v_cache.assign(static_cast<std::size_t>(model.max_seq) * cfg.dim, 0.0f);
    model.layers.push_back(std::move(layer));
  }
  return model;
}

void rmsnorm(const std::vector<float>& x, const std::vector<float>& weight,
             std::vector<float>& out) {
  double sum_sq = 0.0;
  for (float value : x) {
    sum_sq += static_cast<double>(value) * value;
  }
  const float inv = 1.0f / std::sqrt(static_cast<float>(sum_sq / x.size()) + 1e-5f);
  for (std::size_t i = 0; i < x.size(); ++i) {
    out[i] = x[i] * inv * weight[i];
  }
}

void matmul_naive(const std::vector<float>& x, const std::vector<float>& w, int rows,
                  int cols, std::vector<float>& out) {
  for (int r = 0; r < rows; ++r) {
    const float* row = &w[static_cast<std::size_t>(r) * cols];
    float acc = 0.0f;
    for (int c = 0; c < cols; ++c) {
      acc += row[c] * x[c];
    }
    out[r] = acc;
  }
}

void matmul_tiled(const std::vector<float>& x, const std::vector<float>& w, int rows,
                  int cols, int tile, std::vector<float>& out) {
  for (int r = 0; r < rows; ++r) {
    const float* row = &w[static_cast<std::size_t>(r) * cols];
    float acc = 0.0f;
    for (int start = 0; start < cols; start += tile) {
      const int end = std::min(start + tile, cols);
      for (int c = start; c < end; ++c) {
        acc += row[c] * x[c];
      }
    }
    out[r] = acc;
  }
}

void matmul_vec(const Config& cfg, const std::vector<float>& x,
                const std::vector<float>& w, int rows, int cols,
                std::vector<float>& out) {
  if (cfg.matmul == "naive") {
    matmul_naive(x, w, rows, cols, out);
  } else {
    matmul_tiled(x, w, rows, cols, cfg.tile, out);
  }
}

void apply_rope(std::vector<float>& x, int position, int heads) {
  const int dim = static_cast<int>(x.size());
  const int head_dim = dim / heads;
  for (int h = 0; h < heads; ++h) {
    const int base = h * head_dim;
    for (int i = 0; i < head_dim; i += 2) {
      const float inv_freq = 1.0f / std::pow(10000.0f, static_cast<float>(i) / head_dim);
      const float angle = static_cast<float>(position) * inv_freq;
      const float c = std::cos(angle);
      const float s = std::sin(angle);
      const float a = x[base + i];
      const float b = x[base + i + 1];
      x[base + i] = a * c - b * s;
      x[base + i + 1] = a * s + b * c;
    }
  }
}

void causal_attention(const Model& model, const Layer& layer, const std::vector<float>& q,
                      int position, std::vector<float>& out) {
  const int dim = model.cfg.dim;
  const int heads = model.cfg.heads;
  const int head_dim = model.head_dim;
  std::fill(out.begin(), out.end(), 0.0f);

  std::vector<float> scores(position + 1);
  for (int h = 0; h < heads; ++h) {
    const int base = h * head_dim;
    float max_score = -std::numeric_limits<float>::infinity();
    for (int t = 0; t <= position; ++t) {
      const float* key = &layer.k_cache[static_cast<std::size_t>(t) * dim + base];
      float dot = 0.0f;
      for (int d = 0; d < head_dim; ++d) {
        dot += q[base + d] * key[d];
      }
      const float score = dot / std::sqrt(static_cast<float>(head_dim));
      scores[t] = score;
      max_score = std::max(max_score, score);
    }

    float denom = 0.0f;
    for (int t = 0; t <= position; ++t) {
      scores[t] = std::exp(scores[t] - max_score);
      denom += scores[t];
    }

    for (int t = 0; t <= position; ++t) {
      const float p = scores[t] / denom;
      const float* value = &layer.v_cache[static_cast<std::size_t>(t) * dim + base];
      for (int d = 0; d < head_dim; ++d) {
        out[base + d] += p * value[d];
      }
    }
  }
}

float silu(float x) { return x / (1.0f + std::exp(-x)); }

ForwardResult forward_token(Model& model, int token, int position) {
  const Config& cfg = model.cfg;
  const int dim = cfg.dim;
  std::vector<float> x(dim);
  std::copy_n(&model.token_embedding[static_cast<std::size_t>(token) * dim], dim, x.begin());

  std::vector<float> norm(dim);
  std::vector<float> q(dim);
  std::vector<float> k(dim);
  std::vector<float> v(dim);
  std::vector<float> att(dim);
  std::vector<float> proj(dim);
  std::vector<float> ff_in(dim);
  std::vector<float> hidden(model.ffn_dim);
  std::vector<float> ff_out(dim);

  for (Layer& layer : model.layers) {
    rmsnorm(x, layer.rms_att, norm);
    matmul_vec(cfg, norm, layer.wq, dim, dim, q);
    matmul_vec(cfg, norm, layer.wk, dim, dim, k);
    matmul_vec(cfg, norm, layer.wv, dim, dim, v);
    apply_rope(q, position, cfg.heads);
    apply_rope(k, position, cfg.heads);

    std::copy(k.begin(), k.end(),
              layer.k_cache.begin() + static_cast<std::ptrdiff_t>(position) * dim);
    std::copy(v.begin(), v.end(),
              layer.v_cache.begin() + static_cast<std::ptrdiff_t>(position) * dim);

    causal_attention(model, layer, q, position, att);
    matmul_vec(cfg, att, layer.wo, dim, dim, proj);
    for (int i = 0; i < dim; ++i) {
      x[i] += proj[i];
    }

    rmsnorm(x, layer.rms_ffn, ff_in);
    matmul_vec(cfg, ff_in, layer.w1, model.ffn_dim, dim, hidden);
    for (float& value : hidden) {
      value = silu(value);
    }
    matmul_vec(cfg, hidden, layer.w2, dim, model.ffn_dim, ff_out);
    for (int i = 0; i < dim; ++i) {
      x[i] += ff_out[i];
    }
  }

  rmsnorm(x, model.final_norm, norm);
  std::vector<float> logits(cfg.vocab);
  matmul_vec(cfg, norm, model.lm_head, cfg.vocab, dim, logits);

  int best = 0;
  float best_value = logits[0];
  double checksum = 0.0;
  for (int i = 0; i < cfg.vocab; ++i) {
    if (logits[i] > best_value) {
      best_value = logits[i];
      best = i;
    }
    checksum += static_cast<double>(logits[i]) * static_cast<double>((i % 17) + 1);
  }
  return {best, checksum};
}

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

double median(std::vector<double> values) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

std::uint64_t weight_bytes(const Model& model) {
  std::uint64_t count = model.token_embedding.size() + model.final_norm.size() +
                        model.lm_head.size();
  for (const Layer& layer : model.layers) {
    count += layer.rms_att.size() + layer.rms_ffn.size() + layer.wq.size() +
             layer.wk.size() + layer.wv.size() + layer.wo.size() +
             layer.w1.size() + layer.w2.size();
  }
  return count * sizeof(float);
}

std::uint64_t kv_cache_bytes(const Model& model) {
  std::uint64_t count = 0;
  for (const Layer& layer : model.layers) {
    count += layer.k_cache.size() + layer.v_cache.size();
  }
  return count * sizeof(float);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Config cfg = parse_args(argc, argv);
    Model model = build_model(cfg);

    int token = 1 % cfg.vocab;
    double checksum = 0.0;

    const auto prefill_start = Clock::now();
    for (int pos = 0; pos < cfg.prompt_tokens; ++pos) {
      token = (pos * 31 + 7) % cfg.vocab;
      const ForwardResult result = forward_token(model, token, pos);
      checksum += result.logits_checksum;
      token = result.token;
    }
    const auto prefill_end = Clock::now();

    std::vector<double> step_ms;
    step_ms.reserve(cfg.decode_tokens);
    const auto decode_start = Clock::now();
    for (int i = 0; i < cfg.decode_tokens; ++i) {
      const int pos = cfg.prompt_tokens + i;
      const auto step_start = Clock::now();
      const ForwardResult result = forward_token(model, token, pos);
      const auto step_end = Clock::now();
      step_ms.push_back(elapsed_ms(step_start, step_end));
      checksum += result.logits_checksum;
      token = result.token;
    }
    const auto decode_end = Clock::now();

    const double prefill_ms = elapsed_ms(prefill_start, prefill_end);
    const double decode_ms = elapsed_ms(decode_start, decode_end);
    const double total_ms = prefill_ms + decode_ms;
    const int total_tokens = cfg.prompt_tokens + cfg.decode_tokens;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "{\n";
    std::cout << "  \"schema_version\": 1,\n";
    std::cout << "  \"implementation\": \"cpp/my.cpp\",\n";
    std::cout << "  \"mode\": \"synthetic_fp32_tiny_transformer\",\n";
    std::cout << "  \"matmul\": \"" << cfg.matmul << "\",\n";
    std::cout << "  \"layers\": " << cfg.layers << ",\n";
    std::cout << "  \"dim\": " << cfg.dim << ",\n";
    std::cout << "  \"heads\": " << cfg.heads << ",\n";
    std::cout << "  \"head_dim\": " << (cfg.dim / cfg.heads) << ",\n";
    std::cout << "  \"ffn_dim\": " << (cfg.ffn_mult * cfg.dim) << ",\n";
    std::cout << "  \"vocab\": " << cfg.vocab << ",\n";
    std::cout << "  \"prompt_tokens\": " << cfg.prompt_tokens << ",\n";
    std::cout << "  \"decode_tokens\": " << cfg.decode_tokens << ",\n";
    std::cout << "  \"tile\": " << cfg.tile << ",\n";
    std::cout << "  \"seed\": " << cfg.seed << ",\n";
    std::cout << "  \"prefill_ms\": " << prefill_ms << ",\n";
    std::cout << "  \"decode_ms\": " << decode_ms << ",\n";
    std::cout << "  \"total_ms\": " << total_ms << ",\n";
    std::cout << "  \"median_decode_step_ms\": " << median(step_ms) << ",\n";
    std::cout << "  \"decode_tokens_per_s\": "
              << (cfg.decode_tokens * 1000.0 / std::max(decode_ms, 1e-9)) << ",\n";
    std::cout << "  \"total_tokens_per_s\": "
              << (total_tokens * 1000.0 / std::max(total_ms, 1e-9)) << ",\n";
    std::cout << "  \"weight_bytes\": " << weight_bytes(model) << ",\n";
    std::cout << "  \"kv_cache_bytes\": " << kv_cache_bytes(model) << ",\n";
    std::cout << "  \"final_token\": " << token << ",\n";
    std::cout << "  \"checksum\": " << checksum << "\n";
    std::cout << "}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << "\n";
    usage(argv[0]);
    return 2;
  }
}
