#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err__ = (call);                                                 \
    if (err__ != cudaSuccess) {                                                 \
      std::ostringstream oss__;                                                 \
      oss__ << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "        \
            << cudaGetErrorString(err__);                                      \
      throw std::runtime_error(oss__.str());                                   \
    }                                                                          \
  } while (0)

namespace {

struct BenchCase {
  int batch;
  int vocab;
  int unique_tokens;
};

struct CaseResult {
  std::string gpu_name;
  int compute_major;
  int compute_minor;
  int batch;
  int vocab;
  int unique_tokens;
  float dense_ms;
  float sparse_ms;
  float speedup;
  float max_abs_diff;
  float dense_effective_gbs;
  float sparse_effective_gbs;
  std::string policy_provider;
  float policy_ms;
  float policy_speedup;
  float policy_regret;
};

__device__ __forceinline__ float apply_repetition_penalty(float x,
                                                          float penalty) {
  return x > 0.0f ? x / penalty : x * penalty;
}

__global__ void dense_mask_penalty_kernel(float *logits, const uint8_t *mask,
                                          int total, float penalty) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = idx; i < total; i += stride) {
    float value = logits[i];
    if (mask[i] != 0) {
      value = apply_repetition_penalty(value, penalty);
    }
    logits[i] = value;
  }
}

__global__ void sparse_token_penalty_kernel(float *logits,
                                            const int *token_ids,
                                            const int *lengths, int batch,
                                            int max_tokens, int vocab,
                                            float penalty) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch * max_tokens;
  if (idx >= total) {
    return;
  }

  int row = idx / max_tokens;
  int col = idx - row * max_tokens;
  if (col >= lengths[row]) {
    return;
  }

  int token = token_ids[idx];
  float *value = logits + row * vocab + token;
  *value = apply_repetition_penalty(*value, penalty);
}

int ceil_div(int x, int y) { return (x + y - 1) / y; }

bool should_use_sparse_policy(const BenchCase &c) {
  const long long dense_elements =
      static_cast<long long>(c.batch) * static_cast<long long>(c.vocab);
  return c.vocab >= 65536 && dense_elements >= 524288 &&
         c.unique_tokens <= 1024;
}

std::vector<float> make_logits(int batch, int vocab, int seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-6.0f, 6.0f);
  std::vector<float> logits(static_cast<size_t>(batch) * vocab);
  for (float &value : logits) {
    value = dist(rng);
  }
  return logits;
}

void make_unique_tokens_and_mask(int batch, int vocab, int unique_tokens,
                                 std::vector<int> *token_ids,
                                 std::vector<int> *lengths,
                                 std::vector<uint8_t> *mask) {
  token_ids->assign(static_cast<size_t>(batch) * unique_tokens, -1);
  lengths->assign(batch, unique_tokens);
  mask->assign(static_cast<size_t>(batch) * vocab, 0);

  for (int row = 0; row < batch; ++row) {
    int produced = 0;
    int cursor = (row * 104729 + unique_tokens * 8191) % vocab;
    int step = 7919 + row * 131;
    while (produced < unique_tokens) {
      int token = cursor % vocab;
      uint8_t &seen = (*mask)[static_cast<size_t>(row) * vocab + token];
      if (seen == 0) {
        seen = 1;
        (*token_ids)[static_cast<size_t>(row) * unique_tokens + produced] =
            token;
        ++produced;
      }
      cursor += step;
    }
  }
}

float elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  return ms;
}

template <typename LaunchFn>
float time_kernel(LaunchFn launch, int warmup_iters, int iters) {
  for (int i = 0; i < warmup_iters; ++i) {
    launch();
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    launch();
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float per_iter = elapsed_ms(start, stop) / static_cast<float>(iters);
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaGetLastError());
  return per_iter;
}

float max_abs_diff(const std::vector<float> &a, const std::vector<float> &b) {
  if (a.size() != b.size()) {
    throw std::runtime_error("size mismatch in max_abs_diff");
  }
  float result = 0.0f;
  for (size_t i = 0; i < a.size(); ++i) {
    result = std::max(result, std::fabs(a[i] - b[i]));
  }
  return result;
}

CaseResult run_case(const BenchCase &c, int warmup_iters, int iters,
                    float penalty, const cudaDeviceProp &prop) {
  std::vector<float> logits = make_logits(c.batch, c.vocab,
                                          17 + c.batch + c.vocab +
                                              c.unique_tokens);
  std::vector<int> token_ids;
  std::vector<int> lengths;
  std::vector<uint8_t> mask;
  make_unique_tokens_and_mask(c.batch, c.vocab, c.unique_tokens, &token_ids,
                              &lengths, &mask);

  const size_t logits_bytes = logits.size() * sizeof(float);
  const size_t mask_bytes = mask.size() * sizeof(uint8_t);
  const size_t token_bytes = token_ids.size() * sizeof(int);
  const size_t length_bytes = lengths.size() * sizeof(int);

  float *d_original = nullptr;
  float *d_dense = nullptr;
  float *d_sparse = nullptr;
  uint8_t *d_mask = nullptr;
  int *d_token_ids = nullptr;
  int *d_lengths = nullptr;

  CUDA_CHECK(cudaMalloc(&d_original, logits_bytes));
  CUDA_CHECK(cudaMalloc(&d_dense, logits_bytes));
  CUDA_CHECK(cudaMalloc(&d_sparse, logits_bytes));
  CUDA_CHECK(cudaMalloc(&d_mask, mask_bytes));
  CUDA_CHECK(cudaMalloc(&d_token_ids, token_bytes));
  CUDA_CHECK(cudaMalloc(&d_lengths, length_bytes));

  CUDA_CHECK(cudaMemcpy(d_original, logits.data(), logits_bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_mask, mask.data(), mask_bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_token_ids, token_ids.data(), token_bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(d_lengths, lengths.data(), length_bytes, cudaMemcpyHostToDevice));

  const int dense_threads = 256;
  const int dense_blocks =
      std::min(4096, ceil_div(static_cast<int>(logits.size()), dense_threads));
  const int sparse_threads = 256;
  const int sparse_blocks =
      ceil_div(c.batch * c.unique_tokens, sparse_threads);

  auto launch_dense = [&]() {
    dense_mask_penalty_kernel<<<dense_blocks, dense_threads>>>(
        d_dense, d_mask, static_cast<int>(logits.size()), penalty);
  };
  auto launch_sparse = [&]() {
    sparse_token_penalty_kernel<<<sparse_blocks, sparse_threads>>>(
        d_sparse, d_token_ids, d_lengths, c.batch, c.unique_tokens, c.vocab,
        penalty);
  };

  CUDA_CHECK(cudaMemcpy(d_dense, d_original, logits_bytes,
                        cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(d_sparse, d_original, logits_bytes,
                        cudaMemcpyDeviceToDevice));
  launch_dense();
  launch_sparse();
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> dense_out(logits.size());
  std::vector<float> sparse_out(logits.size());
  CUDA_CHECK(cudaMemcpy(dense_out.data(), d_dense, logits_bytes,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(sparse_out.data(), d_sparse, logits_bytes,
                        cudaMemcpyDeviceToHost));
  float diff = max_abs_diff(dense_out, sparse_out);

  CUDA_CHECK(cudaMemcpy(d_dense, d_original, logits_bytes,
                        cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(d_sparse, d_original, logits_bytes,
                        cudaMemcpyDeviceToDevice));

  float dense_ms = time_kernel(launch_dense, warmup_iters, iters);
  float sparse_ms = time_kernel(launch_sparse, warmup_iters, iters);

  CUDA_CHECK(cudaFree(d_original));
  CUDA_CHECK(cudaFree(d_dense));
  CUDA_CHECK(cudaFree(d_sparse));
  CUDA_CHECK(cudaFree(d_mask));
  CUDA_CHECK(cudaFree(d_token_ids));
  CUDA_CHECK(cudaFree(d_lengths));

  double dense_bytes_per_iter =
      static_cast<double>(c.batch) * c.vocab * (sizeof(float) * 2 + 1);
  double sparse_bytes_per_iter =
      static_cast<double>(c.batch) * c.unique_tokens *
      (sizeof(int) + sizeof(float) * 2);

  CaseResult result;
  result.gpu_name = prop.name;
  result.compute_major = prop.major;
  result.compute_minor = prop.minor;
  result.batch = c.batch;
  result.vocab = c.vocab;
  result.unique_tokens = c.unique_tokens;
  result.dense_ms = dense_ms;
  result.sparse_ms = sparse_ms;
  result.speedup = dense_ms / sparse_ms;
  result.max_abs_diff = diff;
  result.dense_effective_gbs =
      static_cast<float>(dense_bytes_per_iter / (dense_ms * 1.0e6));
  result.sparse_effective_gbs =
      static_cast<float>(sparse_bytes_per_iter / (sparse_ms * 1.0e6));
  bool use_sparse = should_use_sparse_policy(c);
  result.policy_provider = use_sparse ? "sparse" : "dense";
  result.policy_ms = use_sparse ? sparse_ms : dense_ms;
  result.policy_speedup = dense_ms / result.policy_ms;
  result.policy_regret = result.policy_ms / std::min(dense_ms, sparse_ms);
  return result;
}

std::vector<BenchCase> default_cases(bool quick) {
  if (quick) {
    return {{1, 32000, 64}, {4, 32000, 256}};
  }
  return {
      {1, 32000, 64},   {1, 32000, 256},   {1, 32000, 1024},
      {4, 32000, 64},   {4, 32000, 256},   {4, 32000, 1024},
      {8, 32000, 64},   {8, 32000, 256},   {8, 32000, 1024},
      {1, 65536, 64},   {1, 65536, 256},   {1, 65536, 1024},
      {4, 65536, 64},   {4, 65536, 256},   {4, 65536, 1024},
      {8, 65536, 64},   {8, 65536, 256},   {8, 65536, 1024},
      {16, 65536, 64},  {16, 65536, 256},  {16, 65536, 1024},
      {32, 65536, 64},  {32, 65536, 256},  {32, 65536, 1024},
      {1, 151936, 64},  {1, 151936, 256},  {1, 151936, 1024},
      {4, 151936, 64},  {4, 151936, 256},  {4, 151936, 1024},
      {8, 151936, 64},  {8, 151936, 256},  {8, 151936, 1024},
      {16, 151936, 64}, {16, 151936, 256}, {16, 151936, 1024},
      {32, 151936, 64}, {32, 151936, 256}, {32, 151936, 1024},
  };
}

void print_table(const std::vector<CaseResult> &results) {
  std::cout << "| batch | vocab | unique history | dense ms | sparse ms | speedup | policy | policy speedup | regret | max diff |\n";
  std::cout << "|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|\n";
  for (const CaseResult &r : results) {
    std::cout << "| " << r.batch << " | " << r.vocab << " | "
              << r.unique_tokens << " | " << std::fixed
              << std::setprecision(4) << r.dense_ms << " | " << r.sparse_ms
              << " | " << std::setprecision(2) << r.speedup << "x | "
              << r.policy_provider << " | " << r.policy_speedup << "x | "
              << r.policy_regret << "x | "
              << std::setprecision(1) << r.max_abs_diff << " |\n";
  }
}

void write_csv(const std::string &path, const std::vector<CaseResult> &results) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("failed to open csv path: " + path);
  }
  out << "gpu,compute_cap,batch,vocab,unique_tokens,dense_ms,sparse_ms,"
         "speedup,max_abs_diff,dense_effective_gbs,sparse_effective_gbs,"
         "policy_provider,policy_ms,policy_speedup,policy_regret\n";
  for (const CaseResult &r : results) {
    out << '"' << r.gpu_name << '"' << ',' << r.compute_major << "."
        << r.compute_minor << ',' << r.batch << ',' << r.vocab << ','
        << r.unique_tokens << ',' << std::fixed << std::setprecision(6)
        << r.dense_ms << ',' << r.sparse_ms << ',' << r.speedup << ','
        << r.max_abs_diff << ',' << r.dense_effective_gbs << ','
        << r.sparse_effective_gbs << ',' << r.policy_provider << ','
        << r.policy_ms << ',' << r.policy_speedup << ','
        << r.policy_regret << '\n';
  }
}

struct Args {
  bool quick = false;
  int warmup_iters = 30;
  int iters = 200;
  float penalty = 1.1f;
  std::string csv_path;
};

Args parse_args(int argc, char **argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--quick") {
      args.quick = true;
      args.warmup_iters = 10;
      args.iters = 50;
    } else if (arg == "--iters" && i + 1 < argc) {
      args.iters = std::atoi(argv[++i]);
    } else if (arg == "--warmup" && i + 1 < argc) {
      args.warmup_iters = std::atoi(argv[++i]);
    } else if (arg == "--penalty" && i + 1 < argc) {
      args.penalty = std::atof(argv[++i]);
    } else if (arg == "--csv" && i + 1 < argc) {
      args.csv_path = argv[++i];
    } else if (arg == "--help" || arg == "-h") {
      std::cout
          << "usage: sparse_penalty_bench [--quick] [--iters N] [--warmup N]\n"
          << "                            [--penalty F] [--csv PATH]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown or incomplete argument: " + arg);
    }
  }
  if (args.iters <= 0 || args.warmup_iters < 0 || args.penalty <= 1.0f) {
    throw std::runtime_error("invalid benchmark arguments");
  }
  return args;
}

} // namespace

int main(int argc, char **argv) {
  try {
    Args args = parse_args(argc, argv);
    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    std::cerr << "GPU: " << prop.name << " sm_" << prop.major << prop.minor
              << ", iters=" << args.iters
              << ", warmup=" << args.warmup_iters
              << ", penalty=" << args.penalty << "\n";

    std::vector<CaseResult> results;
    for (const BenchCase &c : default_cases(args.quick)) {
      if (c.unique_tokens > c.vocab) {
        continue;
      }
      results.push_back(run_case(c, args.warmup_iters, args.iters,
                                 args.penalty, prop));
      const CaseResult &r = results.back();
      std::cerr << "case batch=" << r.batch << " vocab=" << r.vocab
                << " unique=" << r.unique_tokens << " dense=" << r.dense_ms
                << "ms sparse=" << r.sparse_ms << "ms speedup=" << r.speedup
                << "x policy=" << r.policy_provider
                << " policy_speedup=" << r.policy_speedup
                << "x regret=" << r.policy_regret
                << "x diff=" << r.max_abs_diff << "\n";
      if (r.max_abs_diff != 0.0f) {
        throw std::runtime_error("correctness check failed");
      }
    }

    print_table(results);
    if (!args.csv_path.empty()) {
      write_csv(args.csv_path, results);
    }
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
