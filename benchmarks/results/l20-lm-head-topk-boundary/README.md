# L20 LM-Head to Top-k Boundary

This directory measures whether the next sampler optimization should fuse
logits production with top-k/top-p sampling.

The experiment compares:

- full logits: `hidden @ weight.T` followed by `torch.topk`;
- chunked top-k: split vocab into chunks and merge per-chunk top-k candidates,
  avoiding one full `[batch, vocab]` logits tensor;
- experimental Triton direct LM-head top-1: compute top-1 without materializing
  logits.

## Main Result

For the Qwen2.5-Coder-1.5B-shaped case (`hidden=1536`, `vocab=151936`) on one
NVIDIA L20, materializing logits is not the bottleneck. The full logits tensor
is only 0.29 MiB at batch 1 and 1.16 MiB at batch 4, while the LM-head weight
read is 445 MiB. Preserving the optimized GEMM path matters more than avoiding
the logits write.

## Top-k=50 Chunk Sweep

Shape: batch 4, hidden 1536, vocab 151936, FP16, top-k 50.

| Chunk vocab | Full logits + top-k | Chunked top-k | Chunked / full |
| ---: | ---: | ---: | ---: |
| 4,096 | 0.715 ms | 1.628 ms | 2.278x |
| 8,192 | 0.716 ms | 1.360 ms | 1.900x |
| 16,384 | 0.716 ms | 1.236 ms | 1.727x |
| 32,768 | 0.716 ms | 0.895 ms | 1.251x |
| 65,536 | 0.716 ms | 0.821 ms | 1.147x |
| 131,072 | 0.717 ms | 0.785 ms | 1.096x |

Larger chunks approach the full-GEMM baseline, but none beat it.

## Direct Triton Top-1 Sweep

Shape: batch 1, hidden 1536, vocab 151936, FP16, top-k 1.

| Block vocab | Block hidden | Full logits top-1 | Triton direct top-1 | Triton / full |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 64 | 0.661 ms | 0.682 ms | 1.032x |
| 16 | 128 | 0.660 ms | 0.675 ms | 1.022x |
| 64 | 64 | 0.660 ms | 0.711 ms | 1.076x |
| 64 | 128 | 0.661 ms | 0.675 ms | 1.022x |
| 32 | 64 | 0.660 ms | 0.703 ms | 1.065x |

The custom kernel is correct, but it is still slower than the cuBLAS/CUTLASS
full-logits path. This is a useful negative result: a standalone Triton LM-head
replacement is not the right next step.

## Batched Direct Triton Top-1 Sweep

Shape: batch 4, hidden 1536, vocab 151936, FP16, top-k 1. The batched
partial kernel computes four rows per partial program and avoids materializing
the `[batch, vocab]` logits tensor for greedy top-1 only.

| Block vocab | Block hidden | Full logits top-1 | Triton direct top-1 | Triton / full |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 64 | 0.712 ms | 0.689 ms | 0.968x |
| 16 | 128 | 0.712 ms | 0.682 ms | 0.959x |
| 32 | 64 | 0.712 ms | 0.714 ms | 1.003x |
| 32 | 128 | 0.712 ms | 0.680 ms | 0.955x |
| 64 | 64 | 0.711 ms | 0.721 ms | 1.014x |
| 64 | 128 | 0.712 ms | 0.677 ms | 0.952x |

The best measured L20 policy for this batch-4 greedy shape is therefore
`block_vocab=64`, `block_hidden=128`, with a 4.8% median microbenchmark speedup
over full logits top-1. The default experimental policy now uses the old
`32/64` path for batch 1 and the measured `64/128` path for batch >1.

This is the first positive self-written LM-head boundary micro signal in this
repo. It is still not a serving claim: the path only covers greedy top-1, still
uses a separate reduction kernel, and does not implement top-k/top-p, penalties,
logprobs, structured-output masks, or vLLM scheduler semantics.

## Conclusion

Chunked logits and batch-1 direct top-1 do not beat the optimized full-logits
path. A batched direct top-1 kernel can win a narrow greedy microbenchmark, but
it is too limited to replace production sampling. To move serving ITL, the work
still has to happen at a production LM-head/GEMM epilogue or an upstreamable
vLLM/FlashInfer/CUTLASS integration where sampler state is produced while the
optimized LM-head GEMM is already running.

## Artifacts

- `qwen25-b4-h1536-v151936-k50-v1.json`
- `qwen25-b4-h1536-v151936-k50-cv4096.json`
- `qwen25-b4-h1536-v151936-k50-cv16384.json`
- `qwen25-b4-h1536-v151936-k50-cv32768.json`
- `qwen25-b4-h1536-v151936-k50-cv65536.json`
- `qwen25-b4-h1536-v151936-k50-cv131072.json`
- `qwen25-b1-h1536-v151936-k1-v1.json`
- `qwen25-b1-h1536-v151936-k1-bv16-bh64.json`
- `qwen25-b1-h1536-v151936-k1-bv16-bh128.json`
- `qwen25-b1-h1536-v151936-k1-bv64-bh64.json`
- `qwen25-b1-h1536-v151936-k1-bv64-bh128.json`
- `qwen25-b4-h1536-v151936-k1-batched-v1.json`
- `qwen25-b4-h1536-v151936-k1-batched-policy-v2.json`
- `qwen25-b4-h1536-v151936-k1-batched-bv16-bh64.json`
- `qwen25-b4-h1536-v151936-k1-batched-bv16-bh128.json`
- `qwen25-b4-h1536-v151936-k1-batched-bv32-bh64.json`
- `qwen25-b4-h1536-v151936-k1-batched-bv32-bh128.json`
- `qwen25-b4-h1536-v151936-k1-batched-bv64-bh64.json`
- `qwen25-b4-h1536-v151936-k1-batched-bv64-bh128.json`
- `smoke-b1-h512-v8192-k1.json`
