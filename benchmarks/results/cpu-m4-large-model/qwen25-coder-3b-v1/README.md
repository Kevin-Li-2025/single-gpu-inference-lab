# Qwen2.5-Coder 3B On Apple M4

This artifact uses a real 3B model and a fixed code-generation prompt. The
llama.cpp rows use the exact same official GGUF Q4_K_M bytes. The MLX row uses
the same model architecture in MLX 4-bit format, so it is a system comparison,
not a bitwise-identical quantization comparison.

## Throughput

| Runtime | Prefill p512 | Decode tg128 | Real completion decode |
| --- | ---: | ---: | ---: |
| llama.cpp CPU | 115.02 tok/s | 33.03 tok/s | 34.84 tok/s |
| llama.cpp Metal | 408.28 tok/s | 44.24 tok/s | 46.92 tok/s |
| MLX Metal 4-bit | 257.91 tok/s | 54.72 tok/s | 54.72 tok/s |

Metal/CPU llama decode speedup: **1.34x**.
MLX/CPU llama real-completion speedup: **1.57x**.

The CPU thread sweep selected **4 threads**; using all ten M4 cores is slower
for this memory-bound decode workload.

| CPU threads | Decode tg128 |
| ---: | ---: |
| 4 | 33.70 tok/s |
| 6 | 31.26 tok/s |
| 8 | 32.50 tok/s |
| 10 | 17.27 tok/s |

## Correctness Boundary

- llama.cpp CPU and Metal completion outputs exact: `true`;
- MLX repeated output stable: `true`;
- GGUF SHA-256: `724fb256bec1ff062b2f65e4569e871ad2e95ab2a3989723d1769c54294730b7`;
- no mock tensors or synthetic model weights are used in this artifact.

## SME2 Follow-up

The accompanying M4 probe uses Arm KleidiAI's QSI4/QAI8 SME2 GEMV on Qwen 3B
FFN shapes. It is kept as a kernel-boundary result until GGUF weights are
repacked and the path is integrated into full decode.

| Qwen 3B FFN shape | KleidiAI NEON | KleidiAI SME2 | SME2 speedup |
| --- | ---: | ---: | ---: |
| `1x2048 @ 2048x11008` | 226.25 us | 161.91 us | 1.40x |
| `1x11008 @ 11008x2048` | 225.60 us | 163.75 us | 1.38x |

The SME2 probe passes 154/154 upstream correctness cases. KleidiAI v1.28.0's
benchmark process reports complete medians and then exits with `SIGSEGV` on
this macOS host; `sme2-kernel-probe.json` records that teardown status. These
rows are external-kernel research evidence, not a claim about code authored in
this repository.

## Sources

- [Official Qwen2.5-Coder-3B-Instruct GGUF](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct-GGUF)
- [MLX LM](https://github.com/ml-explore/mlx-lm)
- [Arm KleidiAI](https://github.com/ARM-software/kleidiai)
