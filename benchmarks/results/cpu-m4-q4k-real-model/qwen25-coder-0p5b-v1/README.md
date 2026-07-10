# Real Qwen Q4_K Apple M4 A/B

This artifact closes the first real-model gate for the self-written Apple M4
kernel. `cpp/m4_q4k_gguf.cpp` parses the Qwen GGUF v3 file directly, mmaps real
Q4_K tensor bytes, and validates the custom NEON math against llama.cpp. The
opt-in llama.cpp patch then places that same kernel in real decode.

## Correctness And Path Proof

- parsed 291 GGUF tensors without a model framework;
- found 12 Q4_K FFN-down tensors with shape `4864 x 896`;
- real `blk.2.ffn_down.weight` maximum absolute difference versus llama.cpp:
  `0.000001`;
- all 4/4 completion A/B output pairs are byte-identical;
- all 4/4 candidate runs emit the custom-kernel hit trace.

## Performance

Apple M4, Qwen2.5-Coder-0.5B-Instruct, 6 CPU threads:

| Path | Decode throughput | Relative to llama baseline |
| --- | ---: | ---: |
| llama.cpp repacked GGUF Q4_K_M, `tg128` | 165.261 tok/s | 1.000x |
| custom raw Q4_K kernel, `tg128` | 164.772 tok/s | 0.997x |
| llama.cpp real completion median | 166.995 tok/s | 1.000x |
| custom real completion median | 166.180 tok/s | 0.995x |
| MLX persistent same-model 4-bit | 263.553 tok/s | 1.578x |

MLX uses its own 4-bit format and Metal backend, so it is a same-model system
comparison rather than a bitwise-identical quantization comparison.

## Decision

The raw-row custom NEON path is correct and reaches real decode, but it does not
beat llama.cpp's repacked Q4_K path. Keep it opt-in. The next optimization target
is the repacked 8-row GEMV boundary using SME2 or a custom Metal path; replacing
repacking with raw rows is not a production win.
