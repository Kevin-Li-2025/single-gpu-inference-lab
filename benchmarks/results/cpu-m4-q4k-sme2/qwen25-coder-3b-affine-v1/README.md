# Qwen 3B Q4_K Affine SME2 Gate

This artifact uses the official Qwen2.5-Coder-3B-Instruct Q4_K_M GGUF on a
16 GiB Apple M4. No mock weights are used.

## Result

| Boundary | Baseline | Candidate | Candidate / baseline |
| --- | ---: | ---: | ---: |
| FFN up `11008 x 2048`, cold median | 478.50 us custom raw NEON | 422.73 us SME2 + correction | 1.132x |
| FFN down `2048 x 11008`, cold median | 484.31 us custom raw NEON | 418.19 us SME2 + correction | 1.158x |
| Full decode `tg128`, median | 33.98 tok/s llama x8 | 29.13 tok/s hybrid | 0.857x |
| Real prompt decode | 33.26 tok/s | 28.69 tok/s | 0.863x |
| Qualified `tg128`, 6x5 triangle | 33.6642 tok/s llama x8 | 32.6287 tok/s parallel | 0.9692x |
| Qualified correction control | 32.6358 tok/s serial | 32.6287 tok/s parallel | 0.9998x |

The two real-tensor kernel gates pass. The full-decode gate fails. The opt-in
integration therefore remains disabled by default.

## Correctness

- Q4 nibble values are preserved; the transform does not requantize weights.
- The missing Q4_K affine term is restored as
  `(8 * rounded_scale - dmin * minimum) * sum(x_block)`.
- Mapping normalized RMSE is `1.7e-7` for FFN up and `3.7e-7` for FFN down.
- The fixed greedy completion is byte-identical, with SHA-256
  `8ff2391976022289e0b35ded5071463b329a85a615884fdac0febe44a1151c59`.

## Boundary

The candidate keeps original Q4_K bytes for fallback, SME2-packed weights,
FP32 affine coefficients, and llama's x8 fallback layout. The measured model
buffers total about 6.01 GiB. It also shows unstable `tg128` tail latency:
candidate p95/p99 ITL is 40.36/40.58 ms versus 30.16/30.30 ms for baseline.

See `docs/m4-q4k-sme2-case-study.md` for the implementation and failure
analysis. Raw benchmark rows are kept beside this file.

## Qualified Follow-up

The AC-qualified follow-up uses `ffn_down` only, one shared Q8_K activation
pack, 25% SME2 rows, and a rotated native/serial/parallel triangle. It runs six
pairs, five `tg128` repetitions per mode and pair, with each ordering repeated
twice. All three greedy outputs are byte-identical.

Both performance gates fail. Parallel reaches `0.9692x` versus native llama
and `0.9998x` versus serial correction; minimum pair speedups are `0.9480x`
and `0.9691x`. The full SME2 route stays disabled, and parallel correction is
now opt-in through `GGML_M4_Q4K_SME2_PARALLEL_CORRECTION=1`. See
`qualified-triangle.json` for the compact formal evidence.
