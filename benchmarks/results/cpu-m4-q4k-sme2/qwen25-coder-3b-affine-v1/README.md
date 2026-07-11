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

## Follow-up status

The integration now defaults to `ffn_down` only, shares one Q8_K activation
pack across fallback workers, assigns 25% of rows to SME2, and overlaps affine
correction with the x8 fallback work. The serial correction path remains
available through `GGML_M4_Q4K_SME2_PARALLEL_CORRECTION=0` for same-binary A/B.
The original negative result above remains the formal evidence. A new AC-power
interleaved A/B must pass `scripts/run_m4_q4k_sme2_ab.py` before this artifact
can be superseded; battery diagnostics are intentionally not committed as
performance evidence. The final campaign should use `--include-serial-control`
to report native llama, serial correction, and parallel correction in one
rotated three-way experiment.
