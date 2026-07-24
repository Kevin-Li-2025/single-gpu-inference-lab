# qwen3-0p6b-c2c4c8-o32o64-r64-v1

> **Superseded performance comparison:** all recorded deltas are
> historical; only the trace/path structure remains current evidence.

This artifact summarizes a native-vs-standalone-vs-fused repetition-penalty
serving matrix on the L20 vLLM path.

## Recorded summary

- Rows: `4`
- Workload-signature matches: `4`
- Performance comparable: `False`
- Historical fused median ITL positives: `4`
- Historical standalone median ITL positives: `1`
- Historical fused median E2E positives: `4`

## Rows

| Row | c | input | output | prompts | Standalone ITL | Fused ITL | Fused E2E | Fused trace |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `c2_i512_o32_r64` | 2 | 512 | 32 | 64 | -4.093% | +0.562% | +0.801% | 33/35 |
| `c4_i512_o32_r64` | 4 | 512 | 32 | 64 | +2.475% | +5.859% | +8.603% | 34/36 |
| `c4_i512_o64_r64` | 4 | 512 | 64 | 64 | -1.824% | +4.092% | +3.980% | 34/36 |
| `c8_i512_o32_r64` | 8 | 512 | 32 | 64 | -2.908% | +2.430% | +2.330% | 19/37 |

## Claim Boundary

- This is a serving matrix, but each row is still scoped to its model and traffic shape.
- Latency rows are no-trace runs; trace sub-runs are path proof only.
- Positive and negative rows are not current evidence until native-equivalent semantic parity is independently verified.
- Standalone logits-processor rows remain useful as the architecture-control baseline.
