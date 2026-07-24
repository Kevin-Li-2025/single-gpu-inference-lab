# L20 Boundary Impact

Positive serving impact means latency reduction or throughput improvement. Negative values are regressions. Empty cells are unimplemented or not measured.

| Boundary | Status | Micro speedup | Serving impact | GPU time | Budget | Decision |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| RoPE + paged KV append | `confirmed_kernel_amdahl_limited` | 7.699x | +0.58% | 1.58% |  | `case_study_evidence_not_next_target` |
| Q/K norm + Q/K RoPE + KV write | `o2_path_proven_small_fraction` | 1.470x | +4.52% | 1.58% |  | `do_not_micro_optimize_alone` |
| FlashInfer sampling route | `production_route_confirmed` |  | +8.73% | 3.42% |  | `harden_existing_route` |
| Self-written standalone sampler | `superseded_semantics` |  |  | 3.42% |  | `keep_disabled` |
| Standalone LM-head top-k | `negative_micro_result` | 0.979x |  | 62.10% |  | `avoid_standalone_replacement` |
| Batched LM-head greedy top-1 | `positive_greedy_micro_only` | 1.051x |  | 62.10% |  | `epilogue_prototype_only` |
| FlashSampling-style LM-head Gumbel | `positive_gumbel_micro_only` | 1.084x |  | 62.10% |  | `current_epilogue_target` |
| LM-head/logits epilogue | `active_p0_budget` |  |  | 62.10% | 96.00% / 339.93 MiB | `next_core_module` |

## Reading The Table

- RoPE/KV and Q/K fusion rows show why micro wins are not enough.
- The standalone sampler row is superseded; standalone LM-head top-k remains a valid negative control.
- Batched greedy top-1 is a positive micro signal, not a serving claim.
- FlashSampling-style Gumbel is the current positive epilogue target, still micro-only.
- The logits epilogue row is not a speed claim; it is the measured budget for the next implementation.
