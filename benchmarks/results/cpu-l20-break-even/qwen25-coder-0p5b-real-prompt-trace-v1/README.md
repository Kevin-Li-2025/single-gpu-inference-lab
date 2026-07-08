# Real Prompt Trace

This artifact runs fixed code-oriented prompts through the real vLLM HTTP
streaming path instead of a random-token benchmark.

## Summary

| Metric | Value |
| --- | ---: |
| Prompts | 12 / 12 completed |
| Concurrency | 4 |
| Request throughput | 9.233 req/s |
| Output throughput | 914.022 tok/s |
| Median TTFT | 26.198 ms |
| p95 TTFT | 522.091 ms |
| p99 TTFT | 522.563 ms |
| Median E2E | 300.042 ms |
| p95 E2E | 744.298 ms |
| p99 E2E | 744.797 ms |

## Interpretation

This is a small fixed-prompt trace, so its tail values should be read as
trace evidence rather than a stable service SLO. In this run the first
concurrency wave carries a visible TTFT tail, while the decode-side
inter-token latency stays tightly grouped.

- Median TTFT: 26.198 ms.
- p95 TTFT: 522.091 ms.
- Median per-prompt ITL: 2.142 ms.
- p99 per-prompt median ITL: 2.164 ms.

## Prompt Rows

| ID | Category | TTFT | Median ITL | E2E | Output tokens | Chunks |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `p01-python-cache-bug` | python-debugging | 522.681 ms | 2.144 ms | 744.922 ms | 96 | 96 |
| `p02-sql-window` | sql | 519.588 ms | 2.127 ms | 644.918 ms | 59 | 59 |
| `p03-typescript-narrowing` | typescript | 521.608 ms | 2.142 ms | 743.788 ms | 96 | 96 |
| `p04-pandas-groupby` | python-data | 520.318 ms | 2.142 ms | 742.908 ms | 96 | 96 |
| `p05-fastapi-endpoint` | backend | 23.916 ms | 2.165 ms | 313.109 ms | 128 | 128 |
| `p06-cpp-off-by-one` | cpp-debugging | 23.617 ms | 2.160 ms | 245.321 ms | 96 | 96 |
| `p07-shell-script` | shell | 22.916 ms | 2.145 ms | 179.854 ms | 73 | 73 |
| `p08-regex-parser` | python | 22.231 ms | 2.153 ms | 243.173 ms | 96 | 96 |
| `p09-unit-test` | testing | 20.890 ms | 2.133 ms | 247.792 ms | 96 | 95 |
| `p10-json-schema` | schema | 19.155 ms | 2.121 ms | 238.146 ms | 96 | 95 |
| `p11-cuda-explain` | cuda | 28.479 ms | 2.138 ms | 299.728 ms | 128 | 128 |
| `p12-code-review` | review | 29.523 ms | 2.130 ms | 300.356 ms | 128 | 128 |
