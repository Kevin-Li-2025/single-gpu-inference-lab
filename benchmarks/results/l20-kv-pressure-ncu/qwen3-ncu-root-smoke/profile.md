# Nsight Roofline Summary

| Kernel | AI FLOP/B | Roofline | DRAM GB/s | DRAM % | L2 % | L1 hit % | SM % | Active warps % | Reg/thread | Long scoreboard % | Sector excess |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| void reshape_and_cache_flash_kernel<__nv_bfloat16, __nv_bfloat16, 0>(const T1 *, const T1 *, T2 *, T2 *, const long *, long, long, long, long, long, int, int, int, const float *, const float *, int) | n/a | n/a | 11.29 | n/a | 2.01 | 93.75 | 0.43 | 32.61 | 40 | 21.82 | n/a |

Null values mean Nsight Compute did not emit that metric in the imported CSV; they are not inferred.
