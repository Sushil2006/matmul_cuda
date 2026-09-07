# Nsight Compute profiling observations

All profiles use `M=N=K=2048` on a Tesla T4 and capture one post-warm-up launch. The `time_ms` printed by `matmul` includes Nsight Compute replay overhead, so profiler `Duration` is used only where it was collected; use the separate `8192^3` benchmark for final performance claims. DRAM traffic is derived as `sector count * 32 bytes`.

## Naive vs shared-memory tiled

| Metric | Naive | Shared-memory tiled |
|---|---:|---:|
| NCU duration | 74.96 ms | 40.42 ms |
| Global-load instructions | 536.87M | 16.78M |
| L1 global-load sectors | 1,073.72M | 67.11M |
| Useful bytes/global-load sector | 18/32 | 32/32 |
| Derived DRAM read traffic | 4.325 GB | 1.585 GB |
| LG-throttle stall cycles | 35.63 | 0.01 |
| Issued warps/scheduler/cycle | 0.18 | 0.27 |

- Shared-memory tiling is `1.85x` faster. It eliminates redundant global loads (`32x` fewer instructions), improves sector use, and cuts DRAM reads by `63.3%`.
- The naive kernel is dominated by LG-throttle stalls. Tiling removes that bottleneck and raises the issue rate; its remaining pressure is shared-memory throughput, not bank conflicts.

## Shared-memory tiled vs block tiled

This compares the selected configurations: shared memory `(32,32,128)` and block tiling `(128,128,32,8,8)`.

| Metric | Shared-memory tiled | Block tiled |
|---|---:|---:|
| Shared-load instructions | 335.54M | 41.94M |
| Executed instructions | 1,015.68M | 399.60M |
| MIO-throttle cycles/instruction | 19.8 | 5.5 |
| Warp cycles/instruction | 29.59 | 11.12 |
| Achieved occupancy | 99.97% | 46.98% |

- Register tiling reduces shared loads `8x` and total instructions `60.7%`, cutting MIO pressure and warp cycles per instruction by about `62%`.
- It needs `122` rather than `60` registers/thread, halving occupancy, but the reduced shared-memory work repays that cost.

## Block tiled vs bank-conflict-free

Both kernels use `(BM,BN,BK,TM,TN)=(128,128,32,8,8)`.

| Metric | Block tiled | Bank-conflict-free |
|---|---:|---:|
| Shared-load bank conflicts | 67.15M | 11.27K |
| Shared-load wavefronts | 134.26M | 67.12M |
| Eligible warps/scheduler | 0.51 | 1.39 |
| Issued warps/scheduler/cycle | 0.34 | 0.59 |
| Warp cycles/instruction | 11.08 | 6.25 |
| Executed instructions | 399.60M | 443.66M |

- The new layout removes virtually all shared-load conflicts and halves shared-load wavefronts. That makes `2.7x` more warps eligible and cuts warp cycles/instruction by `43.6%`.
- The addressing transformation adds `11.0%` instructions, but its improved shared-memory service more than offsets that overhead.

## Bank-conflict-free vs vectorized

Both kernels use `(BM,BN,BK,TM,TN)=(128,128,32,8,8)`.

| Metric | Bank-conflict-free | Vectorized |
|---|---:|---:|
| NCU duration | 8.70 ms | 8.52 ms |
| Global-load instructions | 4.19M | 1.05M |
| L1 global-load sectors | 16.78M | 16.78M |
| Derived DRAM read traffic | 179.6 MB | 152.7 MB |
| Executed instructions | 443.66M | 401.52M |

- `float4` loads pack four scalar global-load instructions into one, giving a `4x` instruction reduction while requesting the same L1 sectors.
- Total instructions fall `9.5%` and duration improves `2.1%`. The modest speedup is expected because the kernel remains compute/shared-memory limited; DRAM traffic falls only `15.0%`.

## Vectorized vs warp tiled

This matched ablation fixes `(BM,BN,BK,TM,TN)=(128,128,32,8,8)`; it is not the final `BK=16, TN=4` warp configuration.

| Metric | Vectorized | Warp tiled |
|---|---:|---:|
| NCU duration | 8.52 ms | 8.73 ms |
| Shared-load instructions | 67.11M | 12.58M |
| Shared-store bank conflicts | 0.93M | 15.47M |
| Shared-store wavefronts | 5.12M | 19.65M |
| Achieved occupancy | 46.54% | 23.45% |

- Warp-local reuse cuts shared-load instructions `5.3x`, but this lane mapping makes shared stores `16.7x` more conflict-prone and nearly quadruples their wavefronts.
- The warp kernel also halves occupancy, so this matched `2048^3` configuration is `2.5%` slower despite its lower instruction count. The final winner uses a different `BK` and `TN` at `8192^3`.

## Warp WNITER=2 vs WNITER=4

Every parameter except `WNITER` is identical.

| Metric | WNITER=2 | WNITER=4 |
|---|---:|---:|
| Executed instructions | 312.18M | 312.15M |
| Shared-store wavefronts | 10.49M | 10.49M |
| Eligible warps/scheduler | 0.68 | 0.66 |
| Warp cycles/instruction | 4.08 | 4.17 |

- Resource use and bank behavior are unchanged: `WNITER` only changes the order in which each thread traverses its output subtiles.
- `WNITER=2` has a small scheduling edge (`3%` more eligible warps and `2.2%` fewer warp cycles/instruction), consistent with its measured timing advantage.

## Warp tile 64x64 vs 64x32

All parameters other than `WN` are fixed.

| Metric | 64x64 warp tile | 64x32 warp tile |
|---|---:|---:|
| NCU duration | 7.95 ms | 7.94 ms |
| Executed instructions | 312.18M | 326.16M |
| Warp cycles/instruction | 4.07 | 7.76 |
| Eligible warps/scheduler | 0.68 | 1.14 |
| Achieved occupancy | 23.24% | 46.56% |

- The `64x64` tile uses `4.3%` fewer instructions and each warp progresses with `47.6%` fewer cycles/instruction, but its larger register footprint halves occupancy.
- These effects cancel at `2048^3`. At `8192^3`, the larger grid provides enough parallelism for the `64x64` reuse advantage to win.
