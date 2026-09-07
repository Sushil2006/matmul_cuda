# Nsight Compute profiling observations

All profiles use `M=N=K=2048` on a Tesla T4 and select one launch after the benchmark warm-up. The very large `time_ms` values printed by `matmul` are distorted by Nsight Compute's 26-28 replay passes and must not be compared. Use the profiler's `Duration` where it is available, and use the separate `8192^3` benchmark for final performance claims.

The rerun of naive vs shared-memory tiled uses `--print-details all` and now exposes the requested global-load instruction, sector, DRAM-sector, and warp-stall counters. DRAM traffic below is derived as `sector count * 32 bytes`. The other six profiles were captured without full tables; their recommendations expose selected conflict counts, but exact load, sector, and stall counters require rerunning them with the same flag. Across the profiles, warps have 32 active threads and approximately 31.6-31.96 non-predicated threads, so divergence is not a meaningful bottleneck.

## Naive vs shared-memory tiled

| Metric | Naive | Shared-memory tiled |
|---|---:|---:|
| NCU duration | 74.96 ms | 40.42 ms |
| Compute throughput | 61.24% | 78.21% |
| DRAM throughput | 18.13% | 12.39% |
| Memory throughput | 57.98 GB/s | 39.63 GB/s |
| Global-load instructions | 536,870,912 | 16,777,216 |
| L1 global-load sectors | 1,073,723,520 | 67,108,864 |
| Average useful bytes/global-load sector | 18/32 | 32/32 |
| DRAM read sectors | 135,140,826 | 49,544,075 |
| Derived DRAM read traffic | 4.325 GB | 1.585 GB |
| DRAM write sectors | 702,946 | 668,519 |
| L1/TEX global-load hit rate | 87.32% | 0% |
| L2 load hit rate | 50.53% | 48.11% |
| Shared-load instructions | 0 | 335,544,320 |
| Shared-store instructions | 0 | 16,777,216 |
| Shared-memory bank conflicts | 0 | 0 |
| Cycles with an eligible warp | 17.80% | 27.03% |
| Issued warps/scheduler/cycle | 0.18 | 0.27 |
| Warp cycles/issued instruction | 44.80 | 29.60 |
| LG-throttle stall | 35.63 | 0.01 |
| MIO-throttle stall | 0.01 | 19.84 |
| Long-scoreboard stall | 2.19 | 1.04 |
| Short-scoreboard stall | 0.01 | 0.55 |
| Registers/thread | 52 | 60 |
| Achieved occupancy | 99.63% | 100.08% |

- Shared-memory tiling reduces profiler duration by `46.1%`, or about `1.85x`, while occupancy remains effectively identical. The gain therefore comes from changing data movement rather than exposing more warps.
- Tiling cuts executed global-load instructions exactly `32x` and L1 global-load sectors `16x`. It also improves sector utilization from `18/32` to `32/32` useful bytes, confirming both reuse and better coalescing.
- DRAM reads fall from `135.14M` to `49.54M` sectors, equivalent to `4.325 GB -> 1.585 GB`, a `63.3%` reduction. This is smaller than the instruction reduction because the naive kernel serves `87.3%` of its repeated loads from L1; a high hit rate does not make the redundant load instructions free.
- The naive kernel's dominant stall is LG-throttle: `35.63` of its `44.80` warp cycles per issued instruction. Tiling nearly eliminates it (`0.01`), halves long-scoreboard stalls (`2.19 -> 1.04`), raises eligible-warp frequency from `17.8%` to `27.0%`, and raises issue rate from `0.18` to `0.27`.
- The bottleneck moves to shared-memory throughput: the tiled kernel executes `335.5M` shared loads and `16.8M` shared stores, producing `19.84` MIO-throttle stall cycles. Both shared-load and shared-store bank-conflict counters are zero, so this pressure comes from the volume of shared-memory instructions rather than conflicts.
- Short-scoreboard stalls rise from `0.01` to `0.55`, but remain small beside MIO-throttle. This makes reducing shared-load count, rather than fixing bank conflicts in this particular kernel, the clear next target for register/thread tiling.

## Shared-memory tiled vs block tiled

This compares each stage's chosen configuration, not a one-parameter ablation: shared memory uses `(32,32,128)`, whereas block tiling uses `(128,128,32,8,8)`.

| Metric | Shared-memory tiled | Block tiled |
|---|---:|---:|
| Executed instructions | 1,015,676,928 | 399,599,616 |
| MIO-throttle cycles/issued instruction | 19.8 | 5.5 |
| MIO share of warp cycles | 67.1% | 49.3% |
| Warp cycles/issued instruction | 29.60 | 11.08 |
| Issued warps/scheduler/cycle | 0.27 | 0.34 |
| Registers/thread | 60 | 122 |
| Static shared memory/block | 32.77 KB | 32.77 KB |
| Achieved occupancy | 99.97% | 46.96% |

- Block tiling executes `60.7%` fewer instructions and cuts MIO-throttle cycles by about `72%`. Each thread reuses its loaded A and B register values across an `8 x 8` output tile instead of repeatedly loading shared memory for a single output.
- Warp cycles per instruction fall by `62.6%`, and issue rate improves from `0.27` to `0.34`, despite active warps per scheduler falling from `8.00` to `3.75`.
- The cost is much higher register pressure: `60 -> 122` registers/thread. Registers and shared memory restrict the block kernel to two blocks/SM, producing `50%` theoretical and `46.96%` achieved occupancy.
- The reduced instruction and shared-memory demand more than repay the occupancy loss. High occupancy alone was not useful when the shared-memory instruction queue kept most warps ineligible.
- The next bottleneck is visible directly: block tiling has `41,943,040` shared-load requests, `67,146,486` bank conflicts, and `134,255,841` load wavefronts, an average `3.2-way` conflict. NCU also flags its global stores as using only `4` of `32` bytes per sector per thread.

## Block tiled vs bank-conflict-free

Both kernels use exactly `(BM,BN,BK,TM,TN)=(128,128,32,8,8)`, so this is the cleanest causal comparison.

| Metric | Block tiled | Bank-conflict-free |
|---|---:|---:|
| Shared-load conflicts | 67,148,458 | No shared-load warning |
| Shared-load wavefronts | 134,247,725 | No shared-load warning |
| Cycles with an eligible warp | 33.70% | 59.49% |
| Eligible warps/scheduler | 0.50 | 1.39 |
| Issued warps/scheduler/cycle | 0.34 | 0.59 |
| Warp cycles/issued instruction | 11.07 | 6.25 |
| Executed instructions | 399,599,616 | 443,660,288 |
| Registers/thread | 122 | 120 |
| Achieved occupancy | 46.91% | 46.45% |

- The original kernel's shared loads average a `3.2-way` bank conflict; `67.1M` excess conflicts account for about half of its `134.2M` shared-load wavefronts. The bank-free kernel no longer triggers a shared-load conflict warning.
- Removing those conflicts increases eligible warps from `0.50` to `1.39` per scheduler, raises issue rate from `0.34` to `0.59`, and cuts warp cycles per instruction by `43.5%`.
- Registers, shared memory, and occupancy are essentially unchanged. This rules out occupancy as the source of the gain and directly attributes it to more efficient shared-memory service.
- The XOR/interleaved addressing is not free: executed instructions rise by `11.0%`. The scheduling improvement is large enough to repay that address-generation overhead for this configuration.
- A smaller residual issue remains on shared stores: `625,766` conflicts across `4,194,304` requests, averaging about `1.1` wavefronts/request. This is far smaller than the removed shared-load conflict cost.

## Bank-conflict-free vs vectorized

Both kernels use `(BM,BN,BK,TM,TN)=(128,128,32,8,8)`.

| Metric | Bank-conflict-free | Vectorized |
|---|---:|---:|
| NCU duration | 8.71 ms | 8.51 ms |
| Executed instructions | 443,660,288 | 401,522,688 |
| Shared-store requests | 4,194,304 | 2,621,440 |
| Shared-store conflicts | 630,245 | 933,716 |
| L2 hit rate | 77.60% | 77.60% |
| Memory throughput | 22.99 GB/s | 20.32 GB/s |
| Registers/thread | 120 | 127 |
| Achieved occupancy | 46.46% | 46.58% |

- Vectorization reduces executed instructions by `9.5%` and profiler duration by `2.3%`. This supports the expected instruction-side benefit from replacing scalar global loads with `float4` loads.
- L2 hit rate is identical and achieved occupancy is unchanged. The improvement is therefore not caused by better caching or more resident warps.
- Shared-store requests fall by `37.5%`, consistent with a smaller load/staging instruction footprint, but the remaining stores are less bank-efficient: the warning changes from roughly `1.1-way` to `2.0-way`, and conflicts rise from `0.63M` to `0.93M`.
- Register use rises modestly from `120` to `127`. The reduced instruction count still wins, but extra registers and shared-store conflicts explain why the speedup is much smaller than the `4x` reduction suggested by load width alone.
- The detailed global-load instruction, sector, and DRAM-byte values were not printed, so the present profile demonstrates a lower total instruction count but does not yet quantify the exact number of eliminated global-load instructions.

## Vectorized vs warp tiled

This deliberately holds `(BM,BN,BK,TM,TN)=(128,128,32,8,8)` fixed. The profiled warp configuration is therefore the matched ablation, not the final `BK=16, TN=4` winner.

| Metric | Vectorized | Warp tiled |
|---|---:|---:|
| NCU duration | 8.51 ms | 8.75 ms |
| Executed instructions | 401,522,688 | 310,384,640 |
| Warp cycles/issued instruction | 6.77 | 4.51 |
| Issued warps/scheduler/cycle | 0.55 | 0.42 |
| Eligible warps/scheduler | 1.38 | 0.63 |
| Registers/thread | 127 | 187 |
| Achieved occupancy | 46.55% | 23.25% |
| Shared-store conflicts | 917,529 | 15,469,057 |
| Shared-store wavefronts | 5,127,808 | 19,671,497 |

- Warp tiling executes `22.7%` fewer instructions and lowers warp cycles per instruction by `33.4%`, confirming additional warp-local reuse and a more efficient instruction stream.
- That benefit comes with `187` registers/thread and half the occupancy: theoretical occupancy falls from `50%` to `25%`, while achieved active warps fall from `14.89` to `7.44` per SM.
- The transposed-A/vectorized staging layout also causes severe shared-store conflicts in this configuration: average conflict degree rises from `2.0-way` to `7.5-way`, conflicts increase about `16.9x`, and shared-store wavefronts increase about `3.8x`.
- Consequently, fewer warps are eligible and issue rate falls from `0.55` to `0.42`. At `2048^3`, the matched warp kernel is actually `2.8%` slower by NCU duration (`8.75` vs `8.51` ms).
- This profile therefore identifies both the intended reuse benefit and its costs; it does not independently establish the warp kernel's `8192^3` speedup. The final winner also changes `BK` and `TN`, and the larger benchmark has many more block waves, so the performance result is size/configuration dependent.
- NCU additionally flags the warp kernel's global stores as using `16` of `32` sector bytes per thread, leaving output-store coalescing as another possible improvement.

## Warp WNITER=2 vs WNITER=4

Every parameter except `WNITER` is identical.

| Metric | WNITER=2 | WNITER=4 |
|---|---:|---:|
| Executed instructions | 312,176,640 | 312,152,064 |
| Shared-store conflicts | 6,291,456 | 6,291,456 |
| Shared-store wavefronts | 10,485,760 | 10,485,760 |
| Issued warps/scheduler/cycle | 0.46 | 0.45 |
| Eligible warps/scheduler | 0.68 | 0.66 |
| Warp cycles/issued instruction | 4.08 | 4.17 |
| Registers/thread | 188 | 188 |
| Achieved occupancy | 23.24% | 23.26% |

- Instruction count, shared-memory conflict behavior, registers, and occupancy are effectively identical. `WNITER` changes the arrangement/order of the same per-thread work rather than its total amount or resource footprint.
- `WNITER=2` has a small scheduling advantage: about `3%` more eligible warps, a slightly higher issue rate, and `2.2%` fewer warp cycles per issued instruction.
- Only `WNITER=4` triggers a fixed-latency dependency warning (`1.3` cycles, `30.2%` of its `4.2` cycles between instructions). This is consistent with its elongated `1 x 4` subtile traversal creating a slightly less favorable dependency schedule than the balanced `2 x 2` traversal.
- These differences are directionally consistent with the `2.5%` timing advantage previously measured for `WNITER=2`, but this profile did not collect a reliable standalone duration for the pair, so the mechanism should be described as suggestive rather than conclusive.

## Warp tile 64x64 vs 64x32

The block tile and all other tuning parameters are fixed; changing `WN` changes the number of warps/threads required to cover the `128 x 128` block.

| Metric | 64x64 warp tile | 64x32 warp tile |
|---|---:|---:|
| NCU duration | 7.95 ms | 7.94 ms |
| Executed instructions | 312,176,640 | 326,162,432 |
| Block size | 128 threads | 256 threads |
| Registers/thread | 188 | 127 |
| Achieved occupancy | 23.24% | 46.56% |
| Eligible warps/scheduler | 0.68 | 1.14 |
| Issued warps/scheduler/cycle | 0.46 | 0.48 |
| Warp cycles/issued instruction | 4.07 | 7.76 |
| Shared-store conflicts | 6,291,456 | 6,291,456 |

- The `64 x 64` tile executes `4.3%` fewer instructions and needs only four warps/block. Its larger tile gives each warp more reuse, and each warp requires `47.6%` fewer cycles per issued instruction.
- The tradeoff is `188` registers/thread and only `25%` theoretical occupancy. The `64 x 32` tile uses eight warps/block, `127` registers/thread, and reaches roughly twice the active/eligible warp count.
- Shared-store requests, conflicts, and wavefronts are identical, so bank behavior does not explain the shape difference.
- At `2048^3`, the reuse/instruction advantage and occupancy advantage cancel almost exactly: profiler durations are `7.95` and `7.94` ms. At `8192^3`, the measured `64 x 64` timing is about `5%` faster, indicating that its lower instruction/reuse cost wins once the much larger grid supplies enough blocks without relying on the rectangular tile's extra warps.
- Both launches contain only `256` blocks (`3.2` waves/SM), and NCU estimates a possible `25%` tail-wave cost. This makes the `2048^3` duration unusually sensitive to launch shape and is another reason to use the `8192^3` benchmark for the final speed comparison.
