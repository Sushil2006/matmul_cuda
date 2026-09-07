# Programming AI Accelerators Assignment 1 - CUDA GEMM

This project develops a row-major FP32 matrix-multiplication kernel from a naive CUDA implementation to a shared-memory, register-tiled, bank-aware, vectorized, warp-tiled version.

## Layout

- `benchmark.cu` - command-line benchmark driver; allocates inputs, dispatches a kernel, times it with CUDA events, and optionally verifies against cuBLAS.
- `configs.hpp` - compile-time tile configuration types and the supported-configuration dispatcher.
- `kernels/naive.cu` - one thread computes one output element directly from global memory.
- `kernels/smem_tiled.cu` - cooperatively stages A and B tiles in shared memory.
- `kernels/block_tiled.cu` - adds per-thread register tiles to reuse shared-memory values.
- `kernels/smem_bank_conflict_free.cu` - changes shared-memory indexing to remove block-tiled load bank conflicts.
- `kernels/smem_vectorized.cu` - adds `float4` global-memory loads to the bank-conflict-free kernel.
- `kernels/smem_warp_tiled.cu` - assigns output tiles to warps and controls their subtile traversal with `WNITER`.
- `notebook.ipynb` - compiles the project, runs configuration sweeps and benchmarks, produces plots, and captures Nsight Compute profiles.
- `profiling_notes.md` - concise interpretation of the profiling comparisons and their high-signal metrics.

## Build

From this directory:

```bash
nvcc -std=c++20 -O3 benchmark.cu kernels/naive.cu kernels/smem_tiled.cu kernels/block_tiled.cu kernels/smem_bank_conflict_free.cu kernels/smem_vectorized.cu kernels/smem_warp_tiled.cu -lcublas -o matmul
```

For example, run the selected warp-tiled configuration:

```bash
./matmul --kernel warp --M 8192 --N 8192 --K 8192 --BM 128 --BN 128 --BK 16 --WM 64 --WN 64 --WNITER 2 --TM 8 --TN 4
```
