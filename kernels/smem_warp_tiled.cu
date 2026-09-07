#include <cuda_runtime.h>

#include "../configs.hpp"

#include <cstdio>
#include <cstdlib>
#include <tuple>

template <typename Config>
__global__ void smem_warp_tiled_gemm(const float *A, const float *B, float *C, int M, int N, int K)
{
    constexpr int BM = Config::BM;
    constexpr int BN = Config::BN;
    constexpr int BK = Config::BK;
    constexpr int WM = Config::WM;
    constexpr int WN = Config::WN;
    constexpr int WNITER = Config::WNITER;
    constexpr int TM = Config::TM;
    constexpr int TN = Config::TN;
    constexpr int WARPS = (BM / WM) * (BN / WN);
    constexpr int THREADS = WARPS * 32;
    constexpr int WMITER = WM * WN / (32 * TM * TN * WNITER);
    constexpr int WSUBM = WM / WMITER;
    constexpr int WSUBN = WN / WNITER;

    static_assert(BM % WM == 0 && BN % WN == 0);
    static_assert(THREADS <= 1024);
    static_assert(BK % 4 == 0 && BN % 4 == 0 && TN % 4 == 0);
    static_assert(WM * WN % (32 * TM * TN * WNITER) == 0);
    static_assert(WM % WMITER == 0 && WN % WNITER == 0);
    static_assert(WSUBM % TM == 0 && WSUBN % TN == 0);
    static_assert((WSUBM / TM) * (WSUBN / TN) == 32);

    // Transposing A in shared memory makes warp-local A reads vary along the contiguous dimension.
    __shared__ float As[BK][BM];
    __shared__ float Bs[BK][BN];

    const int thread = threadIdx.x;
    const int warp = thread / 32;
    const int lane = thread % 32;
    const int warp_row = warp / (BN / WN);
    const int warp_col = warp % (BN / WN);
    const int lane_cols = WSUBN / TN;
    const int thread_row = lane / lane_cols;
    const int thread_col = lane % lane_cols;
    float accum[WMITER * TM][WNITER * TN] = {};

    for (int k0 = 0; k0 < K; k0 += BK)
    {
        // Load the block tiles with 128-bit global-memory transactions.
        for (int vector_index = thread; vector_index < BM * BK / 4; vector_index += THREADS)
        {
            const int tile_row = vector_index / (BK / 4);
            const int tile_k = vector_index % (BK / 4) * 4;
            const float4 values = *reinterpret_cast<const float4 *>(&A[(blockIdx.y * BM + tile_row) * K + k0 + tile_k]);
            As[tile_k + 0][tile_row] = values.x;
            As[tile_k + 1][tile_row] = values.y;
            As[tile_k + 2][tile_row] = values.z;
            As[tile_k + 3][tile_row] = values.w;
        }

        for (int vector_index = thread; vector_index < BK * BN / 4; vector_index += THREADS)
        {
            const int tile_k = vector_index / (BN / 4);
            const int tile_col = vector_index % (BN / 4) * 4;
            *reinterpret_cast<float4 *>(&Bs[tile_k][tile_col]) =
                *reinterpret_cast<const float4 *>(&B[(k0 + tile_k) * N + blockIdx.x * BN + tile_col]);
        }

        __syncthreads();

        // Each warp owns one WM x WN tile, split into WMITER x WNITER subtiles.
        for (int k = 0; k < BK; ++k)
        {
            float a[WMITER * TM];
            float b[WNITER * TN];

            for (int wm = 0; wm < WMITER; ++wm)
            {
                for (int tm = 0; tm < TM; ++tm)
                {
                    a[wm * TM + tm] = As[k][warp_row * WM + wm * WSUBM + thread_row * TM + tm];
                }
            }

            for (int wn = 0; wn < WNITER; ++wn)
            {
                for (int tn = 0; tn < TN; ++tn)
                {
                    b[wn * TN + tn] = Bs[k][warp_col * WN + wn * WSUBN + thread_col * TN + tn];
                }
            }

            for (int row = 0; row < WMITER * TM; ++row)
            {
                for (int col = 0; col < WNITER * TN; ++col)
                {
                    accum[row][col] += a[row] * b[col];
                }
            }
        }

        __syncthreads();
    }

    // Store adjacent results in 128-bit transactions.
    for (int wm = 0; wm < WMITER; ++wm)
    {
        for (int wn = 0; wn < WNITER; ++wn)
        {
            for (int tm = 0; tm < TM; ++tm)
            {
                const int row = blockIdx.y * BM + warp_row * WM + wm * WSUBM + thread_row * TM + tm;
                const int col = blockIdx.x * BN + warp_col * WN + wn * WSUBN + thread_col * TN;
                for (int tn = 0; tn < TN; tn += 4)
                {
                    const float4 values = {accum[wm * TM + tm][wn * TN + tn + 0], accum[wm * TM + tm][wn * TN + tn + 1],
                                           accum[wm * TM + tm][wn * TN + tn + 2], accum[wm * TM + tm][wn * TN + tn + 3]};
                    *reinterpret_cast<float4 *>(&C[row * N + col + tn]) = values;
                }
            }
        }
    }
}

template <typename Config>
static void launch(const float *A, const float *B, float *C, int M, int N, int K)
{
    constexpr int WARPS = (Config::BM / Config::WM) * (Config::BN / Config::WN);
    const dim3 block(WARPS * 32);
    const dim3 grid(N / Config::BN, M / Config::BM);
    smem_warp_tiled_gemm<Config><<<grid, block>>>(A, B, C, M, N, K);
}

// Sweep K depth, warp shape, and warp-subtile shape around practical 128-thread and 256-thread layouts.
using C0 = WarpTiledConfig<128, 128, 8, 64, 64, 4, 8, 4>;
using C1 = WarpTiledConfig<128, 128, 16, 64, 64, 4, 8, 4>;
using C2 = WarpTiledConfig<128, 128, 32, 64, 64, 4, 8, 4>;
using C3 = WarpTiledConfig<128, 128, 16, 64, 64, 2, 8, 4>;
using C4 = WarpTiledConfig<128, 128, 16, 32, 64, 2, 8, 4>;
using C5 = WarpTiledConfig<128, 128, 16, 64, 32, 2, 8, 4>;
using C6 = WarpTiledConfig<64, 128, 16, 32, 64, 2, 8, 4>;
using C7 = WarpTiledConfig<128, 64, 16, 64, 32, 2, 8, 4>;
using C8 = WarpTiledConfig<64, 64, 16, 32, 32, 2, 4, 4>;
using C9 = WarpTiledConfig<128, 128, 32, 64, 64, 2, 8, 8>;
using C10 = WarpTiledConfig<64, 64, 16, 64, 64, 2, 8, 4>;
using C11 = WarpTiledConfig<128, 64, 16, 64, 64, 2, 8, 4>;
using C12 = WarpTiledConfig<256, 64, 16, 64, 64, 2, 8, 4>;
using C13 = WarpTiledConfig<256, 128, 16, 64, 64, 2, 8, 4>;
using WarpTiledConfigs = std::tuple<C0, C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12, C13>;

void launchWarpTiled(const float *A, const float *B, float *C, int M, int N, int K, int BM, int BN, int BK, int WM, int WN, int WNITER, int TM, int TN)
{
    if (BM <= 0 || BN <= 0 || BK <= 0 || WM <= 0 || WN <= 0 || WNITER <= 0 || TM <= 0 || TN <= 0 || M % BM != 0 || N % BN != 0 ||
        K % BK != 0 || BM % WM != 0 || BN % WN != 0 || BK % 4 != 0 || BN % 4 != 0 || TN % 4 != 0)
    {
        std::fprintf(stderr, "warp-tiled GEMM dimensions must fit positive block, warp, and thread tiles; BK and TN must be divisible by 4\n");
        std::abort();
    }

    auto tryLaunch = [&](auto config)
    {
        using Config = decltype(config);
        if (BM != Config::BM || BN != Config::BN || BK != Config::BK || WM != Config::WM || WN != Config::WN || WNITER != Config::WNITER ||
            TM != Config::TM || TN != Config::TN)
            return false;

        launch<Config>(A, B, C, M, N, K);
        return true;
    };

    if (!dispatchConfig(WarpTiledConfigs{}, tryLaunch))
    {
        std::fprintf(stderr, "unsupported warp-tiled GEMM config\n");
        std::abort();
    }
}
