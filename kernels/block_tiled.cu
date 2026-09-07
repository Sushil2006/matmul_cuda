#include <cuda_runtime.h>

#include "../configs.hpp"

#include <cstdio>
#include <cstdlib>
#include <tuple>

template <typename Config>
__global__ void block_tiled_gemm(const float *A, const float *B, float *C, int M, int N, int K)
{
    constexpr int BM = Config::BM;
    constexpr int BN = Config::BN;
    constexpr int BK = Config::BK;
    constexpr int TM = Config::TM;
    constexpr int TN = Config::TN;

    constexpr int THREADS_Y = BM / TM;
    constexpr int THREADS_X = BN / TN;
    constexpr int THREADS = THREADS_Y * THREADS_X;

    static_assert(BM % TM == 0 && BN % TN == 0);
    static_assert(THREADS <= 1024);

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    const int row = blockIdx.y * BM + threadIdx.y * TM;
    const int col = blockIdx.x * BN + threadIdx.x * TN;
    const int thread = threadIdx.y * THREADS_X + threadIdx.x;
    float accum[TM][TN] = {};

    for (int k0 = 0; k0 < K; k0 += BK)
    {
        // Cooperatively load the block's A and B tiles.
        for (int index = thread; index < BM * BK; index += THREADS)
        {
            const int tile_row = index / BK;
            const int tile_k = index % BK;
            As[tile_row][tile_k] = A[(blockIdx.y * BM + tile_row) * K + k0 + tile_k];
        }

        for (int index = thread; index < BK * BN; index += THREADS)
        {
            const int tile_k = index / BN;
            const int tile_col = index % BN;
            Bs[tile_k][tile_col] = B[(k0 + tile_k) * N + blockIdx.x * BN + tile_col];
        }

        __syncthreads();

        // Reuse one A and B value across this thread's output window.
        for (int k = 0; k < BK; ++k)
        {
            float a[TM], b[TN];

            for (int tm = 0; tm < TM; ++tm)
            {
                a[tm] = As[threadIdx.y * TM + tm][k];
            }

            for (int tn = 0; tn < TN; ++tn)
            {
                b[tn] = Bs[k][threadIdx.x * TN + tn];
            }

            for (int tm = 0; tm < TM; ++tm)
            {
                for (int tn = 0; tn < TN; ++tn)
                {
                    accum[tm][tn] += a[tm] * b[tn];
                }
            }
        }

        __syncthreads();
    }

    for (int tm = 0; tm < TM; ++tm)
    {
        for (int tn = 0; tn < TN; ++tn)
        {
            C[(row + tm) * N + col + tn] = accum[tm][tn];
        }
    }
}

template <typename Config>
static void launch(const float *A, const float *B, float *C, int M, int N, int K)
{
    const dim3 block(Config::BN / Config::TN, Config::BM / Config::TM);
    const dim3 grid(N / Config::BN, M / Config::BM);
    block_tiled_gemm<Config><<<grid, block>>>(A, B, C, M, N, K);
}

// Vary BK, thread-tile shape, and output-tile shape without a full sweep.
using C0 = BlockTiledConfig<64, 64, 8, 4, 4>;
using C1 = BlockTiledConfig<64, 64, 16, 4, 4>;
using C2 = BlockTiledConfig<64, 64, 32, 4, 4>;
using C3 = BlockTiledConfig<64, 64, 8, 8, 4>;
using C4 = BlockTiledConfig<64, 64, 8, 4, 8>;
using C5 = BlockTiledConfig<128, 64, 8, 8, 4>;
using C6 = BlockTiledConfig<64, 128, 8, 4, 8>;
using C7 = BlockTiledConfig<64, 64, 8, 1, 16>;
using C8 = BlockTiledConfig<64, 64, 8, 16, 1>;
using C9 = BlockTiledConfig<64, 64, 64, 4, 4>;
using C10 = BlockTiledConfig<128, 128, 32, 8, 8>;
using BlockTiledConfigs = std::tuple<C0, C1, C2, C3, C4, C5, C6, C7, C8, C9, C10>;

void launchBlockTiled(const float *A, const float *B, float *C, int M, int N, int K, int BM, int BN, int BK, int TM, int TN)
{
    if (BM <= 0 || BN <= 0 || BK <= 0 || TM <= 0 || TN <= 0 || BM % TM != 0 || BN % TN != 0 ||
        M % BM != 0 || N % BN != 0 || K % BK != 0)
    {
        std::fprintf(stderr, "block-tiled GEMM dimensions must be divisible by positive tile sizes\n");
        std::abort();
    }

    auto tryLaunch = [&](auto config)
    {
        using Config = decltype(config);
        if (BM != Config::BM || BN != Config::BN || BK != Config::BK || TM != Config::TM || TN != Config::TN)
            return false;

        launch<Config>(A, B, C, M, N, K);
        return true;
    };

    if (!dispatchConfig(BlockTiledConfigs{}, tryLaunch))
    {
        std::fprintf(stderr, "unsupported block-tiled GEMM config\n");
        std::abort();
    }
}
