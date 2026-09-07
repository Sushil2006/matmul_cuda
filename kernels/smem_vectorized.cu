#include <cuda_runtime.h>

#include "../configs.hpp"

#include <cstdio>
#include <cstdlib>
#include <tuple>

template <typename Config>
__global__ void smem_vectorized_gemm(const float *A, const float *B, float *C, int M, int N, int K)
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
    static_assert(BK > 0 && (BK & (BK - 1)) == 0);
    static_assert(BK % 4 == 0 && BN % 4 == 0);

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    const int row = blockIdx.y * BM + threadIdx.y;
    const int col = blockIdx.x * BN + threadIdx.x;
    const int thread = threadIdx.y * THREADS_X + threadIdx.x;
    float accum[TM][TN] = {};

    for (int k0 = 0; k0 < K; k0 += BK)
    {
        // Load four adjacent A values at once, then scatter them into the conflict-free XOR layout.
        for (int vector_index = thread; vector_index < BM * BK / 4; vector_index += THREADS)
        {
            const int tile_row = vector_index / (BK / 4);
            const int tile_k = vector_index % (BK / 4) * 4;
            const int mask = tile_row & (BK - 1);
            const float4 values = *reinterpret_cast<const float4 *>(&A[(blockIdx.y * BM + tile_row) * K + k0 + tile_k]);
            As[tile_row][(tile_k + 0) ^ mask] = values.x;
            As[tile_row][(tile_k + 1) ^ mask] = values.y;
            As[tile_row][(tile_k + 2) ^ mask] = values.z;
            As[tile_row][(tile_k + 3) ^ mask] = values.w;
        }

        // Load four adjacent B values at once and keep its shared-memory layout contiguous.
        for (int vector_index = thread; vector_index < BK * BN / 4; vector_index += THREADS)
        {
            const int tile_k = vector_index / (BN / 4);
            const int tile_col = vector_index % (BN / 4) * 4;
            const float4 values = *reinterpret_cast<const float4 *>(&B[(k0 + tile_k) * N + blockIdx.x * BN + tile_col]);
            Bs[tile_k][tile_col + 0] = values.x;
            Bs[tile_k][tile_col + 1] = values.y;
            Bs[tile_k][tile_col + 2] = values.z;
            Bs[tile_k][tile_col + 3] = values.w;
        }

        __syncthreads();

        // Reuse one A and B value across this thread's interleaved output window.
        for (int k = 0; k < BK; ++k)
        {
            float a[TM], b[TN];

            for (int tm = 0; tm < TM; ++tm)
            {
                const int tile_row = threadIdx.y + tm * THREADS_Y;
                a[tm] = As[tile_row][k ^ (tile_row & (BK - 1))];
            }

            for (int tn = 0; tn < TN; ++tn)
            {
                b[tn] = Bs[k][threadIdx.x + tn * THREADS_X];
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

    // Interleaved scalar stores are already coalesced across each warp.
    for (int tm = 0; tm < TM; ++tm)
    {
        for (int tn = 0; tn < TN; ++tn)
        {
            C[(row + tm * THREADS_Y) * N + col + tn * THREADS_X] = accum[tm][tn];
        }
    }
}

template <typename Config>
static void launch(const float *A, const float *B, float *C, int M, int N, int K)
{
    const dim3 block(Config::BN / Config::TN, Config::BM / Config::TM);
    const dim3 grid(N / Config::BN, M / Config::BM);
    smem_vectorized_gemm<Config><<<grid, block>>>(A, B, C, M, N, K);
}

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
using VectorizedConfigs = std::tuple<C0, C1, C2, C3, C4, C5, C6, C7, C8, C9, C10>;

void launchVectorized(const float *A, const float *B, float *C, int M, int N, int K, int BM, int BN, int BK, int TM, int TN)
{
    if (BM <= 0 || BN <= 0 || BK <= 0 || TM <= 0 || TN <= 0 || BM % TM != 0 || BN % TN != 0 || BK % 4 != 0 || BN % 4 != 0 ||
        M % BM != 0 || N % BN != 0 || K % BK != 0)
    {
        std::fprintf(stderr, "vectorized GEMM dimensions must be divisible by positive tile sizes, with BK and BN divisible by 4\n");
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

    if (!dispatchConfig(VectorizedConfigs{}, tryLaunch))
    {
        std::fprintf(stderr, "unsupported vectorized GEMM config\n");
        std::abort();
    }
}
