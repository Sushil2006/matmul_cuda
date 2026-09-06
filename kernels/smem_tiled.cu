#include <cuda_runtime.h>

#include "../configs.hpp"

#include <cstdio>
#include <cstdlib>
#include <tuple>
#include <utility>

template <typename Config>
__global__ void smem_tiled_gemm(const float *A, const float *B, float *C, int M, int N, int K)
{
    constexpr int BM = Config::BM;
    constexpr int BN = Config::BN;
    constexpr int BK = Config::BK;
    static_assert(BM * BN <= 1024);

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    const int row = blockIdx.y * BM + threadIdx.y;
    const int col = blockIdx.x * BN + threadIdx.x;
    const int thread = threadIdx.y * BN + threadIdx.x;
    float sum = 0.0f;

    for (int k0 = 0; k0 < K; k0 += BK)
    {
        // Linearized loads cover tiles whose shapes differ from the thread block.
        for (int index = thread; index < BM * BK; index += BM * BN)
        {
            const int tile_row = index / BK;
            const int tile_col = index % BK;
            As[tile_row][tile_col] = A[(blockIdx.y * BM + tile_row) * K + k0 + tile_col];
        }

        for (int index = thread; index < BN * BK; index += BM * BN)
        {
            const int tile_row = index / BN;
            const int tile_col = index % BN;
            Bs[tile_row][tile_col] = B[(k0 + tile_row) * N + blockIdx.x * BN + tile_col];
        }

        __syncthreads();

        for (int k = 0; k < BK; ++k)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    C[row * N + col] = sum;
}

template <typename Config>
static void launch(const float *A, const float *B, float *C, int M, int N, int K)
{
    const dim3 block(Config::BN, Config::BM);
    const dim3 grid(N / Config::BN, M / Config::BM);
    smem_tiled_gemm<Config><<<grid, block>>>(A, B, C, M, N, K);
}

// for fixed BM, produces all combinations of BM,BN,BK
template <int BM, int... BNs, int... BKs>
consteval auto configsForBM(std::integer_sequence<int, BNs...>, std::integer_sequence<int, BKs...>)
{
    return std::tuple_cat(
        []<int BN>(std::integral_constant<int, BN>) {
            return std::tuple<SmemConfig<BM, BN, BKs>...>{};
        }(std::integral_constant<int, BNs>{})...);
}

template <int... BMs, int... BNs, int... BKs>
consteval auto makeSmemConfigs(std::integer_sequence<int, BMs...>, std::integer_sequence<int, BNs...> bns,
                               std::integer_sequence<int, BKs...> bks)
{
    return std::tuple_cat(configsForBM<BMs>(bns, bks)...);
}

using SmemConfigs = decltype(makeSmemConfigs(
    std::integer_sequence<int, 8, 16, 32>{},
    std::integer_sequence<int, 8, 16, 32>{},
    std::integer_sequence<int, 4, 8, 16, 32>{}));

static_assert(std::tuple_size_v<SmemConfigs> == 36);

void launchSmemTiled(const float *A, const float *B, float *C, int M, int N, int K, int BM, int BN, int BK)
{
    if (BM <= 0 || BN <= 0 || BK <= 0 || M % BM != 0 || N % BN != 0 || K % BK != 0)
    {
        std::fprintf(stderr, "tiled GEMM dimensions must be divisible by positive BM, BN, and BK\n");
        std::abort();
    }

    auto tryLaunch = [&](auto config)
    {
        using Config = decltype(config);
        if (BM != Config::BM || BN != Config::BN || BK != Config::BK)
            return false;

        launch<Config>(A, B, C, M, N, K);
        return true;
    };
    if (!dispatchConfig(SmemConfigs{}, tryLaunch))
    {
        std::fprintf(stderr, "unsupported tiled GEMM config\n");
        std::abort();
    }
}
