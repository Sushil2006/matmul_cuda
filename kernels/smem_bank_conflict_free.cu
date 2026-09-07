#include <cuda_runtime.h>

#include "../configs.hpp"

#include <cstdio>
#include <cstdlib>
#include <tuple>

// This is the block_tiled.cu kernel with only the bank-conflict-related edits marked as CHANGE 1-6 below.
template <typename Config>
__global__ void smem_bank_conflict_free_gemm(const float *A, const float *B, float *C, int M, int N, int K)
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
    // CHANGE 1: The A-tile bank fix uses BK - 1 as a bit mask, so BK must be a power of two.
    static_assert(BK > 0 && (BK & (BK - 1)) == 0);

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    // CHANGE 2: Spread each thread's TM x TN results across the block tile. For a fixed tn, consecutive threadIdx.x values now read consecutive
    // B words instead of words TN apart, so they use distinct banks; repeated x values in another thread row read the same word by broadcast.
    const int row = blockIdx.y * BM + threadIdx.y;
    const int col = blockIdx.x * BN + threadIdx.x;
    const int thread = threadIdx.y * THREADS_X + threadIdx.x;
    float accum[TM][TN] = {};

    for (int k0 = 0; k0 < K; k0 += BK)
    {
        // Cooperatively load the block's A and B tiles.
        for (int index = thread; index < BM * BK; index += THREADS)
        {
            const int tile_row = index / BK;
            const int tile_k = index % BK;
            // CHANGE 3: XOR permutes A's K positions within each power-of-two row. Using the row bits breaks the old BK-word bank stride while
            // preserving conflict-free cooperative stores; the compute loop applies the same permutation when it reads the value back.
            As[tile_row][tile_k ^ (tile_row & (BK - 1))] = A[(blockIdx.y * BM + tile_row) * K + k0 + tile_k];
        }

        for (int index = thread; index < BK * BN; index += THREADS)
        {
            const int tile_k = index / BN;
            const int tile_col = index % BN;
            Bs[tile_k][tile_col] = B[(k0 + tile_k) * N + blockIdx.x * BN + tile_col];
        }

        __syncthreads();

        // Reuse one A and B value across this thread's interleaved output window.
        for (int k = 0; k < BK; ++k)
        {
            float a[TM], b[TN];

            for (int tm = 0; tm < TM; ++tm)
            {
                // CHANGE 4: Consecutive threadIdx.y groups select consecutive logical rows. The XOR makes their distinct A addresses land in
                // distinct banks for every supported configuration, while threads with the same y still use the hardware broadcast path.
                const int tile_row = threadIdx.y + tm * THREADS_Y;
                a[tm] = As[tile_row][k ^ (tile_row & (BK - 1))];
            }

            for (int tn = 0; tn < TN; ++tn)
            {
                // CHANGE 5: Interleaved columns turn the old TN-word stride into a one-word stride across threadIdx.x, removing B-tile conflicts.
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

    for (int tm = 0; tm < TM; ++tm)
    {
        for (int tn = 0; tn < TN; ++tn)
        {
            // CHANGE 6: Match the output store to the interleaved row/column ownership introduced in CHANGE 2.
            C[(row + tm * THREADS_Y) * N + col + tn * THREADS_X] = accum[tm][tn];
        }
    }
}

template <typename Config>
static void launch(const float *A, const float *B, float *C, int M, int N, int K)
{
    const dim3 block(Config::BN / Config::TN, Config::BM / Config::TM);
    const dim3 grid(N / Config::BN, M / Config::BM);
    smem_bank_conflict_free_gemm<Config><<<grid, block>>>(A, B, C, M, N, K);
}

// Keep the block-tiled configurations unchanged so timings differ only because of the bank-conflict fix.
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
using BankConflictFreeConfigs = std::tuple<C0, C1, C2, C3, C4, C5, C6, C7, C8, C9, C10>;

void launchBankConflictFree(const float *A, const float *B, float *C, int M, int N, int K, int BM, int BN, int BK, int TM, int TN)
{
    if (BM <= 0 || BN <= 0 || BK <= 0 || TM <= 0 || TN <= 0 || BM % TM != 0 || BN % TN != 0 ||
        M % BM != 0 || N % BN != 0 || K % BK != 0)
    {
        std::fprintf(stderr, "bank-conflict-free GEMM dimensions must be divisible by positive tile sizes\n");
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

    if (!dispatchConfig(BankConflictFreeConfigs{}, tryLaunch))
    {
        std::fprintf(stderr, "unsupported bank-conflict-free GEMM config\n");
        std::abort();
    }
}
