#include <cuda_runtime.h>

__global__ void naive_gemm(const float *A, const float *B, float *C, int M, int N, int K)
{
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N)
        return;

    float sum = 0.0f;
    for (int k = 0; k < K; ++k)
        sum += A[row * K + k] * B[k * N + col];

    C[row * N + col] = sum;
}

void launch_naive(const float *A, const float *B, float *C, int M, int N, int K)
{
    const dim3 block(16, 16);
    const dim3 grid((N - 1) / block.x + 1, (M - 1) / block.y + 1);
    naive_gemm<<<grid, block>>>(A, B, C, M, N, K);
}
