#include <cuda_runtime.h>

#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

void launch_naive(const float* A, const float* B, float* C, int M, int N,
                  int K);

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t error = (call);                                             \
        if (error != cudaSuccess) {                                             \
            std::cerr << "CUDA error: " << cudaGetErrorString(error) << '\n'; \
            return 1;                                                           \
        }                                                                       \
    } while (0)

static bool parse_positive_int(const char* text, int* value) {
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (*text == '\0' || *end != '\0' || parsed <= 0 || parsed > INT_MAX)
        return false;
    *value = static_cast<int>(parsed);
    return true;
}

int main(int argc, char** argv) {
    int M = 0, N = 0, K = 0;
    for (int i = 1; i < argc; i += 2) {
        int* value = !std::strcmp(argv[i], "--M") ? &M :
                     !std::strcmp(argv[i], "--N") ? &N :
                     !std::strcmp(argv[i], "--K") ? &K : nullptr;
        if (i + 1 == argc || value == nullptr ||
            !parse_positive_int(argv[i + 1], value)) {
            std::cerr << "usage: " << argv[0] << " --M <positive> --N <positive> --K <positive>\n";
            return 1;
        }
    }
    if (M == 0 || N == 0 || K == 0) {
        std::cerr << "usage: " << argv[0] << " --M <positive> --N <positive> --K <positive>\n";
        return 1;
    }

    const size_t a_count = static_cast<size_t>(M) * K;
    const size_t b_count = static_cast<size_t>(K) * N;
    const size_t c_count = static_cast<size_t>(M) * N;
    std::vector<float> A(a_count), B(b_count), C(c_count);
    for (size_t i = 0; i < a_count; ++i)
        A[i] = static_cast<float>(static_cast<int>(i % 13) - 6) / 13.0f;
    for (size_t i = 0; i < b_count; ++i)
        B[i] = static_cast<float>(static_cast<int>(i % 17) - 8) / 17.0f;

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, a_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, b_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, c_count * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), a_count * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), b_count * sizeof(float), cudaMemcpyHostToDevice));

    launch_naive(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    constexpr int runs = 100;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int run = 0; run < runs; ++run)
        launch_naive(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    CUDA_CHECK(cudaMemcpy(C.data(), d_C, c_count * sizeof(float), cudaMemcpyDeviceToHost));

    float max_abs_error = 0.0f;
    bool correct = true;
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
                sum += A[static_cast<size_t>(row) * K + k] * B[static_cast<size_t>(k) * N + col];
            const size_t index = static_cast<size_t>(row) * N + col;
            const float error = std::fabs(C[index] - sum);
            max_abs_error = std::fmax(max_abs_error, error);
            correct &= error <= 1e-3f * std::fmax(1.0f, std::fabs(sum));
        }
    }

    const double average_ms = total_ms / runs;
    const double tflops = 2.0 * M * N * K / (average_ms * 1.0e9);
    std::cout << "M=" << M << " N=" << N << " K=" << K
              << " time_ms=" << average_ms << " tflops=" << tflops
              << " max_abs_error=" << max_abs_error
              << " status=" << (correct ? "PASS" : "FAIL") << '\n';

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return correct ? 0 : 1;
}
