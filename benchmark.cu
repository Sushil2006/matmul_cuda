#include <cuda_runtime.h>

#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

void launch_naive(const float *A, const float *B, float *C, int M, int N, int K);

static void check(cudaError_t error)
{
    if (error == cudaSuccess)
        return;
    std::cerr << "CUDA error: " << cudaGetErrorString(error) << '\n';
    std::exit(1);
}

static bool parse_positive(const char *text, int &value)
{
    char *end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (*text == '\0' || *end != '\0' || parsed <= 0 || parsed > INT_MAX)
        return false;
    value = static_cast<int>(parsed);
    return true;
}

int main(int argc, char **argv)
{
    int M = 0, N = 0, K = 0;
    for (int i = 1; i + 1 < argc; i += 2) {
        int *value = !std::strcmp(argv[i], "--M") ? &M : !std::strcmp(argv[i], "--N") ? &N : !std::strcmp(argv[i], "--K") ? &K : nullptr;
        if (value == nullptr || !parse_positive(argv[i + 1], *value)) {
            std::cerr << "usage: " << argv[0] << " --M <positive> --N <positive> --K <positive>\n";
            return 1;
        }
    }
    if (argc != 7 || M == 0 || N == 0 || K == 0) {
        std::cerr << "usage: " << argv[0] << " --M <positive> --N <positive> --K <positive>\n";
        return 1;
    }

    const size_t a_count = static_cast<size_t>(M) * K;
    const size_t b_count = static_cast<size_t>(K) * N;
    const size_t c_count = static_cast<size_t>(M) * N;
    std::vector<float> A(a_count), B(b_count), C(c_count);

    // Deterministic inputs make failures reproducible.
    for (size_t i = 0; i < a_count; ++i)
        A[i] = static_cast<float>(i % 13) / 13.0f - 0.5f;
    for (size_t i = 0; i < b_count; ++i)
        B[i] = static_cast<float>(i % 17) / 17.0f - 0.5f;

    float *d_A, *d_B, *d_C;
    check(cudaMalloc(&d_A, a_count * sizeof(float)));
    check(cudaMalloc(&d_B, b_count * sizeof(float)));
    check(cudaMalloc(&d_C, c_count * sizeof(float)));
    check(cudaMemcpy(d_A, A.data(), a_count * sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemcpy(d_B, B.data(), b_count * sizeof(float), cudaMemcpyHostToDevice));

    // Warm up before timing the repeated kernel launches.
    launch_naive(d_A, d_B, d_C, M, N, K);
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());

    constexpr int runs = 100;
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start));
    check(cudaEventCreate(&stop));
    check(cudaEventRecord(start));
    for (int run = 0; run < runs; ++run)
        launch_naive(d_A, d_B, d_C, M, N, K);
    check(cudaEventRecord(stop));
    check(cudaGetLastError());
    check(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    check(cudaEventElapsedTime(&total_ms, start, stop));
    check(cudaMemcpy(C.data(), d_C, c_count * sizeof(float), cudaMemcpyDeviceToHost));

    // Check the final output against a CPU FP32 reference.
    float max_error = 0.0f;
    bool correct = true;
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float expected = 0.0f;
            for (int k = 0; k < K; ++k)
                expected += A[static_cast<size_t>(row) * K + k] * B[static_cast<size_t>(k) * N + col];
            const float error = std::fabs(C[static_cast<size_t>(row) * N + col] - expected);
            if (error > max_error)
                max_error = error;
            correct &= error <= 1e-3f * std::fmax(1.0f, std::fabs(expected));
        }
    }

    const double time_ms = total_ms / runs;
    const double tflops = 2.0 * M * N * K / (time_ms * 1.0e9);
    std::cout << "M=" << M << " N=" << N << " K=" << K << " time_ms=" << time_ms << " tflops=" << tflops
              << " max_abs_error=" << max_error << " status=" << (correct ? "PASS" : "FAIL") << '\n';

    check(cudaEventDestroy(start));
    check(cudaEventDestroy(stop));
    check(cudaFree(d_A));
    check(cudaFree(d_B));
    check(cudaFree(d_C));
    return correct ? 0 : 1;
}
