#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

void launch_naive(const float *A, const float *B, float *C, int M, int N, int K);
void launchSmemTiled(const float *A, const float *B, float *C, int M, int N, int K, int BM, int BN, int BK);
void launchBlockTiled(const float *A, const float *B, float *C, int M, int N, int K, int BM, int BN, int BK, int TM, int TN);

static void check(cudaError_t error)
{
    if (error == cudaSuccess)
        return;
    std::cerr << "CUDA error: " << cudaGetErrorString(error) << '\n';
    std::exit(1);
}

static void check(cublasStatus_t status)
{
    if (status == CUBLAS_STATUS_SUCCESS)
        return;
    std::cerr << "cuBLAS error: " << static_cast<int>(status) << '\n';
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

static int usage(const char *program)
{
    std::cerr << "usage: " << program << " --M <positive> --N <positive> --K <positive>"
              << " [--kernel naive|smem|block] [--BM <positive> --BN <positive> --BK <positive>]"
              << " [--TM <positive> --TN <positive>] [--runs <positive>] [--no-verify]\n";
    return 1;
}

int main(int argc, char **argv)
{
    int M = 0, N = 0, K = 0, BM = 0, BN = 0, BK = 0, TM = 0, TN = 0, runs = 3;
    const char *kernel = "naive";
    bool verify = true;
    for (int i = 1; i < argc;)
    {
        if (!std::strcmp(argv[i], "--no-verify"))
        {
            verify = false;
            ++i;
            continue;
        }

        if (i + 1 == argc)
            return usage(argv[0]);
        if (!std::strcmp(argv[i], "--kernel"))
        {
            kernel = argv[i + 1];
            i += 2;
            continue;
        }

        int *value = !std::strcmp(argv[i], "--M")    ? &M
                     : !std::strcmp(argv[i], "--N")  ? &N
                     : !std::strcmp(argv[i], "--K")  ? &K
                     : !std::strcmp(argv[i], "--BM") ? &BM
                     : !std::strcmp(argv[i], "--BN") ? &BN
                     : !std::strcmp(argv[i], "--BK") ? &BK
                     : !std::strcmp(argv[i], "--TM") ? &TM
                     : !std::strcmp(argv[i], "--TN") ? &TN
                     : !std::strcmp(argv[i], "--runs") ? &runs
                                                     : nullptr;
        if (value == nullptr || !parse_positive(argv[i + 1], *value))
            return usage(argv[0]);
        i += 2;
    }
    const bool use_naive = !std::strcmp(kernel, "naive");
    const bool use_smem = !std::strcmp(kernel, "smem");
    const bool use_block = !std::strcmp(kernel, "block");
    if (M == 0 || N == 0 || K == 0 || (!use_naive && !use_smem && !use_block) ||
        ((use_smem || use_block) && (BM == 0 || BN == 0 || BK == 0)) || (use_block && (TM == 0 || TN == 0)))
        return usage(argv[0]);

    const size_t a_count = static_cast<size_t>(M) * K;
    const size_t b_count = static_cast<size_t>(K) * N;
    const size_t c_count = static_cast<size_t>(M) * N;
    std::vector<float> A(a_count), B(b_count), C(c_count), reference(c_count);

    // A fixed seed gives varied but reproducible inputs.
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> random(-1.0f, 1.0f);
    for (float &value : A)
    {
        value = random(rng);
    }
    for (float &value : B)
    {
        value = random(rng);
    }

    float *d_A, *d_B, *d_C, *d_reference;
    check(cudaMalloc(&d_A, a_count * sizeof(float)));
    check(cudaMalloc(&d_B, b_count * sizeof(float)));
    check(cudaMalloc(&d_C, c_count * sizeof(float)));
    check(cudaMalloc(&d_reference, c_count * sizeof(float)));
    check(cudaMemcpy(d_A, A.data(), a_count * sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemcpy(d_B, B.data(), b_count * sizeof(float), cudaMemcpyHostToDevice));

    // Warm up before timing the repeated kernel launches.
    if (use_smem)
        launchSmemTiled(d_A, d_B, d_C, M, N, K, BM, BN, BK);
    else if (use_block)
        launchBlockTiled(d_A, d_B, d_C, M, N, K, BM, BN, BK, TM, TN);
    else
        launch_naive(d_A, d_B, d_C, M, N, K);
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    check(cudaEventCreate(&start));
    check(cudaEventCreate(&stop));
    check(cudaEventRecord(start));
    for (int run = 0; run < runs; ++run)
    {
        if (use_smem)
            launchSmemTiled(d_A, d_B, d_C, M, N, K, BM, BN, BK);
        else if (use_block)
            launchBlockTiled(d_A, d_B, d_C, M, N, K, BM, BN, BK, TM, TN);
        else
            launch_naive(d_A, d_B, d_C, M, N, K);
    }
    check(cudaEventRecord(stop));
    check(cudaGetLastError());
    check(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    check(cudaEventElapsedTime(&total_ms, start, stop));

    float max_error = 0.0f;
    bool correct = true;
    if (verify)
    {
        cublasHandle_t cublas;
        check(cublasCreate(&cublas));
        check(cublasSetMathMode(cublas, CUBLAS_PEDANTIC_MATH));
        const float alpha = 1.0f;
        const float beta = 0.0f;

        // cuBLAS is column-major: C^T = B^T * A^T matches our row-major C = A * B.
        check(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_reference, N));
        check(cudaMemcpy(C.data(), d_C, c_count * sizeof(float), cudaMemcpyDeviceToHost));
        check(cudaMemcpy(reference.data(), d_reference, c_count * sizeof(float), cudaMemcpyDeviceToHost));

        // Compare against the cuBLAS FP32 result on the host.
        for (size_t index = 0; index < c_count; ++index)
        {
            const float error = std::fabs(C[index] - reference[index]);
            if (error > max_error)
                max_error = error;
            correct &= error <= 1e-3f * std::fmax(1.0f, std::fabs(reference[index]));
        }
        check(cublasDestroy(cublas));
    }

    const double time_ms = total_ms / runs;
    const double tflops = 2.0 * M * N * K / (time_ms * 1.0e9);
    std::cout << "kernel=" << kernel << " M=" << M << " N=" << N << " K=" << K
              << " BM=" << BM << " BN=" << BN << " BK=" << BK << " TM=" << TM << " TN=" << TN
              << " runs=" << runs << " time_ms=" << time_ms << " tflops=" << tflops
              << " max_abs_error=" << max_error << " status=" << (verify ? (correct ? "PASS" : "FAIL") : "SKIPPED") << '\n';

    check(cudaEventDestroy(start));
    check(cudaEventDestroy(stop));
    check(cudaFree(d_A));
    check(cudaFree(d_B));
    check(cudaFree(d_C));
    check(cudaFree(d_reference));
    return correct ? 0 : 1;
}
