#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

extern "C" void solve(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K
);

extern "C" void solve_v3(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int k
);

// CPU reference: A[M,K] * B[K,N] = C[M,N]
void cpu_gemm(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K
) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main() {
    int M = 1024;
    int N = 1024;
    int K = 1024;

    std::cout << "Testing GEMM: "
              << "M=" << M
              << ", N=" << N
              << ", K=" << K
              << std::endl;

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C_cpu(M * N);
    std::vector<float> h_C_gpu(M * N);

    for (int i = 0; i < M * K; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < K * N; ++i) {
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice);

    /* ---------------- GPU timing ---------------- */
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    solve_v3(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_ms = 0.0f;
    cudaEventElapsedTime(&gpu_ms, start, stop);

    cudaMemcpy(h_C_gpu.data(), d_C, size_C, cudaMemcpyDeviceToHost);

    std::cout << "GPU Kernel Time: "
              << gpu_ms << " ms" << std::endl;

    /* ---------------- CPU timing ---------------- */
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_gemm(h_A.data(), h_B.data(), h_C_cpu.data(), M, N, K);
    auto cpu_end = std::chrono::high_resolution_clock::now();

    double cpu_ms = std::chrono::duration<double, std::milli>(
                        cpu_end - cpu_start
                    ).count();

    std::cout << "CPU GEMM Time: "
              << cpu_ms << " ms" << std::endl;

    /* ---------------- Verification ---------------- */
    std::cout << "Verifying..." << std::endl;

    int errors = 0;
    for (int i = 0; i < M * N; ++i) {
        if (std::fabs(h_C_cpu[i] - h_C_gpu[i]) > 1e-3f) {
            if (errors < 5) {
                std::cout << "Mismatch at " << i
                          << ": CPU " << h_C_cpu[i]
                          << ", GPU " << h_C_gpu[i]
                          << std::endl;
            }
            errors++;
        }
    }

    if (errors == 0) {
        std::cout << "PASSED!" << std::endl;
    } else {
        std::cout << "FAILED with "
                  << errors << " errors." << std::endl;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
