#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void gemm_v1(
    const float* a,
    const float* b,
    float*c,
    int m, int n, int k
){
    // a[m, k], b[k, n], c[m, n]
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;

    if(row < m && col < n){
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += a[row * k + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
    }
}

__global__ void gemm_v2()

void gemm_launch_fp32(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    int m = a.size(0);
    int k = a.size(1);
    int n = b.size(1);

    // 定义二维 Block
    dim3 block(32, 32); 
    // 定义二维 Grid，确保覆盖 M 行 N 列
    dim3 grid((m + block.x - 1) / block.x, (n + block.y - 1) / block.y);    // x 负责行，y负责列

    gemm_v1<<<grid, block>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        m, n, k
    );
}