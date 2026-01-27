#include <cuda_runtime.h>
#include <math.h>

__global__ void softmax_kernel(const float* input, float* output, int N) {
    // 假设 N 较小，或者我们只演示单 Block 逻辑
    // 如果 N 很大，通常需要分两次 Kernel：第一次算全局 Max 和 Sum，第二次算结果
    extern __shared__ float sdata[]; 

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 1. 求最大值 (Max Trick)
    float local_max = (idx < N) ? input[idx] : -1e38f;
    sdata[tid] = local_max;
    __syncthreads();

    // 块内规约求 Max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    float max_val = sdata[0]; 

    // 2. 求指数和 (Sum)
    float exp_val = (idx < N) ? expf(input[idx] - max_val) : 0.0f;
    sdata[tid] = exp_val; // 复用 shared memory 存 exp 结果
    __syncthreads();

    // 块内规约求 Sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    float sum_val = sdata[0];

    // 3. 计算并写回
    if (idx < N) {
        output[idx] = exp_val / sum_val;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    // 这里的简化逻辑：假设我们处理一个能够放入单个 Block 或分块处理的场景
    // 对于 N=500,000，通常建议使用 cuBLAS 或分三个阶段的 Kernel
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // 注意：这里的简单实现仅适用于数据量在一个 Block 覆盖范围内能得出结果的情况
    // 或者在多 Kernel 间传递全局 Max/Sum。
    softmax_kernel<<<blocks, threads, threads * sizeof(float)>>>(input, output, N);
}