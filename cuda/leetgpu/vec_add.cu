#include <cuda_runtime.h>

__global__ void vector_add_v1(const float* A, const float* B, float* C, int N) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N)
        C[tid] = B[tid] + A[tid];
}

__global__ void vector_add_v2(const float* A, const float* B, float* C, int N){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;    // 这是为了健壮性，防止 N 比线程开得更大
    const float4* A4 = reinterpret_cast<const float4*> (A); // reinterpret_cast 重新解释转型
    const float4* B4 = reinterpret_cast<const float4*> (B);
    float4* C4 = reinterpret_cast<float4*> (C);

    int n4 = N / 4;

    // float4 向量化处理
    for (int i = tid; i < n4; i += stride){
        float4 a = A4[i];
        float4 b = B4[i];
        float4 res;

        res.x = a.x + b.x;
        res.y = a.y + b.y;
        res.z = a.z + b.z;
        res.w = a.w + b.w;

        C4[i] = res;
    }

    // 边界处理
    int reset = N % 4;
    if(reset != 0 && tid == 0){
        for (int i = N - reset; i < N; ++i){
            C[i] = A[i] + B[i];
        }
    }
}

__global__ void vector_add(const float* A, const float* B, float* C, int N){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;    // 这是为了健壮性，防止 N 比线程开得更大
    const float4* A4 = reinterpret_cast<const float4*> (A); // reinterpret_cast 重新解释转型
    const float4* B4 = reinterpret_cast<const float4*> (B);
    float4* C4 = reinterpret_cast<float4*> (C);

    int n4 = N / 4;

    // float4 向量化处理
    for (int i = tid; i < n4; i += stride){
        float4 a = A4[i];
        float4 b = B4[i];
        float4 res;

        res.x = a.x + b.x;
        res.y = a.y + b.y;
        res.z = a.z + b.z;
        res.w = a.w + b.w;

        C4[i] = res;
    }

    // 优化边界处理
    int restart = n4 * 4;
    if (tid + restart < N){
        C[tid + restart] = A[tid + restart] + B[tid + restart];
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
