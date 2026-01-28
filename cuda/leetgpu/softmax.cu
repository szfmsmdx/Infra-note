#include <cuda_runtime.h>
#include <math.h>

constexpr int THREAD_PER_BLOCK = 1024;
constexpr int WARP_SIZE = 32;
constexpr int REDUCTION_SIZE = THREAD_PER_BLOCK / WARP_SIZE;

__global__ void softmax_kernel_v4(const float* input, float* output, int N) {
    // 复用同一个 shared memory 数组，节省空间
    __shared__ float shared_mem[REDUCTION_SIZE];

    int tid = threadIdx.x;
    int n_threads = blockDim.x;
    int warp_id = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;

    // --- Step 1: Find Global Max ---
    float thread_max = -__FLT_MAX__;
    
    // 使用 float4 进行向量化读取
    const float4* input_v4 = reinterpret_cast<const float4*>(input);
    int v4_limit = N / 4;

    for (int i = tid; i < v4_limit; i += n_threads) {
        float4 val4 = input_v4[i];
        thread_max = fmaxf(thread_max, fmaxf(fmaxf(val4.x, val4.y), fmaxf(val4.z, val4.w)));
    }
    // 处理余数 (N % 4 != 0)
    for (int i = v4_limit * 4 + tid; i < N; i += n_threads) {
        thread_max = fmaxf(thread_max, input[i]);
    }

    // Warp 规约 Max
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1)
        thread_max = fmaxf(thread_max, __shfl_down_sync(0xffffffff, thread_max, offset));
    
    if (lane == 0) shared_mem[warp_id] = thread_max;
    __syncthreads();

    if (warp_id == 0) {
        float val = (tid < REDUCTION_SIZE) ? shared_mem[tid] : -__FLT_MAX__;
        for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1)
            val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
        if (tid == 0) shared_mem[0] = val;
    }
    __syncthreads();
    float global_max = shared_mem[0];

    // --- Step 2: Sum & Cache Exp ---
    float thread_sum = 0.0f;
    float4* output_v4 = reinterpret_cast<float4*>(output);

    for (int i = tid; i < v4_limit; i += n_threads) {
        float4 val4 = input_v4[i];
        float4 res4;
        res4.x = __expf(val4.x - global_max);
        res4.y = __expf(val4.y - global_max);
        res4.z = __expf(val4.z - global_max);
        res4.w = __expf(val4.w - global_max);
        
        // 将 exp 结果存入 output，后续直接读
        output_v4[i] = res4;
        thread_sum += (res4.x + res4.y + res4.z + res4.w);
    }
    for (int i = v4_limit * 4 + tid; i < N; i += n_threads) {
        float val = __expf(input[i] - global_max);
        output[i] = val;
        thread_sum += val;
    }

    // Warp 规约 Sum
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1)
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    
    if (lane == 0) shared_mem[warp_id] = thread_sum;
    __syncthreads();

    if (warp_id == 0) {
        float val = (tid < REDUCTION_SIZE) ? shared_mem[tid] : 0.0f;
        for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
        if (tid == 0) shared_mem[0] = val;
    }
    __syncthreads();
    float global_sum = shared_mem[0];

    // --- Step 3: Final Output ---
    // 计算 global_sum 的倒数，将除法变乘法（更快）
    float inv_sum = 1.0f / global_sum;
    for (int i = tid; i < v4_limit; i += n_threads) {
        float4 res4 = output_v4[i];
        res4.x *= inv_sum;
        res4.y *= inv_sum;
        res4.z *= inv_sum;
        res4.w *= inv_sum;
        output_v4[i] = res4;
    }
    for (int i = v4_limit * 4 + tid; i < N; i += n_threads) {
        output[i] *= inv_sum;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    // 注意：float4 要求地址对齐（通常 input 指针是 128-bit 对齐的，所以没问题）
    softmax_kernel_v4<<<1, THREAD_PER_BLOCK>>>(input, output, N);
    cudaDeviceSynchronize();
}