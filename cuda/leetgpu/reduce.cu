#include <cuda_runtime.h>

__global__ void reduce_kernel(
    const float* input, float* output, int N
){
    __shared__ float sm[32];
    int tid = threadIdx.x;
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;

    int idx = blockDim.x * blockIdx.x + tid;
    int stride = gridDim.x * blockDim.x;

    // 先取到后面block的值
    float val = 0.0f;
    for (int i = idx; i < N; i += stride) {
        val += input[i];
    }

    // warp 内规约
    for (int offset = 16; offset; offset >>= 1){
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    if(lane == 0)
        sm[warp] = val;

    __syncthreads();

    if(warp == 0){
        val = (lane < 32) ? sm[lane] : 0.0f;
        for (int offset = 16; offset; offset >>= 1){
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        // if(lane == 0){
        //     output[blockIdx.x] = val;
        // }
        if (lane == 0)
            atomicAdd(output, val);  // ✅ 不越界
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int block_dim = 32 * 32;
    dim3 ThreadPerBlock(block_dim);
    dim3 BlcokPerGrid((N + block_dim -1) / block_dim);
    reduce_kernel<<<BlcokPerGrid, ThreadPerBlock>>>(input, output, N);
}
