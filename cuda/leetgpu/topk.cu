#include <cuda_runtime.h>

__global__ void topk_kernel(
    const float* input, float* output, int N, int K
){
    
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N, int k) {
    dim3 thread_per_block(1);
}
