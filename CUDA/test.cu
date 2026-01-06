#include <cuda_runtime.h>
#include <iostream>

__global__ void hello() {
    int tid = threadIdx.x;
    printf("Hello from GPU thread %d!\n", tid);
    // 在下一行设断点
    tid += 100;  // 断点这里，看能否看到 tid 的值
}

int main() {
    hello<<<1, 5>>>();
    cudaDeviceSynchronize();
    return 0;
}