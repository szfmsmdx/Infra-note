#include <cuda_runtime.h>
#define TILE 16

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    __shared__ float sm[TILE][TILE+1];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int irow = blockIdx.y * TILE + threadIdx.y;
    int icol = blockIdx.x * TILE + threadIdx.x;

    if(irow < rows && icol < cols){
        sm[ty][tx] = input[irow * cols + icol];
    }

    __syncthreads();

    int orow = blockIdx.x * TILE + ty;
    int ocol = blockIdx.y * TILE + tx;

    if(orow < cols && ocol < rows){
        output[orow * rows + ocol] = sm[tx][ty];
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}
