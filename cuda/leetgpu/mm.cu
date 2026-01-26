#include <cuda_runtime.h>

#define TILE_SIZE 32
#define TSIZE 8
#define THREADS_PER_BLOCK (TILE_SIZE/TSIZE)

// A: [M, K]
// B: [K, N]
// C: [M, N]
__global__ void matrix_multiplication_kernel_v2(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K
) {
    __shared__ float a[TILE_SIZE][TILE_SIZE];
    __shared__ float b[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;

    // 遍历 K 维度的 tile
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {

        // A tile: [row, t * TILE_SIZE + tx]
        if (row < M && (t * TILE_SIZE + tx) < K) {
            a[ty][tx] = A[row * K + (t * TILE_SIZE + tx)];
        } else {
            a[ty][tx] = 0.0f;
        }

        // B tile: [t * TILE_SIZE + ty, col]
        if (col < N && (t * TILE_SIZE + ty) < K) {
            b[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        } else {
            b[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll  // 无分支跳转，unroll 1等价于 unroll，数字是总的迭代次数，是 for循环次数 / 数字
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += a[ty][k] * b[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

__global__ void matrix_multiplication_kernel_v3(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K
) {
    __shared__ float sa[TILE_SIZE][TILE_SIZE];
    __shared__ float sb[TILE_SIZE][TILE_SIZE];

    float ta[TSIZE];
    float tb[TSIZE];
    float tc[TSIZE][TSIZE] = {0.0f};

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row_start = ty * TSIZE; // 当前线程负责的C矩阵起点(相对block)
    int col_start = tx * TSIZE;
    int block_row = blockIdx.y * TILE_SIZE; // 当前block负责的C矩阵起点
    int block_col = blockIdx.x * TILE_SIZE;

    for (int k_tile = 0; k_tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++k_tile){
        // 为了效率，每个线程取一列条带
        int linear_id = ty * (TILE_SIZE / TSIZE) + tx;
    }
}

extern "C" void solve(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K
) {
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE
    );

    matrix_multiplication_kernel_v2<<<blocksPerGrid, threadsPerBlock>>>(
        A, B, C, M, N, K
    );
}


extern "C" void solve_v3(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K
) {
    dim3 threadsPerBlock(TILE_SIZE / TSIZE, TILE_SIZE / TSIZE); 
    
    dim3 blocksPerGrid(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE
    );

    matrix_multiplication_kernel_v3<<<blocksPerGrid, threadsPerBlock>>>(
        A, B, C, M, N, K
    );
}