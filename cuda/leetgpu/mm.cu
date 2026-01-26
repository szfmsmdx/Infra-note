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

    int tx = threadIdx.x; // 0..3
    int ty = threadIdx.y; // 0..3

    int row_start = ty * TSIZE; // 0,8,16,24
    int col_start = tx * TSIZE;

    int block_row = blockIdx.y * TILE_SIZE;
    int block_col = blockIdx.x * TILE_SIZE;

    int tid = ty * (TILE_SIZE / TSIZE) + tx; // 0..15

    for (int k_tile = 0; k_tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++k_tile) {

        // ======================
        // Load A tile (行条带)
        // ======================
        int row0 = 2 * tid;
        int row1 = 2 * tid + 1;

        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i) {
            int global_col = k_tile * TILE_SIZE + i;

            // row0
            if (block_row + row0 < M && global_col < K)
                sa[row0][i] = A[(block_row + row0) * K + global_col];
            else
                sa[row0][i] = 0.0f;

            // row1
            if (block_row + row1 < M && global_col < K)
                sa[row1][i] = A[(block_row + row1) * K + global_col];
            else
                sa[row1][i] = 0.0f;
        }

        // ======================
        // Load B tile (列条带)
        // ======================
        int col0 = 2 * tid;
        int col1 = 2 * tid + 1;

        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i) {
            int global_row = k_tile * TILE_SIZE + i;

            // col0
            if (global_row < K && block_col + col0 < N)
                sb[i][col0] = B[global_row * N + (block_col + col0)];
            else
                sb[i][col0] = 0.0f;

            // col1
            if (global_row < K && block_col + col1 < N)
                sb[i][col1] = B[global_row * N + (block_col + col1)];
            else
                sb[i][col1] = 0.0f;
        }

        __syncthreads();

        // ======================
        // Compute: 32 次 8×8 外积
        // ======================
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {

            // A: 一列取 8 行
            #pragma unroll
            for (int i = 0; i < TSIZE; ++i)
                ta[i] = sa[row_start + i][k];

            // B: 一行取 8 列
            #pragma unroll
            for (int j = 0; j < TSIZE; ++j)
                tb[j] = sb[k][col_start + j];

            // 外积
            #pragma unroll
            for (int i = 0; i < TSIZE; ++i)
                #pragma unroll
                for (int j = 0; j < TSIZE; ++j)
                    tc[i][j] += ta[i] * tb[j];
        }

        __syncthreads();
    }

    // ======================
    // Write back C (8×8)
    // ======================
    #pragma unroll
    for (int i = 0; i < TSIZE; ++i) {
        int global_row = block_row + row_start + i;
        if (global_row < M) {
            #pragma unroll
            for (int j = 0; j < TSIZE; ++j) {
                int global_col = block_col + col_start + j;
                if (global_col < N) {
                    C[global_row * N + global_col] = tc[i][j];
                }
            }
        }
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