#include <torch/extension.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

#define TILE_SIZE 32
#define BM 128  // Block 处理 A 的行数
#define BN 128  // Block 处理 B 的列数
#define BK 16    // Block 每次 K 维度滑动的长度
#define TM 8    // 每个线程处理 A 的行数
#define TN 8    // 每个线程处理 B 的列数

__global__ void gemm_v1(
    const float* a,
    const float* b,
    float*c,
    int m, int n, int k
){
    // a[m, k], b[k, n], c[m, n]
    // int row = blockDim.x * blockIdx.x + threadIdx.x;
    // int col = blockDim.y * blockIdx.y + threadIdx.y;

    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if(row < m && col < n){
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += a[row * k + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
    }
}

__global__ void gemm_v2(
    const float* a,
    const float* b,
    float*c,
    int m, int n, int k
){
    // 问题：计算过程是 c[0][0]、c[0][1] .. 这里会复用 A 的行数据
    // 针对 C 矩阵的一个 [TILE_SIZE][TILE_SIZE] 大小的cache矩阵
    __shared__ float share_a[TILE_SIZE][TILE_SIZE]; // 1024 -> 一个 block大小
    __shared__ float share_b[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;

    float sum = 0.0f;

    // 沿 k 切成 TILE 小块
    for(int i = 0; i < (k + TILE_SIZE - 1) / TILE_SIZE; ++i){
        // 搬运到 share mem 中
        // 搬运A
        if(row < m && (i * TILE_SIZE + tx) < k){
            // 行坐标:row * k, 列坐标 i * TILE_SIZE + tx
            share_a[ty][tx] = a[row * k + i * TILE_SIZE + tx];
        } else {
            share_a[ty][tx] = 0.0f;
        }

        // 搬运B
        if(col < n && (i * TILE_SIZE + ty) < k){
            share_b[ty][tx] = b[(i * TILE_SIZE + ty) * n + col];
        } else {
            share_b[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll  // 告诉编译器展开循环
        for(int j = 0; j < TILE_SIZE; ++j){
            sum += share_a[ty][j] * share_b[j][tx];
        }

        __syncthreads();
    }

    if (row < m && col < n){
        c[row * n + col] = sum;
    }
}

__global__ void gemm_v3(
    const float* a,
    const float* b,
    float*c,
    int m, int n, int k
){
    // 针对 v2 还可以改进两个点：1. share mem访存延迟 2. 计算强度太低，v2 取出两个share mem但是只做了一次加法
    // 利用寄存器，让一个线程负责结果的一个 TILE 块

    // 1. 声明共享内存，block 共享
    __shared__ float s_a[BM][BK];
    __shared__ float s_b[BK][BN];

    // 2. 声明寄存器，线程私有
    float r_c[TM][TN] = {0.0f}; // 存储 8x8 的中间累加结果
    float r_a[TM];              // 缓存 s_a 数据到寄存器
    float r_b[TN];              // 缓存 s_b 数据到寄存器

    // 线程索引：16x16 = 256 线程
    int tx = threadIdx.x; 
    int ty = threadIdx.y;
    int tid = ty * (BN / TN) + tx; // 线性线程 ID (0-255)

    // 计算当前 Block 在全局矩阵中的位置
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // --- 核心循环开始 ---
    for (int step = 0; step < k; step += BK) {
        
        // 步骤 A: 协作搬运 A 到 s_a [BM x BK] = [128 x 8]
        // 总共 1024 个元素，256 个线程，每人搬 4 个
        for (int i = 0; i < 8; i++) {
            int logic_id = tid * 8 + i;
            int row = logic_id / BK; // 0-127
            int col = logic_id % BK; // 0-7
            if ((by * BM + row) < m && (step + col) < k) {
                s_a[row][col] = a[(by * BM + row) * k + (step + col)];
            } else {
                s_a[row][col] = 0.0f;
            }
        }

        // 步骤 B: 协作搬运 B 到 s_b [BK x BN] = [8 x 128]
        // 总共 1024 个元素，256 个线程，每人搬 4 个
        for (int i = 0; i < 8; i++) {
            int logic_id = tid * 8 + i;
            int row = logic_id / BN; // 0-7
            int col = logic_id % BN; // 0-127
            if ((step + row) < k && (bx * BN + col) < n) {
                s_b[row][col] = b[(step + row) * n + (bx * BN + col)];
            } else {
                s_b[row][col] = 0.0f;
            }
        }

        // 等待所有线程完成搬运
        __syncthreads();

        // 步骤 C: 寄存器级计算 (Thread Tiling)
        #pragma unroll
        for (int dot_idx = 0; dot_idx < BK; dot_idx++) {
            // 1. 先从 Shared Memory 载入数据到寄存器
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                r_a[i] = s_a[ty * TM + i][dot_idx];
            }
            #pragma unroll
            for (int j = 0; j < TN; j++) {
                r_b[j] = s_b[dot_idx][tx * TN + j];
            }

            // 2. 在寄存器中进行 8x8 外积累加
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                #pragma unroll
                for (int j = 0; j < TN; j++) {
                    r_c[i][j] += r_a[i] * r_b[j];
                }
            }
        }

        // 确保本轮计算结束，才能开始下一轮搬运
        __syncthreads();
    }

    // 3. 写回结果：从寄存器到显存
    for (int i = 0; i < TM; i++) {
        for (int j = 0; j < TN; j++) {
            int cur_row = by * BM + ty * TM + i;
            int cur_col = bx * BN + tx * TN + j;
            if (cur_row < m && cur_col < n) {
                c[cur_row * n + cur_col] = r_c[i][j];
            }
        }
    }
}

__global__ void gemm_v4(
    const float* __restrict__ a, 
    const float* __restrict__ b, 
    float* __restrict__ c,
    int m, int n, int k
) {
    // v3 的优化点在于：显存 -> share mem -> 计算 -> register mem 是串行的
    // 当计算单元在处理 Buffer 0 的数据时，内存单元同时去显存搬第 2 块数据到 Buffer 1。
    // 下一轮，计算单元处理 Buffer 1，内存单元搬第 3 块到 Buffer 0。

    // 1. 双倍共享内存空间 [2]
    __shared__ float s_a[2][BM][BK];
    __shared__ float s_b[2][BK][BN];

    float r_c[TM][TN] = {0.0f};
    float r_a[TM];
    float r_b[TN];

    int tx = threadIdx.x; 
    int ty = threadIdx.y;
    int tid = ty * (BN / TN) + tx;

    int bx = blockIdx.x;
    int by = blockIdx.y;

    // --- 预加载第一块数据 (Stage 0) ---
    {
        for (int i = 0; i < 8; i++) {
            int logic_id = tid * 8 + i;
            s_a[0][logic_id / BK][logic_id % BK] = (by * BM + logic_id / BK < m && logic_id % BK < k) ? a[(by * BM + logic_id / BK) * k + logic_id % BK] : 0.0f;
            s_b[0][logic_id / BN][logic_id % BN] = (logic_id / BN < k && bx * BN + logic_id % BN < n) ? b[(logic_id / BN) * n + (bx * BN + logic_id % BN)] : 0.0f;
        }
    }
    __syncthreads();

    // --- 主循环 ---
    int write_stage = 1; // 下一次搬运的目标
    int read_stage = 0;  // 当前计算的来源

    for (int step = BK; step < k; step += BK) {
        
        // 步骤 A: 异步启动下一块数据的搬运 (搬到 write_stage)
        for (int i = 0; i < 8; i++) {
            int logic_id = tid * 8 + i;
            int row_a = logic_id / BK;
            int col_a = logic_id % BK;
            s_a[write_stage][row_a][col_a] = (by * BM + row_a < m && step + col_a < k) ? a[(by * BM + row_a) * k + (step + col_a)] : 0.0f;

            int row_b = logic_id / BN;
            int col_b = logic_id % BN;
            s_b[write_stage][row_b][col_b] = (step + row_b < k && bx * BN + col_b < n) ? b[(step + row_b) * n + (bx * BN + col_b)] : 0.0f;
        }

        // 步骤 B: 计算当前 read_stage 的数据
        // 注意：计算前不需要同步下一块搬运，但需要确保上一轮搬运已完成（本循环开始前的同步已保证）
        #pragma unroll
        for (int dot_idx = 0; dot_idx < BK; dot_idx++) {
            for (int i = 0; i < TM; i++) r_a[i] = s_a[read_stage][ty * TM + i][dot_idx];
            for (int j = 0; j < TN; j++) r_b[j] = s_b[read_stage][dot_idx][tx * TN + j];
            for (int i = 0; i < TM; i++) {
                for (int j = 0; j < TN; j++) r_c[i][j] += r_a[i] * r_b[j];
            }
        }

        // 步骤 C: 等待搬运 write_stage 完成，并切换状态
        __syncthreads(); 
        read_stage = write_stage;
        write_stage = 1 - write_stage;
    }

    // --- 最后一块数据的计算 ---
    #pragma unroll
    for (int dot_idx = 0; dot_idx < BK; dot_idx++) {
        for (int i = 0; i < TM; i++) r_a[i] = s_a[read_stage][ty * TM + i][dot_idx];
        for (int j = 0; j < TN; j++) r_b[j] = s_b[read_stage][dot_idx][tx * TN + j];
        for (int i = 0; i < TM; i++) {
            for (int j = 0; j < TN; j++) r_c[i][j] += r_a[i] * r_b[j];
        }
    }

    // 写回结果
    for (int i = 0; i < TM; i++) {
        for (int j = 0; j < TN; j++) {
            int cur_row = by * BM + ty * TM + i;
            int cur_col = bx * BN + tx * TN + j;
            if (cur_row < m && cur_col < n) c[cur_row * n + cur_col] = r_c[i][j];
        }
    }
}

__global__ void gemm_v5(
    const float* __restrict__ a, 
    const float* __restrict__ b, 
    float* __restrict__ c,
    int m, int n, int k
) {
    // 1. 共享内存：增加 padding 彻底消除存储冲突
    __shared__ half s_a[BM][BK + 8];
    __shared__ half s_b[BN][BK + 8]; // 存储 B 的转置以便于合并访存

    // 2. Fragment 定义
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag[4];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag[2][4];

    // 初始化累加器
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++) {
            wmma::fill_fragment(acc_frag[i][j], 0.0f);
        }
    }

    int tid = threadIdx.y * blockDim.x + threadIdx.x; 
    int warp_id = tid / 32;
    int warp_row = (warp_id / 2) * 32; 
    int warp_col = (warp_id % 2) * 64; 

    // 主循环
    for (int step = 0; step < k; step += BK) {
        
        // 协作搬运 (Global -> Shared)
        // 这里的逻辑确保了访存对齐和合并
        #pragma unroll
        for (int i = 0; i < 8; i++) { 
            int logic_id = tid * 8 + i; 
            
            // 搬运 A
            int row_a = logic_id / BK;
            int col_a = logic_id % BK;
            int g_row_a = blockIdx.y * BM + row_a;
            int g_col_a = step + col_a;
            if (g_row_a < m && g_col_a < k)
                s_a[row_a][col_a] = __float2half(a[g_row_a * k + g_col_a]);
            else
                s_a[row_a][col_a] = __float2half(0.0f);

            // 搬运 B 并转置存储在 Shared Memory
            int col_b_tile = logic_id % BN;
            int row_b_tile = logic_id / BN;
            int g_row_b = step + row_b_tile;
            int g_col_b = blockIdx.x * BN + col_b_tile;
            if (g_row_b < k && g_col_b < n)
                s_b[col_b_tile][row_b_tile] = __float2half(b[g_row_b * n + g_col_b]);
            else
                s_b[col_b_tile][row_b_tile] = __float2half(0.0f);
        }

        __syncthreads(); // 确保搬运完成

        // 计算 WMMA
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            wmma::load_matrix_sync(a_frag[i], (half*)&s_a[warp_row + i * 16][0], BK + 8);
        }
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::load_matrix_sync(b_frag[j], (half*)&s_b[warp_col + j * 16][0], BK + 8);
        }

        #pragma unroll
        for (int i = 0; i < 2; i++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                wmma::mma_sync(acc_frag[i][j], a_frag[i], b_frag[j], acc_frag[i][j]);
            }
        }

        __syncthreads(); // 确保计算完成，才能加载下一块
    }

    // 3. 写回结果
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int cur_row = blockIdx.y * BM + warp_row + i * 16;
            int cur_col = blockIdx.x * BN + warp_col + j * 16;
            if (cur_row + 15 < m && cur_col + 15 < n) {
                wmma::store_matrix_sync(&c[cur_row * n + cur_col], acc_frag[i][j], n, wmma::mem_row_major);
            }
        }
    }
}

void gemm_launch_fp32(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    int m = a.size(0);
    int k = a.size(1);
    int n = b.size(1);

    // 定义二维 Block
    dim3 block(BN / TN, BM / TM);
    // 定义二维 Grid，确保覆盖 M 行 N 列
    dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);   // x 负责行，y负责列

    gemm_v3<<<grid, block>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        m, n, k
    );
}

void gemm_launch_tc_fp32(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    int m = a.size(0);
    int k = a.size(1);
    int n = b.size(1);

    // v5 使用 256 个线程 (8个 Warp)
    dim3 block(32, 8); 
    dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);

    gemm_v5<<<grid, block>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        m, n, k
    );
}