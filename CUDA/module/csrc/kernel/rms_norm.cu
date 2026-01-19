#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_SIZE 1024

__global__ void rms_norm_kernel(
    float* output, 
    const float* input,
    const float* weight,
    int dim,
    float eps
){
    // 2D
    int row = blockIdx.x;                           // 当前 block 在grid中的索引
    const float* input_row = input + row * dim;     // 指针偏移
    float* out_row = output + row * dim;            // 输出指针偏移

    if (threadIdx.x == 0){
        float sum = 0.0f;                           // 线程级别私有变量
        for (int i = 0; i < dim; ++i){
            sum += input_row[i] * input_row[i]; //不能换成 pow，否则每个线程调用一次非常慢
        }
        float rms = rsqrt(sum / dim + eps);
        for(int i = 0; i < dim; ++i){
            out_row[i] = (input_row[i] * rms) * weight[i];
        }
    }
};

__global__ void rms_norm_kernel_v2(
    float* output, 
    const float* input,
    const float* weight,
    int dim,
    float eps
){
    int row = blockIdx.x;
    int col = threadIdx.x;

    const float* x_row = input + row * dim;         // 这也是私有变量，不过指针的开销非常小
    float* out_row = output + row * dim;

    __shared__ float rms;                           // share memory
    __shared__ float s_diff_sq[BLOCK_SIZE];

    float val = 0.0f;
    if (col < dim){
        s_diff_sq[col] = x_row[col] * x_row[col];   // 存平方
    } else s_diff_sq[col] = 0.0;

    __syncthreads();    // 线程同步

    // 规约 reduce
    for(int stride = blockDim.x / 2; stride > 0; stride /= 2){
        if(col < stride){
            s_diff_sq[col] += s_diff_sq[col + stride];
        }
        __syncthreads();    // 等待线程同步;
    }

    if(col == 0){
        float sum = s_diff_sq[0] / dim;
        rms = rsqrt(sum + eps);
    }

    __syncthreads(); // 同步，等 0 算完
    
    if(col < dim){
        out_row[col] = x_row[col] * rms * weight[col];
    }
};


#include <cooperative_groups.h> // 引入高级协作组头文件

// Warp 级别的求和规约
__device__ float warpReduceSum(float val) {
    for (int offset = 32/2; offset > 0; offset /= 2) {
        // __shfl_down_sync 让当前线程直接拿到 tid + offset 线程里的 val
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void rms_norm_kernel_v3(
    float* output, 
    const float* input,
    const float* weight,
    int dim,
    float eps
){
    /*
    优化思路:
        1. 处理非 2幂次维度：找到大于 dim 的最小2幂次
        2. 优化reduce过程中写入share_mem：使用 warp 寄存器操作
        3. 向量化访存
    */

    int row = blockIdx.x;
    int tid = threadIdx.x;

    const float* x_row = input + row * dim;
    float* out_row = output + row * dim;

    float sum = 0.0f;

    // 处理线程和dim不对应的情况
    // 线程 0 处理 0、blockDim、2*blockDim ...
    // 线程 1 处理 1+0、1+blockDim、1+2*blockDim ...
    // ...
    for (int i = tid; i < dim; i += blockDim.x){
        float v = x_row[i];
        sum += v * v;   // 存在自己线程的 sum 寄存器中
    }

    // block 内规约
    __shared__ float warp_sum[32];  // warp(32线程)局部和, block内部可见;
    int lane = tid % 32;            // warp 内部索引
    int wid = tid / 32;             // warp id

    sum = warpReduceSum(sum);

    if (lane == 0) warp_sum[wid] = sum;
    __syncthreads();

    sum = (tid < blockDim.x / 32) ? warp_sum[lane] : 0.0f;
    if (wid == 0) sum = warpReduceSum(sum);

    // 3. 广播 RMS 结果
    __shared__ float s_rms;
    if (tid == 0) {
        s_rms = rsqrtf(sum / dim + eps);
    }
    __syncthreads();

    // 4. 写回结果
    for (int i = tid; i < dim; i += blockDim.x) {
        output[row * dim + i] = (x_row[i] * s_rms) * weight[i];
    }
};

void rms_norm_cuda_launch(torch::Tensor out, torch::Tensor x, torch::Tensor weight, float eps){
    int num_token = x.size(0);
    int dim = x.size(1);

    dim3 grid(num_token);   // grid 决定有多少个线程块 block
    dim3 block(dim);        // block 决定里面的线程个数

    rms_norm_kernel_v3<<<grid, block>>>(
        out.data_ptr<float>(),
        x.data_ptr<float>(),    // 构建模板，以float形式传入
        weight.data_ptr<float>(),
        dim,
        eps
    );
}