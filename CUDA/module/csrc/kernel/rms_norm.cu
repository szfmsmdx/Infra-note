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
        float sum = 0.0f;
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

    const float* x_row = input + row * dim;
    float* out_row = output + row * dim;

    __shared__ float rms; // share memory
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

void rms_norm_cuda_launch(torch::Tensor out, torch::Tensor x, torch::Tensor weight, float eps){
    int num_token = x.size(0);
    int dim = x.size(1);

    dim3 grid(num_token);   // grid 决定有多少个线程块 block
    dim3 block(dim);        // block 决定里面的线程个数

    rms_norm_kernel_v2<<<grid, block>>>(
        out.data_ptr<float>(),
        x.data_ptr<float>(),    // 构建模板，以float形式传入
        weight.data_ptr<float>(),
        dim,
        eps
    );
}