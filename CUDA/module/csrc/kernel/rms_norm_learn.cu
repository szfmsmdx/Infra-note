#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cooperative_groups.h> // 引入高级协作组头文件
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define BLOCK_SIZE 1024

// Warp 级别的求和规约
__device__ float warpReduceSum(float val) {
    for (int offset = 32/2; offset > 0; offset /= 2) {
        // __shfl_down_sync 让当前线程直接拿到 tid + offset 线程里的 val
        // mask = 0xffffffff 表示整个 warp（32 线程）都参与同步。
        val += __shfl_down_sync(0xffffffff, val, offset);   // 自动忽略越界线程
    }
    return val;
}

__global__ void rms_norm_kernel(
    float* output, 
    const float* input,
    const float* weight,
    int dim,
    float eps
){
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
    // ... 归到 blockDim 大小
    for (int i = tid; i < dim; i += blockDim.x){
        float v = x_row[i];
        sum += v * v;       // 加到自己的 sum 寄存器中
    }

    // block 内规约
    __shared__ float warp_sum[32];      // warp 是 block 内共享，block 内所有线程都能看到，存32个float
    int lane = tid % 32;                // warp 内索引
    int wid = tid / 32;                 // warp idx

    sum = warpReduceSum(sum);           // 每32个线程进行规约，也就是规约到 warp 个数

    if (lane == 0) warp_sum[wid] = sum; // 由 0 号线程干活, 这里 wid 之所以不会越界是因为 block_size最多1024
    __syncthreads();

    // warp 规约，这里需要连续的32，所以取前 32 个
    // blockDim < 1024 / 32 < 32, 所以对应的第一个warp wid，这时候他们的 lane 对应的就是 wid
    // 就是把 warp 每个 wid 写入到前32个线程的 sum中
    sum = (tid < blockDim.x / 32) ? warp_sum[lane] : 0.0f; // 归到前 32 线程
    if (wid == 0)
        sum = warpReduceSum(sum);       //  前32线程规约

    __shared__ float s_rms;             
    if(tid == 0)                        // share mem方便后面存取，由 0 号进行计算
        s_rms = rsqrt(sum / dim + eps);

    __syncthreads();

    for (int i = tid; i < dim; i += blockDim.x){
        out_row[i] = (x_row[i] * s_rms) * weight[i];
    }
};

__global__ void rms_norm_kernel_v4(
    float* output, 
    const float* input,
    const float* weight,
    int dim,
    float eps
){
    int block_idx = blockIdx.x;
    int tid = threadIdx.x;

    // input 和 output 本质是 float 指针，我们强转成 float4 指针
    // float4 是一次性处理 4 个 float 的内置类型
    const float4* x_vec = (const float4*)(input + block_idx * dim);
    const float4* w_vec = (const float4*)weight;
    float4* out_vec = (float4*)(output + block_idx * dim);

    int vec_dim = dim / 4; // 向量化的维度是原来的 1/4
    float sum = 0.0f;

    // 1. 向量化读取与平方累加
    for (int i = tid; i < vec_dim; i += blockDim.x) {
        float4 data = x_vec[i];
        // dim 能被 4 整除得到 vec_dim，且内存地址必须是 16 字节对齐（Tensor 一般是 16字节的）
        sum += data.x * data.x;
        sum += data.y * data.y;
        sum += data.z * data.z;
        sum += data.w * data.w;
    }

    // 2. Block 级规约
    __shared__ float s_warp_sums[32];
    int lane = tid % 32;
    int wid = tid / 32;

    sum = warpReduceSum(sum);
    if (lane == 0) s_warp_sums[wid] = sum;
    __syncthreads();

    sum = (tid < blockDim.x / 32) ? s_warp_sums[lane] : 0.0f;
    if (wid == 0) sum = warpReduceSum(sum);

    __shared__ float s_rms;
    if (tid == 0) s_rms = rsqrtf(sum / dim + eps);
    __syncthreads();

    // 3. 向量化写回
    for (int i = tid; i < vec_dim; i += blockDim.x) {
        float4 data = x_vec[i];
        float4 w = w_vec[i];
        float4 res;
        res.x = data.x * s_rms * w.x;
        res.y = data.y * s_rms * w.y;
        res.z = data.z * s_rms * w.z;
        res.w = data.w * s_rms * w.w;
        out_vec[i] = res; // 一次写回 16 字节
    }
}

__global__ void rms_norm_kernel_v5(
    half* output,       
    const half* input,  
    const half* weight, 
    int dim,
    float eps
){
    int block_idx = blockIdx.x;
    int tid = threadIdx.x;

    // fp16 格式版本 2byte
    const uint4* x_vec = (const uint4*)(input + block_idx * dim);       // 每次读取 16 字节
    const uint4* w_vec = (const uint4*)(weight);
    uint4* out_vec = (uint4*)(output + block_idx * dim);

    int vec_dim = dim / 8;
    float sum = 0.0f;

    for (int i = tid; i < vec_dim; i += blockDim.x){
        uint4 raw_data = x_vec[i];
        half2* h2_ptr = reinterpret_cast<half2*>(&raw_data);    //  类型转换, 16字节转成8*2字节的fp16格式

        #pragma unroll
        for (int j = 0; j < 4; ++j){
            float2 f2 = __half22float2(h2_ptr[j]);
            sum += f2.x * f2.x;
            sum += f2.y * f2.y;
        }
    }

    __shared__ float s_warp_sums[32];
    int lane = tid % 32;
    int wid = tid / 32;

    sum = warpReduceSum(sum);
    if (lane == 0) s_warp_sums[wid] = sum;
    __syncthreads();

    sum = (tid < blockDim.x / 32) ? s_warp_sums[lane] : 0.0f;
    if (wid == 0) sum = warpReduceSum(sum);

    __shared__ float s_rms;
    if (tid == 0) s_rms = rsqrtf(sum / dim + eps);
    __syncthreads();

    for (int i = tid; i < vec_dim; i += blockDim.x){
        uint4 raw_x = x_vec[i];
        uint4 raw_w = w_vec[i];

        // 强制类型转换
        half2* x_h2 = reinterpret_cast<half2*>(&raw_x);
        half2* w_h2 = reinterpret_cast<half2*>(&raw_w);
        uint4 res_vec;
        half2* res_h2 = reinterpret_cast<half2*>(&res_vec);

        #pragma unroll
        for (int j = 0; j < 4; ++j){
            float2 f2_x = __half22float2(x_h2[j]);
            float2 f2_w = __half22float2(w_h2[j]);
            float2 out_f2;
            out_f2.x = f2_x.x * s_rms * f2_w.x;
            out_f2.y = f2_x.y * s_rms * f2_w.y;
            res_h2[j] = __float22half2_rn(out_f2);   // f2 -> h2
        }
        out_vec[i] = res_vec;
    }
}

void rms_norm_cuda_launch(torch::Tensor out, torch::Tensor x, torch::Tensor weight, float eps){
    int num_token = x.size(0);
    int dim = x.size(1);

    // 检查是否是 16 字节对齐
    bool is_aigned = ((uintptr_t)x.data_ptr<float>()) % 16 == 0;
    assert(is_aigned && "Tensor must be 16-byte aligned for float4");

    dim3 grid(num_token);   // grid 决定有多少个线程块 block
    dim3 block(dim);        // block 决定里面的线程个数

    rms_norm_kernel_v4<<<grid, block>>>(
        out.data_ptr<float>(),
        x.data_ptr<float>(),    // 构建模板，以float形式传入
        weight.data_ptr<float>(),
        dim,
        eps
    );
}

void rms_norm_cuda_launch_half(torch::Tensor out, torch::Tensor x, torch::Tensor weight, float eps) {
    int num_token = x.size(0);
    int dim = x.size(1);

    // 检查 16 字节对齐 (uint4 要求)
    // 注意这里 data_ptr 的类型改成了 at::Half
    bool is_aligned = ((uintptr_t)x.data_ptr<at::Half>()) % 16 == 0;
    assert(is_aligned && "Tensor must be 16-byte aligned for uint4/half8");

    dim3 grid(num_token);
    // 因为一个线程处理 8 个数，所以开启 dim/8 个线程即可
    // 如果 dim/8 超过 1024，建议固定设为 256/512，依靠 Kernel 里的循环处理
    int threads = (dim / 8 <= 1024) ? (dim / 8) : 1024;
    dim3 block(threads);

    rms_norm_kernel_v5<<<grid, block>>>(
        (half*)out.data_ptr<at::Half>(),
        (const half*)x.data_ptr<at::Half>(),
        (const half*)weight.data_ptr<at::Half>(),
        dim,
        eps
    );
}

__global__ void fused_add_rms_norm_fp32(
    float* x,                 // [B*L, D] - 残差，会被原地更新
    const float* attn_output, // [B*L, D] - Attention 输出
    const float* weight,      // [D]
    float* output,            // [B*L, D] - 归一化后的输出
    int dim,
    float eps
){
    int block_idx = blockIdx.x;
    int tid = threadIdx.x;

    float4* x_vec = (float4*)(x + block_idx * dim);
    const float4* y_vec = (const float4*)(attn_output + block_idx * dim);
    const float4* w_vec = (const float4*)weight;
    float4* out_vec = (float4*)(output + block_idx * dim);

    int vec_dim = dim / 4;
    float sum = 0.0f;

    for(int i = tid; i < vec_dim; i += blockDim.x){
        float4 x_data = x_vec[i];
        float4 y_data = y_vec[i];
        
        x_data.x += y_data.x; x_data.y += y_data.y;
        x_data.z += y_data.z; x_data.w += y_data.w;

        x_vec[i] = x_data; // 原地更新残差
        sum += (x_data.x * x_data.x + x_data.y * x_data.y + x_data.z * x_data.z + x_data.w * x_data.w);
    }

    __shared__ float s_warp_sums[32];
    int lane = tid % 32;
    int wid = tid / 32;

    sum = warpReduceSum(sum);
    if(lane == 0) s_warp_sums[wid] = sum;
    __syncthreads();

    // 修复：使用 blockDim.x 
    sum = (tid < blockDim.x / 32) ? s_warp_sums[lane] : 0.0f;
    if (wid == 0) sum = warpReduceSum(sum);

    __shared__ float s_rms;
    if(tid == 0) s_rms = rsqrtf(sum / dim + eps);
    __syncthreads();

    for(int i = tid; i < vec_dim; i += blockDim.x){
        float4 x_data = x_vec[i];
        float4 w_data = w_vec[i];
        float4 o;
        o.x = x_data.x * s_rms * w_data.x;
        o.y = x_data.y * s_rms * w_data.y;
        o.z = x_data.z * s_rms * w_data.z;
        o.w = x_data.w * s_rms * w_data.w;
        out_vec[i] = o;
    }
}

// --- Fused Add + RMSNorm BF16 ---
__global__ void fused_add_rms_norm_bf16(
    __nv_bfloat16* residual, 
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    __nv_bfloat16* output,
    int dim,
    float eps
){
    int block_idx = blockIdx.x;
    int tid = threadIdx.x;

    uint4* res_vec = (uint4*)(residual + block_idx * dim);
    const uint4* x_vec = (const uint4*)(input + block_idx * dim);
    const uint4* w_vec = (const uint4*)(weight);
    uint4* out_vec = (uint4*)(output + block_idx * dim);

    int vec_dim = dim / 8;
    float sum_sq = 0.0f;

    for (int i = tid; i < vec_dim; i += blockDim.x){
        uint4 raw_res = res_vec[i];
        uint4 raw_x = x_vec[i];
        
        __nv_bfloat162* bf2_res = reinterpret_cast<__nv_bfloat162*>(&raw_res);
        __nv_bfloat162* bf2_x = reinterpret_cast<__nv_bfloat162*>(&raw_x);

        #pragma unroll
        for (int j = 0; j < 4; ++j){
            bf2_res[j] = __hadd2(bf2_res[j], bf2_x[j]);
            // 修复：__bfloat162float2 -> __bfloat1622float2
            float2 f2 = __bfloat1622float2(bf2_res[j]);
            sum_sq += f2.x * f2.x + f2.y * f2.y;
        }
        res_vec[i] = raw_res;
    }

    __shared__ float s_warp_sums[32];
    int lane = tid % 32; int wid = tid / 32;
    sum_sq = warpReduceSum(sum_sq);
    if (lane == 0) s_warp_sums[wid] = sum_sq;
    __syncthreads();
    sum_sq = (tid < blockDim.x / 32) ? s_warp_sums[lane] : 0.0f;
    if (wid == 0) sum_sq = warpReduceSum(sum_sq);

    __shared__ float s_rms;
    if (tid == 0) s_rms = rsqrtf(sum_sq / dim + eps);
    __syncthreads();

    for (int i = tid; i < vec_dim; i += blockDim.x){
        uint4 raw_res = res_vec[i]; 
        uint4 raw_w = w_vec[i];
        __nv_bfloat162* bf2_res = reinterpret_cast<__nv_bfloat162*>(&raw_res);
        __nv_bfloat162* bf2_w = reinterpret_cast<__nv_bfloat162*>(&raw_w);
        
        uint4 out_raw;
        __nv_bfloat162* bf2_out = reinterpret_cast<__nv_bfloat162*>(&out_raw);

        #pragma unroll
        for (int j = 0; j < 4; ++j){
            float2 f2_r = __bfloat1622float2(bf2_res[j]);
            float2 f2_w = __bfloat1622float2(bf2_w[j]);
            float2 f2_o;
            f2_o.x = f2_r.x * s_rms * f2_w.x;
            f2_o.y = f2_r.y * s_rms * f2_w.y;
            bf2_out[j] = __float22bfloat162_rn(f2_o);
        }
        out_vec[i] = out_raw;
    }
}

// Launch 函数
void fused_add_rms_norm_fp32_launch(torch::Tensor x, torch::Tensor attn_output, torch::Tensor weight, torch::Tensor output, float eps){
    int num_token = x.size(0);
    int dim = x.size(1);
    dim3 grid(num_token);
    int threads = (dim / 4 <= 1024) ? (dim / 4) : 1024;
    fused_add_rms_norm_fp32<<<grid, threads>>>(
        x.data_ptr<float>(), attn_output.data_ptr<float>(), 
        weight.data_ptr<float>(), output.data_ptr<float>(), dim, eps
    );
}

void fused_add_rms_norm_bf16_launch(torch::Tensor x, torch::Tensor attn_output, torch::Tensor weight, torch::Tensor output, float eps){
    int num_token = x.size(0);
    int dim = x.size(1);
    dim3 grid(num_token);
    int threads = (dim / 8 <= 1024) ? (dim / 8) : 1024;
    fused_add_rms_norm_bf16<<<grid, threads>>>(
        (__nv_bfloat16*)x.data_ptr<at::BFloat16>(), (__nv_bfloat16*)attn_output.data_ptr<at::BFloat16>(),
        (__nv_bfloat16*)weight.data_ptr<at::BFloat16>(), (__nv_bfloat16*)output.data_ptr<at::BFloat16>(), dim, eps
    );
}