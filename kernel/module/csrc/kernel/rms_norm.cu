#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <assert.h>

#define BLOCK_SIZE 1024

// --- 通用 Warp 规约 ---
__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// --- Block 规约获取 RMS ---
__device__ __forceinline__ float blockReduceRMS(float sum, int dim, float eps, float* s_warp_sums) {
    int tid = threadIdx.x;
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
    return s_rms;
}

// --- RMS Norm Kernels ---

__global__ void rms_norm_kernel_fp32(float* output, const float* input, const float* weight, int dim, float eps) {
    int block_idx = blockIdx.x;
    int tid = threadIdx.x;
    const float4* x_vec = (const float4*)(input + block_idx * dim);
    const float4* w_vec = (const float4*)weight;
    float4* out_vec = (float4*)(output + block_idx * dim);

    int vec_dim = dim / 4;
    float sum = 0.0f;
    for (int i = tid; i < vec_dim; i += blockDim.x) {
        float4 data = x_vec[i];
        sum += data.x * data.x + data.y * data.y + data.z * data.z + data.w * data.w;
    }

    __shared__ float s_warp_sums[32];
    float s_rms = blockReduceRMS(sum, dim, eps, s_warp_sums);

    for (int i = tid; i < vec_dim; i += blockDim.x) {
        float4 data = x_vec[i]; float4 w = w_vec[i];
        out_vec[i] = make_float4(data.x * s_rms * w.x, data.y * s_rms * w.y, data.z * s_rms * w.z, data.w * s_rms * w.w);
    }
}

// FP16/BF16 的模板化实现，因为逻辑一致
template<typename T, typename T2, typename V4>
__global__ void rms_norm_kernel_half(T* output, const T* input, const T* weight, int dim, float eps) {
    int block_idx = blockIdx.x;
    int tid = threadIdx.x;
    const V4* x_vec = (const V4*)(input + block_idx * dim);
    const V4* w_vec = (const V4*)weight;
    V4* out_vec = (V4*)(output + block_idx * dim);

    int vec_dim = dim / 8;
    float sum = 0.0f;
    for (int i = tid; i < vec_dim; i += blockDim.x) {
        V4 raw = x_vec[i];
        T2* h2_ptr = reinterpret_cast<T2*>(&raw);
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float2 f2;
            if constexpr (std::is_same_v<T, at::Half>) f2 = __half22float2(*(half2*)&h2_ptr[j]);
            else f2 = __bfloat1622float2(*(__nv_bfloat162*)&h2_ptr[j]);
            sum += f2.x * f2.x + f2.y * f2.y;
        }
    }

    __shared__ float s_warp_sums[32];
    float s_rms = blockReduceRMS(sum, dim, eps, s_warp_sums);

    for (int i = tid; i < vec_dim; i += blockDim.x) {
        V4 raw_x = x_vec[i]; V4 raw_w = w_vec[i];
        T2* x_h2 = reinterpret_cast<T2*>(&raw_x); T2* w_h2 = reinterpret_cast<T2*>(&raw_w);
        V4 res_vec; T2* res_h2 = reinterpret_cast<T2*>(&res_vec);
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float2 f2_x, f2_w;
            if constexpr (std::is_same_v<T, at::Half>) {
                f2_x = __half22float2(*(half2*)&x_h2[j]);
                f2_w = __half22float2(*(half2*)&w_h2[j]);
                float2 out_f2 = make_float2(f2_x.x * s_rms * f2_w.x, f2_x.y * s_rms * f2_w.y);
                *(half2*)&res_h2[j] = __float22half2_rn(out_f2);
            } else {
                f2_x = __bfloat1622float2(*(__nv_bfloat162*)&x_h2[j]);
                f2_w = __bfloat1622float2(*(__nv_bfloat162*)&w_h2[j]);
                float2 out_f2 = make_float2(f2_x.x * s_rms * f2_w.x, f2_x.y * s_rms * f2_w.y);
                *(__nv_bfloat162*)&res_h2[j] = __float22bfloat162_rn(out_f2);
            }
        }
        out_vec[i] = res_vec;
    }
}

// --- Fused Add Kernels (省略重复模板，直接展示 FP16 版实现逻辑) ---

__global__ void fused_add_rms_norm_kernel_fp32(
    float* residual, const float* input, const float* weight, float* output, 
    int dim, float eps
) {
    int block_idx = blockIdx.x;
    int tid = threadIdx.x;
    int vec_dim = dim / 4;

    float4* res_vec = (float4*)(residual + block_idx * dim);
    const float4* in_vec = (const float4*)(input + block_idx * dim);
    const float4* w_vec = (const float4*)weight;
    float4* out_vec = (float4*)(output + block_idx * dim);

    float sum = 0.0f;
    for (int i = tid; i < vec_dim; i += blockDim.x) {
        float4 r = res_vec[i];
        float4 in = in_vec[i];
        r.x += in.x; r.y += in.y; r.z += in.z; r.w += in.w;
        res_vec[i] = r; // 写回残差
        sum += r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
    }

    __shared__ float s_warp_sums[32];
    float s_rms = blockReduceRMS(sum, dim, eps, s_warp_sums);

    for (int i = tid; i < vec_dim; i += blockDim.x) {
        float4 r = res_vec[i]; float4 w = w_vec[i];
        out_vec[i] = make_float4(r.x * s_rms * w.x, r.y * s_rms * w.y, r.z * s_rms * w.z, r.w * s_rms * w.w);
    }
}

// 2. FP16/BF16 模板化实现 (Vectorized 8 elements)
template<typename T, typename T2, bool is_fp16>
__global__ void fused_add_rms_norm_kernel_half(
    T* residual, const T* input, const T* weight, T* output, 
    int dim, float eps
) {
    int block_idx = blockIdx.x;
    int tid = threadIdx.x;
    int vec_dim = dim / 8;

    uint4* res_vec = (uint4*)(residual + block_idx * dim);
    const uint4* in_vec = (const uint4*)(input + block_idx * dim);
    const uint4* w_vec = (const uint4*)weight;
    uint4* out_vec = (uint4*)(output + block_idx * dim);

    float sum = 0.0f;
    for (int i = tid; i < vec_dim; i += blockDim.x) {
        uint4 raw_res = res_vec[i];
        uint4 raw_in = in_vec[i];
        T2* h2_res = reinterpret_cast<T2*>(&raw_res);
        T2* h2_in = reinterpret_cast<T2*>(&raw_in);

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float2 f2;
            if constexpr (is_fp16) {
                *(half2*)&h2_res[j] = __hadd2(*(half2*)&h2_res[j], *(half2*)&h2_in[j]);
                f2 = __half22float2(*(half2*)&h2_res[j]);
            } else {
                *(__nv_bfloat162*)&h2_res[j] = __hadd2(*(__nv_bfloat162*)&h2_res[j], *(__nv_bfloat162*)&h2_in[j]);
                f2 = __bfloat1622float2(*(__nv_bfloat162*)&h2_res[j]);
            }
            sum += f2.x * f2.x + f2.y * f2.y;
        }
        res_vec[i] = raw_res;
    }

    __shared__ float s_warp_sums[32];
    float s_rms = blockReduceRMS(sum, dim, eps, s_warp_sums);

    for (int i = tid; i < vec_dim; i += blockDim.x) {
        uint4 raw_res = res_vec[i]; uint4 raw_w = w_vec[i];
        T2* h2_res = reinterpret_cast<T2*>(&raw_res);
        T2* h2_w = reinterpret_cast<T2*>(&raw_w);
        uint4 raw_out; T2* h2_out = reinterpret_cast<T2*>(&raw_out);

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float2 f2_r, f2_w;
            if constexpr (is_fp16) {
                f2_r = __half22float2(*(half2*)&h2_res[j]);
                f2_w = __half22float2(*(half2*)&h2_w[j]);
                float2 f2_out = make_float2(f2_r.x * s_rms * f2_w.x, f2_r.y * s_rms * f2_w.y);
                *(half2*)&h2_out[j] = __float22half2_rn(f2_out);
            } else {
                f2_r = __bfloat1622float2(*(__nv_bfloat162*)&h2_res[j]);
                f2_w = __bfloat1622float2(*(__nv_bfloat162*)&h2_w[j]);
                float2 f2_out = make_float2(f2_r.x * s_rms * f2_w.x, f2_r.y * s_rms * f2_w.y);
                *(__nv_bfloat162*)&h2_out[j] = __float22bfloat162_rn(f2_out);
            }
        }
        out_vec[i] = raw_out;
    }
}

// --- Launchers ---

void rms_norm_fp32_launch(torch::Tensor out, torch::Tensor x, torch::Tensor weight, float eps) {
    int dim = x.size(1);
    rms_norm_kernel_fp32<<<x.size(0), std::min(dim/4, BLOCK_SIZE)>>>(out.data_ptr<float>(), x.data_ptr<float>(), weight.data_ptr<float>(), dim, eps);
}

void rms_norm_fp16_launch(torch::Tensor out, torch::Tensor x, torch::Tensor weight, float eps) {
    int dim = x.size(1);
    rms_norm_kernel_half<half, half2, uint4><<<x.size(0), std::min(dim/8, BLOCK_SIZE)>>>((half*)out.data_ptr<at::Half>(), (const half*)x.data_ptr<at::Half>(), (const half*)weight.data_ptr<at::Half>(), dim, eps);
}

void rms_norm_bf16_launch(torch::Tensor out, torch::Tensor x, torch::Tensor weight, float eps) {
    int dim = x.size(1);
    rms_norm_kernel_half<__nv_bfloat16, __nv_bfloat162, uint4><<<x.size(0), std::min(dim/8, BLOCK_SIZE)>>>((__nv_bfloat16*)out.data_ptr<at::BFloat16>(), (const __nv_bfloat16*)x.data_ptr<at::BFloat16>(), (const __nv_bfloat16*)weight.data_ptr<at::BFloat16>(), dim, eps);
}

void fused_add_rms_norm_fp32_launch(torch::Tensor residual, torch::Tensor input, torch::Tensor weight, torch::Tensor output, float eps) {
    int dim = residual.size(1);
    int threads = std::min(dim / 4, BLOCK_SIZE);
    fused_add_rms_norm_kernel_fp32<<<residual.size(0), threads>>>(
        residual.data_ptr<float>(), input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), dim, eps);
}

void fused_add_rms_norm_fp16_launch(torch::Tensor residual, torch::Tensor input, torch::Tensor weight, torch::Tensor output, float eps) {
    int dim = residual.size(1);
    int threads = std::min(dim / 8, BLOCK_SIZE);
    fused_add_rms_norm_kernel_half<half, half2, true><<<residual.size(0), threads>>>(
        (half*)residual.data_ptr<at::Half>(), (const half*)input.data_ptr<at::Half>(), (const half*)weight.data_ptr<at::Half>(), (half*)output.data_ptr<at::Half>(), dim, eps);
}

void fused_add_rms_norm_bf16_launch(torch::Tensor residual, torch::Tensor input, torch::Tensor weight, torch::Tensor output, float eps) {
    int dim = residual.size(1);
    int threads = std::min(dim / 8, BLOCK_SIZE);
    fused_add_rms_norm_kernel_half<__nv_bfloat16, __nv_bfloat162, false><<<residual.size(0), threads>>>(
        (__nv_bfloat16*)residual.data_ptr<at::BFloat16>(), (const __nv_bfloat16*)input.data_ptr<at::BFloat16>(), (__nv_bfloat16*)weight.data_ptr<at::BFloat16>(), (__nv_bfloat16*)output.data_ptr<at::BFloat16>(), dim, eps);
}