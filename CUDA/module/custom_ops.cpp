#include <torch/extension.h>

// rms norm
void rms_norm_fp32_launch(torch::Tensor out, torch::Tensor x, torch::Tensor weight, float eps);
void rms_norm_fp16_launch(torch::Tensor out, torch::Tensor x, torch::Tensor weight, float eps);
void rms_norm_bf16_launch(torch::Tensor out, torch::Tensor x, torch::Tensor weight, float eps);

// fused add rms norm
void fused_add_rms_norm_fp32_launch(torch::Tensor x, torch::Tensor attn, torch::Tensor w, torch::Tensor out, float eps);
void fused_add_rms_norm_fp16_launch(torch::Tensor x, torch::Tensor attn, torch::Tensor w, torch::Tensor out, float eps);
void fused_add_rms_norm_bf16_launch(torch::Tensor x, torch::Tensor attn, torch::Tensor w, torch::Tensor out, float eps);

// gemm
void gemm_launch_fp32(torch::Tensor a, torch::Tensor b, torch::Tensor c);


// 统一桥接函数：智能分发类型
void rms_norm_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor out, float eps) {
    auto scalar_type = x.scalar_type();
    if (scalar_type == at::ScalarType::Float) rms_norm_fp32_launch(out, x, weight, eps);
    else if (scalar_type == at::ScalarType::Half) rms_norm_fp16_launch(out, x, weight, eps);
    else if (scalar_type == at::ScalarType::BFloat16) rms_norm_bf16_launch(out, x, weight, eps);
    else TORCH_CHECK(false, "Unsupported dtype for rms_norm");
}

void fused_add_rms_norm_forward(torch::Tensor x, torch::Tensor attn, torch::Tensor weight, torch::Tensor output, float eps) {
    auto scalar_type = x.scalar_type();
    if (scalar_type == at::ScalarType::Float) fused_add_rms_norm_fp32_launch(x, attn, weight, output, eps);
    else if (scalar_type == at::ScalarType::Half) fused_add_rms_norm_fp16_launch(x, attn, weight, output, eps);
    else if (scalar_type == at::ScalarType::BFloat16) fused_add_rms_norm_bf16_launch(x, attn, weight, output, eps);
    else TORCH_CHECK(false, "Unsupported dtype for fused_add_norm");
}

void gemm_forward(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
    gemm_launch_fp32(a, b, c);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rms_norm", &rms_norm_forward, "RMSNorm Forward");
    m.def("fused_add_norm", &fused_add_rms_norm_forward, "Fused Add + RMSNorm Forward");
    m.def("gemm", &gemm_forward, "GEMM Forward");
}