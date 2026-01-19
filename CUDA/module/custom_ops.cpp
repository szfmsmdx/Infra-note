#include <torch/extension.h>

void rms_norm_cuda_launch(torch::Tensor out, torch::Tensor x, torch::Tensor weight, float epsilon);
void rms_norm_cuda_launch_half(torch::Tensor out, torch::Tensor x, torch::Tensor weight, float epsilon);
void fused_add_rms_norm_fp32_launch(torch::Tensor x, torch::Tensor attn_output, torch::Tensor weight, torch::Tensor output, float eps);
void fused_add_rms_norm_bf16_launch(torch::Tensor x, torch::Tensor attn_output, torch::Tensor weight, torch::Tensor output, float eps);

void rms_norm_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor out, float epsilon) {
    auto x_c = x.contiguous();
    auto w_c = weight.contiguous();
    rms_norm_cuda_launch(out, x_c, w_c, epsilon);
}

void rms_norm_forward_half(torch::Tensor x, torch::Tensor weight, torch::Tensor out, float epsilon) {
    auto x_c = x.contiguous();
    auto w_c = weight.contiguous();
    // 检查输入是否真的是 Half 类型，防止 Python 传错
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Half, "x must be Half");
    rms_norm_cuda_launch_half(out, x_c, w_c, epsilon);
}

void fused_add_rms_norm_forward(torch::Tensor x, torch::Tensor attn_output, torch::Tensor weight, torch::Tensor output, float eps) {
    if (x.scalar_type() == at::ScalarType::Float) {
        fused_add_rms_norm_fp32_launch(x, attn_output, weight, output, eps);
    } else if (x.scalar_type() == at::ScalarType::BFloat16) {
        fused_add_rms_norm_bf16_launch(x, attn_output, weight, output, eps);
    } else {
        TORCH_CHECK(false, "Unsupported scalar type");
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "rms_norm",           // Python 端调用的函数名：custom_ops_cuda.forward(...)
        &rms_norm_forward,   // 对应的 C++ 函数地址
        "RMSNorm forward (CUDA)" // 提示文档
    );
    m.def("rms_norm_fp16", &rms_norm_forward_half, "RMSNorm forward half(CUDA)");
    m.def("fused_add_norm", &fused_add_rms_norm_forward, "Fused Add and RMSNorm forward");
}