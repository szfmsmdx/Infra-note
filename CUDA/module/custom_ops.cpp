#include <torch/extension.h>

void rms_norm_cuda_launch(torch::Tensor out, torch::Tensor x, torch::Tensor weight, float epsilon);
void rms_norm_cuda_launch_half(torch::Tensor out, torch::Tensor x, torch::Tensor weight, float epsilon);

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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward",           // Python 端调用的函数名：custom_ops_cuda.forward(...)
        &rms_norm_forward,   // 对应的 C++ 函数地址
        "RMSNorm forward (CUDA)" // 提示文档
    );
    m.def("forward_half", &rms_norm_forward_half, "RMSNorm forward half(CUDA)");
}