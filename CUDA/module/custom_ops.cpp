#include <torch/extension.h>

// 1. 声明在 rms_norm.cu 中定义的 launch 函数
// 这个函数名必须和你在 .cu 文件最后写的那个函数名一模一样
void rms_norm_cuda_launch(torch::Tensor out, torch::Tensor x, torch::Tensor weight, float epsilon);

// 2. 定义一个包装函数，供 PyBind 调用
// 这里我们可以做一些简单的形状检查
void rms_norm_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor out, float epsilon) {
    // 确保输入是连续的（CUDA 优化通常要求内存连续）
    auto x_c = x.contiguous();
    auto w_c = weight.contiguous();
    
    // 调用 .cu 文件里的发射器
    rms_norm_cuda_launch(out, x_c, w_c, epsilon);
}

// 3. PyBind11 绑定
// TORCH_EXTENSION_NAME 会自动获取 setup.py 里定义的那个名字（custom_ops_cuda）
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward",           // Python 端调用的函数名：custom_ops_cuda.forward(...)
        &rms_norm_forward,   // 对应的 C++ 函数地址
        "RMSNorm forward (CUDA)" // 提示文档
    );
}