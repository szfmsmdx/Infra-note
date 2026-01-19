import torch
import torch.nn as nn
import custom_ops_cuda
from model.llama import RMSNorm  # 直接导入原始实现

def run_fused_profile():
    # 保持与 benchmark 一致的参数
    B, L, D = 4, 1024, 4096
    device = "cuda"
    dtype = torch.float16 
    eps = 1e-6

    # 准备数据
    # x 模拟残差连接中的输入，attn_out 模拟 Attention 的输出
    x = torch.randn(B * L, D, device=device, dtype=dtype)
    attn_out = torch.randn(B * L, D, device=device, dtype=dtype)
    
    # 实例化原始 RMSNorm 并加载到 GPU 和指定精度
    # 这样可以确保控制变量：权重初始化、EPS 以及 forward 内部的 record_function 是一致的
    naive_norm = RMSNorm(D, eps=eps).to(device).to(dtype)
    
    # Ours 算子需要的输出 Buffer
    output = torch.empty_like(x)

    print(f"Profiling with Dtype: {dtype}, Dim: {D}")
    
    # --- 1. Warmup ---
    # 预热非常重要，排除 CUDA Context 初始化和冷启动的影响
    for _ in range(20):
        # 模拟 Naive 的 Fused 逻辑：先加后 Norm
        _ = naive_norm(x + attn_out)
        # 模拟 Ours 的 Fused 逻辑
        custom_ops_cuda.fused_add_norm(x.clone(), attn_out, naive_norm.weight, output, eps)

    # --- 2. Nsight Profiling ---
    torch.cuda.synchronize()
    
    # 使用 NVTX 标记，在 Nsight Systems 的时间轴上可以清晰看到这两个区域
    print("Starting NVTX ranges...")

    with torch.cuda.nvtx.range("Profile_Naive_Implementation"):
        for _ in range(100):
            res = x + attn_out 
            _ = naive_norm(res)
    
    torch.cuda.synchronize()

    with torch.cuda.nvtx.range("Profile_Ours_Fused_Implementation"):
        for _ in range(100):
            custom_ops_cuda.fused_add_norm(x, attn_out, naive_norm.weight, output, eps)

    torch.cuda.synchronize()
    print("Profiling complete. Please check the results in Nsight Systems/Compute.")

if __name__ == "__main__":
    run_fused_profile()