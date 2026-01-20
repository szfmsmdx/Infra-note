import torch
import torch.nn as nn
import custom_ops_cuda
from model.llama import RMSNorm  # 直接导入原始实现

def run_fused_profile():
    B, L, D = 4, 1024, 4096
    device = "cuda"
    dtype = torch.float16 
    eps = 1e-6

    x = torch.randn(B * L, D, device=device, dtype=dtype)
    attn_out = torch.randn(B * L, D, device=device, dtype=dtype)
    
    naive_norm = RMSNorm(D, eps=eps).to(device).to(dtype)
    
    output = torch.empty_like(x)

    print(f"Profiling with Dtype: {dtype}, Dim: {D}")
    
    # --- 1. Warmup ---
    for _ in range(20):
        _ = naive_norm(x + attn_out)
        custom_ops_cuda.fused_add_norm(x.clone(), attn_out, naive_norm.weight, output, eps)

    torch.cuda.synchronize()
    
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