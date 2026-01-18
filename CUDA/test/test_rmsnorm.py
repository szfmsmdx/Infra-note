import torch
import time
import custom_ops_cuda  
from model.llama import RMSNorm as PyRMSNorm

def benchmark():
    device = "cuda"
    bsz, seq_len, hidden_size = 4, 1024, 512
    x = torch.randn(bsz * seq_len, hidden_size, device=device)
    weight = torch.ones(hidden_size, device=device)
    eps = 1e-6
    
    # 准备输出容器
    out_cuda = torch.zeros_like(x)
    
    # --- 1. 测试 Python (PyTorch 原生) 速度 ---
    py_norm = PyRMSNorm(hidden_size).to(device)
    # Warmup
    for _ in range(10): py_norm(x)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        py_out = py_norm(x)
    torch.cuda.synchronize()
    print(f"Python 版耗时: {(time.time() - start):.4f}s")

    # --- 2. 测试你写的 CUDA 版速度 ---
    # Warmup
    for _ in range(10): custom_ops_cuda.forward(x, weight, out_cuda, eps)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        custom_ops_cuda.forward(x, weight, out_cuda, eps)
    torch.cuda.synchronize()
    print(f"你的 CUDA 版耗时: {(time.time() - start):.4f}s")

    # --- 3. 验证正确性 (非常重要！) ---
    # 只有算的对，比速度才有意义
    diff = torch.abs(py_out - out_cuda).max()
    print(f"最大误差: {diff.item()}")

if __name__ == "__main__":
    benchmark()