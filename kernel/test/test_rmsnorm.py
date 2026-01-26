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

def benchmark_half():
    device = "cuda"
    # 模拟真实 LLM 场景，把 hidden_size 调大到 4096 (Llama-7B 级别)
    # 大数据量下 FP16 的优势更明显
    bsz, seq_len, hidden_size = 4, 1024, 4096 
    
    # 初始化数据并转为 FP16
    x = torch.randn(bsz * seq_len, hidden_size, device=device).half()
    weight = torch.ones(hidden_size, device=device).half()
    eps = 1e-6
    
    out_cuda_half = torch.zeros_like(x)
    
    # --- 1. 测试 PyTorch 原生 (Half 精度) ---
    # PyTorch 内部也会调用高度优化的 Half 算子
    py_norm = PyRMSNorm(hidden_size).to(device).half()
    
    # Warmup
    for _ in range(10): py_norm(x)
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(100):
        py_out = py_norm(x)
    torch.cuda.synchronize()
    print(f"Python (Half) 版耗时: {(time.time() - start):.4f}s")

    # --- 2. 测试你写的 CUDA V6 (Half + Float4/Uint4) ---
    # 确保你 setup.py 里的函数映射正确
    # Warmup
    for _ in range(10): custom_ops_cuda.forward_half(x, weight, out_cuda_half, eps)
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(100):
        custom_ops_cuda.forward_half(x, weight, out_cuda_half, eps)
    torch.cuda.synchronize()
    print(f"你的 CUDA (Half) 版耗时: {(time.time() - start):.4f}s")

    # --- 3. 验证正确性 ---
    # 注意：FP16 的比较，误差在 1e-3 左右是合理的
    diff = torch.abs(py_out.float() - out_cuda_half.float()).max()
    print(f"Half 版本最大误差: {diff.item()}")

if __name__ == "__main__":
    print("--- 运行 FP32 测试 ---")
    benchmark() 
    print("\n--- 运行 FP16 测试 ---")
    benchmark_half()