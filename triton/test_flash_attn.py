import torch
import time
from flash_v1 import torch_stand_attn, torch_flash_attn

def benchmark():
    # 为了 Python 循环能跑完，L 不能太大
    L, D = 128, 64 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    q = torch.randn(L, D, device=device)
    k = torch.randn(L, D, device=device)
    v = torch.randn(L, D, device=device)

    print(f"Testing with L={L}, D={D} on {device}\n")

    # --- 1. Standard Attention ---
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    res_stand = torch_stand_attn(q, k, v)
    stand_time = time.time() - start_time
    stand_mem = torch.cuda.max_memory_allocated() / 1024**2

    # --- 2. Manual Flash Attention (Python Loop) ---
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    res_flash = torch_flash_attn(q, k, v)
    flash_time = time.time() - start_time
    flash_mem = torch.cuda.max_memory_allocated() / 1024**2

    # --- 结果比对 ---
    max_diff = torch.max(torch.abs(res_stand - res_flash)).item()
    
    print(f"{'Method':<20} | {'Time (ms)':<12} | {'Peak Mem (MB)':<15}")
    print("-" * 55)
    print(f"{'Torch Standard':<20} | {stand_time*1000:>12.2f} | {stand_mem:>15.2f}")
    print(f"{'Manual Flash (Py)':<20} | {flash_time*1000:>12.2f} | {flash_mem:>15.2f}")
    print(f"\nMax Numerical Difference: {max_diff:.2e}")

    if max_diff < 1e-5:
        print("✅ Correctness Check Passed!")
    else:
        print("❌ Correctness Check Failed!")

if __name__ == "__main__":
    benchmark()