import torch
import time
import triton
from tabulate import tabulate 
from flash_attn import torch_stand_attn, torch_flash_attn, triton_flash_attn
import sys


def compute_tflops(ms, L, D):
    # Attention 的计算量约为 4 * L^2 * D (QK^T 计算 + Softmax*V 计算)
    # TFLOPS = 运算量 / (时间 * 10^12)
    flops = 4 * (L ** 2) * D
    tflops = flops / (ms * 1e-3 * 1e12)
    return tflops

def run_benchmark(L, D, methods, device="cuda"):
    q = torch.randn(L, D, device=device, dtype=torch.float16)
    k = torch.randn(L, D, device=device, dtype=torch.float16)
    v = torch.randn(L, D, device=device, dtype=torch.float16)

    results = []

    for name, func in methods.items():
        # 跳过太慢的 Python 循环版本
        if name == "Torch Flash (Py)":
            if L > 128:
                results.append([name, "Skip (too slow)", "-", "-"])
                continue
            else:
                iters = 1
        else:
            iters = 10 if L <= 1024 else 100

        print(f"\n▶ 正在测试: {name} (L={L}, iters={iters})")
        sys.stdout.flush()

        # 预热
        for _ in range(3):
            func(q, k, v)
        torch.cuda.synchronize()

        # 正式计时
        torch.cuda.reset_peak_memory_stats()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for i in range(iters):
            func(q, k, v)
            # 显示进度（每完成 10% 或最后一轮）
            if iters > 1 and (i + 1) % max(1, iters // 10) == 0:
                progress = (i + 1) / iters * 100
                print(f"\r   进度: {progress:5.1f}% ({i+1}/{iters})", end="", flush=True)
        end_event.record()
        torch.cuda.synchronize()

        print(f"\r   完成: {' ' * 30}")  # 清除进度行

        avg_ms = start_event.elapsed_time(end_event) / iters
        peak_mem = torch.cuda.max_memory_allocated()
        base_mem = q.element_size() * q.nelement() * 3  # q,k,v
        extra_mem_mb = (peak_mem - base_mem) / (1024**2)
        tflops = compute_tflops(avg_ms, L, D)

        results.append([name, f"{avg_ms:.4f}", f"{extra_mem_mb:.2f}", f"{tflops:.2f}"])

    return results

def main():
    device = "cuda"
    D = 64
    
    # 定义测试方法
    methods = {
        "Torch Standard": torch_stand_attn,
        "Torch Flash (Py)": torch_flash_attn,
        "Triton Flash v2": triton_flash_attn,
    }

    # 1. 正确性验证 (L=128)
    print("验证正确性 (L=128)...")
    L_test = 128
    q = torch.randn(L_test, D, device=device)
    k = torch.randn(L_test, D, device=device)
    v = torch.randn(L_test, D, device=device)
    
    ref = torch_stand_attn(q, k, v)
    for name, func in methods.items():
        out = func(q, k, v)
        diff = torch.max(torch.abs(ref - out)).item()
        status = "✅" if diff < 1e-3 else "❌"
        print(f"{name:20} Max Diff: {diff:.2e} {status}")

    # 2. 性能对比 (小规模 L=512，为了让 Python 循环版能跑完)
    print("\n[规模: L=512, D=64]")
    res_small = run_benchmark(512, D, methods)
    print(tabulate(res_small, headers=["Method", "Time (ms)", "Mem (MB)", "TFLOPS"], tablefmt="grid"))

    # 3. 性能对比 (大规模 L=4096，观察 Memory Wall)
    print("\n[规模: L=4096, D=64]")
    res_large = run_benchmark(4096, D, methods)
    print(tabulate(res_large, headers=["Method", "Time (ms)", "Mem (MB)", "TFLOPS"], tablefmt="grid"))

if __name__ == "__main__":
    main()