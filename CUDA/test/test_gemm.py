import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision('highest')
import custom_ops_cuda  # your compiled extension
import time
import argparse
from typing import Callable, Dict, Any, Tuple

# ----------------------------
# Global cache for compiled functions (avoid recompilation)
# ----------------------------
_COMPILED_FUNCTIONS = {}

def get_compiled_matmul(dtype: torch.dtype, shape_key: Tuple[Tuple[int, ...], Tuple[int, ...]]):
    key = ("matmul", dtype, shape_key)
    if key not in _COMPILED_FUNCTIONS:
        def _matmul(x, y):
            return torch.matmul(x, y)
        # Use reduce-overhead mode for small ops like GEMM
        _COMPILED_FUNCTIONS[key] = torch.compile(_matmul, mode="reduce-overhead")
    return _COMPILED_FUNCTIONS[key]

# ----------------------------
# Helper: Memory usage
# ----------------------------
def get_gpu_memory():
    return torch.cuda.memory_allocated() / 1024**3  # GB

# ----------------------------
# Implementations to benchmark
# ----------------------------
def run_torch_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.matmul(a, b)

def run_torch_compile(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    shape_key = (tuple(a.shape), tuple(b.shape))
    compiled_fn = get_compiled_matmul(a.dtype, shape_key)
    return compiled_fn(a, b)

def run_custom_cuda(a: torch.Tensor, b: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    c = torch.empty(a.size(0), b.size(1), device=a.device, dtype=dtype)
    
    if dtype == torch.float32:
        custom_ops_cuda.gemm(a, b, c)  # assumes your binding exports 'gemm'
    elif dtype == torch.float16:
        raise NotImplementedError("FP16 kernel not implemented yet")
    elif dtype == torch.bfloat16:
        raise NotImplementedError("BF16 kernel not implemented yet")
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    
    return c

# ----------------------------
# Benchmark runner
# ----------------------------
def benchmark_impl(
    name: str,
    func: Callable,
    a: torch.Tensor,
    b: torch.Tensor,
    num_warmup: int = 10,
    num_iter: int = 50,
    verify_ref: torch.Tensor = None,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, Any]:
    # Warmup
    for _ in range(num_warmup):
        _ = func(a, b)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    mem_before = get_gpu_memory()

    # Timed runs
    start = time.time()
    for _ in range(num_iter):
        out = func(a, b)
    torch.cuda.synchronize()
    total_time = time.time() - start

    peak_mem = torch.cuda.max_memory_allocated() / 1024**3  # GB
    avg_time_ms = total_time / num_iter * 1000

    M, K = a.shape
    N = b.shape[1]
    tflops = (M * N * K * 2) / (avg_time_ms / 1000) / 1e12

    # Correctness check
    correct = False
    max_diff = float('inf')
    if verify_ref is not None:
        try:
            atol = 1e-2 if dtype in [torch.float16, torch.bfloat16] else 1e-4
            rtol = 1e-2 if dtype in [torch.float16, torch.bfloat16] else 1e-5
            correct = torch.allclose(out, verify_ref, atol=atol, rtol=rtol)
            max_diff = (out - verify_ref).abs().max().item()
        except Exception as e:
            print(f"⚠️  Verification failed for {name}: {e}")

    return {
        "name": name,
        "correct": correct,
        "max_diff": max_diff,
        "avg_time_ms": avg_time_ms,
        "tflops": tflops,
        "peak_mem_gb": peak_mem,
    }

# ----------------------------
# Main test function
# ----------------------------
def run_benchmark(M: int, K: int, N: int, dtypes: list, impls: dict):
    device = "cuda"
    print(f"🧪 Running GEMM benchmark: M={M}, K={K}, N={N}\n")

    for dtype in dtypes:
        print(f"--- Testing dtype: {dtype} ---")
        
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            print("   ⚠️  BF16 not supported on this GPU. Skipping.\n")
            continue

        try:
            a = torch.randn(M, K, device=device, dtype=dtype)
            b = torch.randn(K, N, device=device, dtype=dtype)

            # Reference result
            ref_high = torch.matmul(a.to(torch.float32), b.to(torch.float32))
            ref = ref_high.to(dtype)

            results = []

            for name, func in impls.items():
                if name == "Custom CUDA":
                    func_wrapped = lambda x, y, d=dtype: run_custom_cuda(x, y, d)
                else:
                    func_wrapped = func

                try:
                    res = benchmark_impl(
                        name=name,
                        func=func_wrapped,
                        a=a,
                        b=b,
                        verify_ref=ref,
                        dtype=dtype,
                    )
                    results.append(res)
                except NotImplementedError as e:
                    print(f"   ⚠️  {name} ({dtype}): Not implemented yet.")
                except Exception as e:
                    print(f"   ❌ {name} ({dtype}) failed: {e}")

            # Print results
            print(f"{'Method':<15} {'Correct':<8} {'MaxDiff':<10} {'Time(ms)':<10} {'TFLOPS':<10} {'PeakMem(GB)':<12}")
            print("-" * 75)
            for r in results:
                print(f"{r['name']:<15} {'✅' if r['correct'] else '❌':<8} "
                      f"{r['max_diff']:<10.2e} {r['avg_time_ms']:<10.3f} "
                      f"{r['tflops']:<10.2f} {r['peak_mem_gb']:<12.3f}")
            print()

        except Exception as e:
            print(f"   ❌ Failed to run {dtype}: {e}\n")

# ----------------------------
# Entry point
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=1024)
    parser.add_argument("--K", type=int, default=4096)
    parser.add_argument("--N", type=int, default=1024)
    parser.add_argument("--dtypes", nargs="+", default=["fp32"], 
                        choices=["fp32", "fp16", "bf16"])
    args = parser.parse_args()

    str_to_dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    dtypes = [str_to_dtype[d] for d in args.dtypes]

    # Replace TorchScript with Torch Compile
    implementations = {
        "PyTorch": run_torch_matmul,
        "TorchCompile": run_torch_compile,      # ✅ 替换这里
        "Custom CUDA": lambda a, b: None,       # handled specially
    }

    torch.cuda.empty_cache()
    run_benchmark(args.M, args.K, args.N, dtypes, implementations)