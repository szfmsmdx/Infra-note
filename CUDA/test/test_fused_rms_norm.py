import torch
import time
from model.Config import LlamaConfig
from model.llama import LlamaLayer as OriginalLayer
from model.llama_opt import LlamaLayer as OptimizedLayer

def benchmark_layer():
    device = "cuda"
    dtype = torch.float32 # 先用 float32 测试，成功后再改 bf16
    config = LlamaConfig(dim=4096, num_heads=32, num_kv_heads=32)
    config.num_group = 1 # 强制修正
    
    x = torch.randn(4, 1024, config.dim, device=device, dtype=dtype)
    freqs_cis = torch.randn(1024, config.dim // 64, device=device).to(torch.complex64)

    layer_orig = OriginalLayer(config).to(device).to(dtype)
    layer_opt = OptimizedLayer(config).to(device).to(dtype)
    layer_opt.load_state_dict(layer_orig.state_dict())

    def run_test(layer, name):
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(50): layer(x, freqs_cis)
        torch.cuda.synchronize()
        print(f"{name} 耗时: {time.time()-start:.4f}s")

    run_test(layer_orig, "Original")
    run_test(layer_opt, "Optimized")

if __name__ == "__main__":
    benchmark_layer()