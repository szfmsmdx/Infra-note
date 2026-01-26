import torch
import time
import custom_ops_cuda
from model.llama import LlamaLayer as NaiveLayer

# JIT 优化的 RMSNorm 模拟
@torch.jit.script
def jit_rms_norm(x, weight, eps: float):
    norm_x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return norm_x * weight

class JITLayer(NaiveLayer):
    def forward(self, x, freqs_cis, mask=None, kv_cache=None, start_pos=0):
        # 仅优化 Norm 部分
        h = x + self.attention(jit_rms_norm(x, self.attention_norm.weight, self.attention_norm.eps), freqs_cis)
        return h + self.mlp(jit_rms_norm(h, self.ffn_norm.weight, self.ffn_norm.eps))

def benchmark():
    configs = [torch.float32, torch.float16, torch.bfloat16]
    B, L, D = 4, 1024, 4096
    from model.Config import LlamaConfig
    cfg = LlamaConfig(dim=D)
    
    print(f"{'Dtype':<10} | {'Method':<12} | {'Time(ms)':<10} | {'Mem(MB)':<10}")
    print("-" * 50)

    for dtype in configs:
        x = torch.randn(B, L, D, device="cuda", dtype=dtype)
        freqs = torch.randn(L, D//16, device="cuda").to(torch.complex64)
        
        methods = {
            "Naive": NaiveLayer(cfg).to("cuda").to(dtype),
            "JIT": JITLayer(cfg).to("cuda").to(dtype),
            "Ours": None # 逻辑在下面
        }
        
        # 特殊处理 Ours，因为它需要修改 forward 逻辑调用 custom_ops
        from model.llama_opt import LlamaLayer as OptLayer
        methods["Ours"] = OptLayer(cfg).to("cuda").to(dtype)

        for name, model in methods.items():
            # 预热
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            for _ in range(10): model(x, freqs)
            
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(100):
                model(x, freqs)
            torch.cuda.synchronize()
            avg_time = (time.time() - start) * 10 # 100次总和转为单次ms平均
            
            mem = torch.cuda.max_memory_allocated() / 1024 / 1024
            print(f"{str(dtype):<10} | {name:<12} | {avg_time:>9.2f} | {mem:>9.2f}")

if __name__ == "__main__":
    benchmark()