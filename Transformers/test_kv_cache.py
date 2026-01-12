import time
import torch
from Transformer_kvcache import Model
from Config import T5Config
from Tokenizer.BPE import BPE_Tokenizer

def run_stress_test(model, src_ids, use_cache, max_tokens=256):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    
    start_time = time.time()
    
    with torch.no_grad():
        output = model.generate(src_ids, use_cache=use_cache, max_new_token=max_tokens)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
        peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
    else:
        peak_allocated, peak_reserved = 0, 0
        
    total_time = time.time() - start_time
    # 排除起始 token 的数量
    num_generated = output.size(1) - 1
    ms_per_token = (total_time / num_generated) * 1000
    
    return total_time, ms_per_token, peak_allocated, peak_reserved

if __name__ == "__main__":
    # 配置更深、更长的模型以观察差异
    torch.manual_seed(42)
    tokenizer = BPE_Tokenizer.load("/data3/szf/Infra-note/Transformers/Tokenizer/tokenizer.pt")
    config = T5Config(tokenizer)
    config.num_layers = 12 
    config.model_dim = 768
    config.num_head = 12
    config.ffn_dim = 3072
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_mem = 0
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        before_model = torch.cuda.memory_allocated()
        model = Model(config).to(device).eval()
        base_mem = (torch.cuda.memory_allocated() - before_model) / (1024**2)
        print(f"📦 模型参数占用显存: {base_mem:.2f} MB")
    else:
        model = Model(config).eval()

    # 模拟高压输入：Batch Size = 4, 输入长度 256, 生成长度 512
    batch_size = 4
    input_len = 256
    gen_len = 1024
    src_ids = torch.randint(10, 4000, (batch_size, input_len)).to(device)
    
    print(f"\n🚀 开始高压测试 [Batch: {batch_size}, 生成长度: {gen_len}]")
    print("-" * 60)

    # 1. 无缓存模式 (Baseline)
    # 注意：如果显存不够，请调小 gen_len，因为 O(N^2) 的显存增长非常快
    try:
        t_no, ms_no, alloc_no, res_no = run_stress_test(model, src_ids, use_cache=False, max_tokens=gen_len)
        print(f"【无缓存全量模式】:")
        print(f"  > 总耗时: {t_no:.2f}s | 每 Token 均摊: {ms_no:.2f}ms")
        print(f"  > 峰值分配 (含权重): {alloc_no + base_mem:.2f}MB")
        print(f"  > 系统预留 (接近smi): {res_no:.2f}MB")
    except RuntimeError as e:
        print("❌ 无缓存模式 OOM (显存溢出)！这证明了全量计算对显存的巨大压力。")

    print("-" * 60)

    # 2. KV Cache 模式
    t_ca, ms_ca, alloc_ca, res_ca = run_stress_test(model, src_ids, use_cache=True, max_tokens=gen_len)
    print(f"【KV Cache 增量模式】:")
    print(f"  > 总耗时: {t_ca:.2f}s | 每 Token 均摊: {ms_ca:.2f}ms")
    print(f"  > 峰值分配 (含权重): {alloc_ca + base_mem:.2f}MB")
    print(f"  > 系统预留 (接近smi): {res_ca:.2f}MB")

    print("-" * 60)
    print(f"🔥 加速比: {t_no / t_ca:.2f}x")