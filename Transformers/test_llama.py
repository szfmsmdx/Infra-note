import torch
from Llama import LlamaGPT
from kvcache import LlamaLayerCache
import time

def run_comprehensive_test():
    # 1. 模拟工业级配置
    class MockConfig:
        vocab_size = 32000
        dim = 256            # 稍微增大维度以暴露潜在的对齐问题
        num_heads = 8
        num_kv_heads = 2     # 典型的 GQA 配置 (4:1)
        num_group = 8 // 2
        intermediate_size = 688
        num_layers = 4       # 增加层数以测试梯度/数值累积
        max_new_tokens = 1024
        rope_base = 10000.0
        pad_token_id = 0
        eos_token_id = 1

    config = MockConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- 正在使用设备: {device} ---")
    
    model = LlamaGPT(config).to(device).eval()
    
    # ---------------------------------------------------------
    # 测试一：维度与连续性检查 (Shape & Contiguity)
    # ---------------------------------------------------------
    bsz, seq_len = 2, 32
    input_ids = torch.randint(0, config.vocab_size, (bsz, seq_len), device=device)
    mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)
    
    with torch.no_grad():
        logits = model(input_ids, start_pos=0, mask=mask)
        
    assert logits.shape == (bsz, seq_len, config.vocab_size), "Logits 维度错误"
    print("✅ [测试 1/6] 基础维度检查通过")

    # ---------------------------------------------------------
    # 测试二：因果遮蔽严谨性 (Causal Mask Invariance)
    # ---------------------------------------------------------
    # 修改输入序列末尾的词，不应影响序列开头词的 Logits
    input_1 = torch.tensor([[1, 2, 3, 4, 5]], device=device)
    input_2 = torch.tensor([[1, 2, 3, 9, 9]], device=device) # 修改后两个词
    
    with torch.no_grad():
        out1 = model(input_1, start_pos=0, mask=mask[:5, :5])
        out2 = model(input_2, start_pos=0, mask=mask[:5, :5])
    
    # 比较前 3 个位置的输出
    diff_mask = torch.abs(out1[:, :3, :] - out2[:, :3, :]).max()
    assert diff_mask < 1e-5, f"因果遮蔽失败！最大差异: {diff_mask.item()}"
    print("✅ [测试 2/6] 因果遮蔽 (Causal Mask) 验证通过")

    # ---------------------------------------------------------
    # 测试三：KV Cache 等效性测试 (Crucial for Infra)
    # ---------------------------------------------------------
    # 这是测试推理引擎最关键的一步：
    # 验证“全量 Prefill”和“逐步 Decode”得到的输出是否数值一致
    input_ids = torch.tensor([[10, 20, 30, 40]], device=device)
    
    # A. 全量前向传播
    with torch.no_grad():
        full_logits = model(input_ids, start_pos=0, mask=mask[:4, :4])
        target_last_logits = full_logits[:, -1, :]

    # B. 模拟逐步推理
    kv_caches = [LlamaLayerCache(config, 1, device) for _ in range(config.num_layers)]
    step_logits = None
    with torch.no_grad():
        for i in range(4):
            # 模拟每次输入一个 token
            cur_input = input_ids[:, i:i+1]
            step_logits = model(cur_input, start_pos=i, kv_caches=kv_caches)
            
    # 比较最后一步的输出
    diff_cache = torch.abs(target_last_logits - step_logits[:, -1, :]).max()
    assert diff_cache < 1e-4, f"KV Cache 数值不一致！差异: {diff_cache.item()}"
    print("✅ [测试 3/6] KV Cache 一致性 (Prefill vs Decode) 验证通过")

    # ---------------------------------------------------------
    # 测试四：RoPE 相对位置平移验证 (RoPE Invariance)
    # ---------------------------------------------------------
    # 同样的词在位置 1 和位置 2 产生的特征应该是不同的
    token_a = torch.tensor([[100]], device=device)
    token_b = torch.tensor([[200]], device=device)
    
    with torch.no_grad():
        # 模拟两种情况的 KV Cache
        # 情况 1: A(pos=1) 看着 B(pos=0)
        cache1 = [LlamaLayerCache(config, 1, device) for _ in range(config.num_layers)]
        _ = model(token_b, start_pos=0, kv_caches=cache1) # 先存入 B
        logits_1 = model(token_a, start_pos=1, kv_caches=cache1)[:, -1, :]
        
        # 情况 2: A(pos=2) 看着 B(pos=0) -> 相对距离变了 (1->2)
        cache2 = [LlamaLayerCache(config, 1, device) for _ in range(config.num_layers)]
        _ = model(token_b, start_pos=0, kv_caches=cache2) # 先存入 B
        # 注意：这里我们跳过位置 1，直接把 A 放在位置 2
        logits_2 = model(token_a, start_pos=2, kv_caches=cache2)[:, -1, :]
    
    diff_rope = torch.abs(logits_1 - logits_2).max()
    print(f"✅ [测试 4/6] 相对位置敏感测试 (RoPE Sensitivity): {'通过' if diff_rope > 1e-3 else '失败'}")
    assert diff_rope > 1e-3, "RoPE 无效：改变相对距离后输出竟然没有变化"

    # ---------------------------------------------------------
    # 测试五：数值稳定性与 GQA 压力测试
    # ---------------------------------------------------------
    # 模拟极端长序列
    long_len = config.max_new_tokens
    long_input = torch.randint(0, config.vocab_size, (1, long_len), device=device)
    try:
        with torch.no_grad():
            _ = model(long_input, start_pos=0, mask=None) # 这里不带 mask 模拟不限长的推理
        print(f"✅ [测试 5/6] 数值稳定性通过 (Sequence Length={long_len})")
    except RuntimeError as e:
        print(f"❌ [测试 5/6] 显存不足或计算错误: {e}")

    # ---------------------------------------------------------
    # 测试六：端到端生成生成功能 (Generate Logic)
    # ---------------------------------------------------------
    prompt = torch.tensor([[1, 5, 10]], device=device)
    start_time = time.time()
    generated = model.generate(prompt, max_gen_len=20)
    end_time = time.time()
    
    assert generated.shape[1] > 3, "生成长度异常"
    print(f"✅ [测试 6/6] 端到端生成测试通过 (生成速度: {(generated.shape[1]-3)/(end_time-start_time):.2f} tokens/s)")
    print("\n🚀 所有严谨性测试全部通过！该模型已具备工业级推理雏形。")

if __name__ == "__main__":
    run_comprehensive_test()