import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from Config import LlamaConfig
from Tokenizer.BPE import BPE_Tokenizer
from Transformers.Modules.MLA import *

def precompute_freqs_cis(dim: int, end: int, base: float=1e4):
    theta = torch.pow(base, -2 * torch.arange(0, dim / 2) / dim)    # [dim / 2]
    i_vals = torch.arange(end).unsqueeze(1)                         # [end, 1]    
    freqs = i_vals * theta                                          # [end, dim / 2] 广播
    return torch.polar(torch.ones_like(freqs), freqs)               # [end, dim / 2]

def repeat_kv(x : torch.Tensor, num_rpt: int):
    """
    x : [B, H_kv, L, D]
    """
    bsz, n_kv_heads, q_len, head_dim = x.shape
    if num_rpt == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(bsz, n_kv_heads, num_rpt, q_len, head_dim)
        .reshape(bsz, n_kv_heads * num_rpt, q_len, head_dim)
    )

def apply_rope_emb(q, k, freqs_cis):
    """
    q : [B, H, L, D]
    k : [B, H, L, D]
    freqs_cis : [L, D / 2]
    """
    # reshape 为 [B, H, L, D / 2, 2] -> [B, H, L, D / 2] complex 复数
    q_ = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
    k_ = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))

    # freqs_cis [L, D/2] -> [1, 1, L, D/2]
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(1)

    # 广播 -> 相乘[B, H, L, D / 2] -> view_as_real [B, H, L, D/2, 2] -> flattn(-2) (D/2, 2) ->(D)
    # 这里 freqs_cis 和 q_ 都是复数相乘，复数相乘已经实现了相邻元素的旋转了
    q_out = torch.view_as_real(q_ * freqs_cis).flatten(-2)  # GQA 硬编码变成 5D 向量会崩溃，采用 -2 的相对编码
    k_out = torch.view_as_real(k_ * freqs_cis).flatten(-2)

    return q_out.type_as(q), k_out.type_as(k)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight

class LlamaMLP(nn.Module):
    def __init__(self, dim, intermediate_size):
        super().__init__()
        # Llama 使用 SwiGLU，这里由三个线性层组成
        self.gate_proj = nn.Linear(dim, intermediate_size, bias=False)
        self.up_proj = nn.Linear(dim, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, dim, bias=False)

    def forward(self, x):
        # SwiGLU: (silu(gate) * up) * down
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class LlamaAttention(nn.Module):
    def __init__(self, config:LlamaConfig):
        super().__init__()
        dim = config.dim
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.num_group = config.num_group
        self.head_dim = dim // self.num_heads
        
        self.q_proj = torch.nn.Linear(dim, self.head_dim * self.num_heads, bias=False)
        self.k_proj = torch.nn.Linear(dim, self.head_dim * self.num_kv_heads, bias=False)
        self.v_proj = torch.nn.Linear(dim, self.head_dim * self.num_kv_heads, bias=False)
        self.o_proj = torch.nn.Linear(dim, dim, bias=False)

    def forward(self, x, freqs_cis, mask=None, kv_cache=None):
        """GQA"""
        bsz, q_len, _ = x.shape
        q = self.q_proj(x).reshape(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)   # [B, H, L, D]
        k = self.k_proj(x).reshape(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rope_emb(q, k, freqs_cis[:q_len])




class LlamaLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.attention = LlamaAttention(config)
        self.mlp = LlamaMLP(config.dim, config.intermediate_size)
        self.attention_norm = RMSNorm(config.dim)
        self.ffn_norm = RMSNorm(config.dim)

    def forward(self, x, freqs_cis, mask=None, kv_cache=None):
        # Pre-Norm 结构
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask, kv_cache)
        out = h + self.mlp(self.ffn_norm(h))
        return out

class LlamaGPT(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([
            LlamaLayer(config)
            for _ in range(config.num_layers)
        ])
        self.norm = RMSNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.freqs_cis = precompute_freqs_cis(
            dim=config.dim // config.num_heads,
            end=config.max_new_tokens,
            base=config.rope_base
        )
        self.config = config

    def forward(self, input_ids, mask=None):
        x = self.embed(input_ids)
        freqs_cis = self.freqs_cis.to(x.device)
        for layer in self.layers:
            x = layer(x, freqs_cis, mask)
        return self.lm_head(self.norm(x))
    
if __name__ == "__main__":
    # 模拟 Config 对象，避免依赖外部文件
    class MockConfig:
        vocab_size = 32000
        dim = 128
        num_heads = 8
        intermediate_size = 256
        num_layers = 2
        max_new_tokens = 512
        rope_base = 10000.0

    config = MockConfig()
    model = LlamaGPT(config)
    
    # 构造数据
    bsz, seq_len = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (bsz, seq_len))
    
    # 构造 Causal Mask
    mask = torch.full((seq_len, seq_len), float("-inf"))
    mask = torch.triu(mask, diagonal=1)

    print(f"开始测试... 输入形状: {input_ids.shape}")
    
    try:
        logits = model(input_ids, mask=mask)
        print("--- 测试通过 ---")
        print(f"Logits 形状: {logits.shape}") # 预期 [2, 16, 32000]
        
        # 简单的一致性检查：RoPE 后的 logits 不应包含 NaN
        if not torch.isnan(logits).any():
            print("数值检查: 正常 (无 NaN)")
        else:
            print("数值检查: 异常 (存在 NaN)")
            
    except Exception as e:
        print(f"测试失败！错误信息: {e}")
        import traceback
        traceback.print_exc()