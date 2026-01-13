import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from Config import LlamaConfig
from Tokenizer.BPE import BPE_Tokenizer

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_emb(q, k):
    """
    q,k : [B, H, L, D]
    """


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
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, mask=None, kv_cache=None):
        bsz, q_len, _ = x.shape
        # q,k,v [B, H, L, D]
        q = self.q_proj(x).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 这里预留给 RoPE (Rotary Positional Embedding)
        # q, k = apply_rotary_emb(q, k, ...) 

        # 基础 KV Cache 逻辑
        if kv_cache is not None:
            k, v = kv_cache.update(k, v)

        # Scaled Dot-Product Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        
        attn = F.softmax(scores.float(), dim=-1).type_as(q)
        output = torch.matmul(attn, v) # [bsz, num_heads, q_len, head_dim]
        
        output = output.transpose(1, 2).reshape(bsz, q_len, -1)
        return self.o_proj(output)

class LlamaLayer(nn.Module):
    def __init__(self, dim, num_heads, intermediate_size):
        super().__init__()
        self.attention = LlamaAttention(dim, num_heads)
        self.mlp = LlamaMLP(dim, intermediate_size)
        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)

    def forward(self, x, mask=None, kv_cache=None):
        # Pre-Norm 结构
        h = x + self.attention(self.attention_norm(x), mask, kv_cache)
        out = h + self.mlp(self.ffn_norm(h))
        return out

class LlamaGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([
            LlamaLayer(config.dim, config.num_heads, config.intermediate_size)
            for _ in range(config.num_layers)
        ])
        self.norm = RMSNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

    def forward(self, input_ids, mask=None):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x, mask)
        return self.lm_head(self.norm(x))

if __name__ == "__main__":
    # 1. 初始化配置与模型
    tokenizer = BPE_Tokenizer.load("./Tokenizer/tokenizer.pt")
    config = LlamaConfig(tokenizer)
    model = LlamaGPT(config)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Llama-style 模型初始化完成，参数量: {num_params / 1e6:.2f} M")

    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    mask = torch.full((seq_len, seq_len), float("-inf"))
    mask = torch.triu(mask, diagonal=1) 

    print(f"正在进行前向传播测试，输入形状: {input_ids.shape}...")
    try:
        logits = model(input_ids, mask=mask)
        print(f"前向传播成功！输出 Logits 形状: {logits.shape}") # 预期 [2, 10, 32000]
        
        if torch.isnan(logits).any():
            print("警告：输出包含 NaN，检查初始化或 Norm 层")
        else:
            print("输出数值正常。")
            
    except Exception as e:
        print(f"前向传播失败，错误信息: {e}")

    print("\n--- 优化点确认 ---")
    print(f"当前 Attention 结构: Multi-Head Attention (MHA)")
    print(f"Q/K/V Heads 数量: {config.num_heads}/{config.num_heads}/{config.num_heads}")
    print(f"下一步目标: 修改为 GQA (例如 K/V Heads = 2)")