import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from Config import LlamaConfig
from Tokenizer.BPE import BPE_Tokenizer
from kvcache import LlamaLayerCache
from torch.autograd.profiler import record_function
# from Modules.MLA import *

def precompute_freqs_cis(dim: int, end: int, base: float=1e4):
    theta = torch.pow(base, -2 * torch.arange(0, dim / 2) / dim)    # [dim / 2]
    i_vals = torch.arange(end).unsqueeze(1)                         # [end, 1]    
    freqs = i_vals * theta                                          # [end, dim / 2] 广播
    return torch.polar(torch.ones_like(freqs), freqs)               # [end, dim / 2]

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

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        # norm_x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # return (norm_x.float()).type_as(x) * self.weight
        with record_function("OP: RMSNorm"):
            norm_x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            return (norm_x.float()).type_as(x) * self.weight

class LlamaMLP(nn.Module):
    def __init__(self, dim, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(dim, intermediate_size, bias=False)
        self.up_proj = nn.Linear(dim, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, dim, bias=False)
    def forward(self, x):
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

    def forward(self, x, freqs_cis, mask=None, kv_cache=None, start_pos=0):
        bsz, q_len, _ = x.shape
        q = self.q_proj(x).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rope_emb(q.contiguous(), k.contiguous(), freqs_cis[start_pos : start_pos + q_len])

        if kv_cache is not None:
            k, v = kv_cache.update(k, v, start_pos)
            
        k, v = repeat_kv(k, self.num_group), repeat_kv(v, self.num_group)

        attn_score = torch.matmul(q, k.transpose(-2, -1)) / sqrt(self.head_dim)
        if mask is not None:
            attn_score += mask
        score = torch.softmax(attn_score.float(), dim=-1).to(v.dtype).matmul(v).type_as(q)

        output = score.transpose(1, 2).reshape(bsz, q_len, -1)
        return self.o_proj(output)

class LlamaLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = LlamaAttention(config)
        self.mlp = LlamaMLP(config.dim, config.intermediate_size)
        self.attention_norm = RMSNorm(config.dim)
        self.ffn_norm = RMSNorm(config.dim)

    def forward(self, x, freqs_cis, mask=None, kv_cache=None, start_pos=0):
        # h = x + self.attention(self.attention_norm(x), freqs_cis, mask, kv_cache, start_pos)
        # out = h + self.mlp(self.ffn_norm(h))
        with record_function("LlamaLayer_Total"):
            with record_function("Attention_Block"):
                h = x + self.attention(self.attention_norm(x), freqs_cis, mask, kv_cache, start_pos)
            with record_function("MLP_Block"):
                out = h + self.mlp(self.ffn_norm(h))
        return out

class LlamaGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([LlamaLayer(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.freqs_cis = precompute_freqs_cis(
            dim=config.dim // config.num_heads,
            end=config.max_new_tokens,
            base=config.rope_base
        )

    def forward(self, input_ids, start_pos=0, kv_caches=None, mask=None):
        x = self.embed(input_ids)
        freqs_cis = self.freqs_cis.to(x.device)
        for i, layer in enumerate(self.layers):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            x = layer(x, freqs_cis, mask=mask, kv_cache=layer_cache, start_pos=start_pos)
        return self.lm_head(self.norm(x))

    @torch.no_grad()
    def generate(self, prompt_tokens, max_gen_len):
        bsz, prompt_len = prompt_tokens.shape
        total_len = min(self.config.max_new_tokens, prompt_len + max_gen_len)
        device = prompt_tokens.device

        # 初始化每层的 Cache
        kv_caches = [LlamaLayerCache(self.config, bsz, device) for _ in range(self.config.num_layers)]
        tokens = torch.full((bsz, total_len), self.config.pad_token_id, dtype=torch.long, device=device)
        tokens[:, :prompt_len] = prompt_tokens
        
        # 1. Prefill 阶段
        mask = torch.triu(torch.full((prompt_len, prompt_len), float("-inf"), device=device), diagonal=1)
        logits = self.forward(tokens[:, :prompt_len], start_pos=0, kv_caches=kv_caches, mask=mask)
        next_token = torch.argmax(logits[:, -1, :], dim=-1)
        tokens[:, prompt_len] = next_token
        
        # 2. Decode 阶段
        for cur_pos in range(prompt_len + 1, total_len):
            # Decode 阶段输入 q_len=1，不需要 mask
            logits = self.forward(tokens[:, cur_pos-1 : cur_pos], start_pos=cur_pos-1, kv_caches=kv_caches)
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            tokens[:, cur_pos] = next_token
            if (next_token == self.config.eos_token_id).all(): break
            
        return tokens[:, :cur_pos + 1]
    
