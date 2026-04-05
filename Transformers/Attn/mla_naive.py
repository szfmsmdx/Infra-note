import torch
import torch.nn as nn

def apply_rope(x, freqs_cis):
    # x: [B, S, H, D] 或 [B, S, D]
    # freqs_cis: [S, D/2], 复数形式
    xq = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq * freqs_cis).flatten(-2)
    return xq_out.type_as(x)


class MLA(nn.Module):
    def __init__(
        self, 
        dim,                # model_dim
        n_heads,            # 总注意力头数
        q_lora_rank,        # q 低秩维度
        kv_lora_rank,       # kv 低秩维度
        qk_nope_head_dim,   # qk 非位置编码维度
        qk_rope_head_dim,   # qk 位置编码维度
        v_head_dim,         # v  头维度
        max_batch_size,
        max_seq_len
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.softmax_scale = self.qk_head_dim ** -0.5

        # cache
        self.register_buffer("k_nope_cache", torch.zeros(max_batch_size, max_seq_len, n_heads, qk_nope_head_dim), persistent=False)
        self.register_buffer("k_rope_cache", torch.zeros(max_batch_size, max_seq_len, qk_rope_head_dim), persistent=False)
        self.register_buffer("v_cache",      torch.zeros(max_batch_size, max_seq_len, n_heads, v_head_dim), persistent=False)
        
        # init weight
        self.q_norm = nn.RMSNorm(self.q_lora_rank)
        self.k_norm = nn.RMSNorm(self.kv_lora_rank)
        self.wq_a = nn.Linear(self.dim, self.q_lora_rank)
        self.wq_b = nn.Linear(self.q_lora_rank, self.qk_head_dim * self.n_heads)
        self.wkv_a = nn.Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        self.wkv_b = nn.Linear(self.kv_lora_rank, (self.qk_nope_head_dim + self.v_head_dim) * self.n_heads)
        self.wo    = nn.Linear(self.n_heads * self.v_head_dim, self.dim)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor = None
    ):
        # naive 版本
        # x
        # ├─→ [Q路径] wq_a → RMSNorm → wq_b → split → [q_nope, q_rope→RoPE]
        # │
        # └─→ [KV路径] wkv_a → split
        #                 ├─ c_kv → RMSNorm → wkv_b → split → [k_nope, v]
        #                 └─ k_rope → RoPE

        # [q_nope; q_rope] · [k_nope; k_rope]ᵀ → scale → mask → softmax → · v → wo → y


        # rope
        bsz, seqlen, _ = x.shape
        end_pos = start_pos + seqlen

        q = self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.view(bsz, seqlen, self.n_heads, self.qk_head_dim)
        q_nope, q_rope = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_rope = apply_rope(q_rope, freqs_cis)  # 这里 q:(B, S, H, D), 每个头配了单独的 rope

        kv = self.wkv_a(x)
        # 这里 k_rope 分出来没有 num_head 这个维度
        c_kv, k_rope = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        c_kv = self.wkv_b(self.k_norm(c_kv)).view(bsz, seqlen, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(c_kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_rope = apply_rope(k_rope, freqs_cis)

        # 写入 kv cache
        self.k_nope_cache[:bsz, start_pos:end_pos] = k_nope
        self.k_rope_cache[:bsz, start_pos:end_pos] = k_rope
        self.v_cache[:bsz, start_pos:end_pos]      = v

        scores = (
            torch.einsum("bshd,bthd->bsht", q_nope, self.k_nope_cache[:bsz, :end_pos]) +
            torch.einsum("bshr,btr->bsht",  q_rope, self.k_rope_cache[:bsz, :end_pos])
        ) * self.softmax_scale

        if mask is not None:
            scores += mask.unsqueeze(1)

        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)

        out = torch.einsum("bsht,bthd->bshd", scores, self.v_cache[:bsz, :end_pos])
        out = self.wo(out.flatten(2))
        return out