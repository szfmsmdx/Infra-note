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
        self.register_buffer("c_kv_cache", torch.zeros(max_batch_size, max_seq_len, self.kv_lora_rank))
        
        # init weight
        self.q_norm = nn.RMSNorm(self.q_lora_rank)
        self.k_norm = nn.RMSNorm(self.kv_lora_rank)
        self.w_q_a = nn.Linear(self.dim, self.q_lora_rank)
        self.w_q_b = nn.Linear(self.q_lora_rank, self.qk_head_dim * self.n_heads)
        self.w_kv_a = nn.Linear(self.dim, self.qk_head_dim)
        self.w_uk = nn.Linear(self.qk_nope_head_dim, self.qk_nope_head_dim)
        self.w_uv = nn.Linear(self.qk_nope_head_dim, self.dim)
        
    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor = None
    ):
        # absorb 版本
        # x
        # ├─→ [Q路径] wq_a → RMSNorm → wq_b → split → [q_nope, q_rope → RoPE]
        # │
        # └─→ [KV路径] wkv_a → split → [c_kv(缓存), k_rope → RoPE]
        #
        # attention score:
        #   score = (q_nope @ W_UK) @ c_kv^T + q_rope @ k_rope^T
        #
        # attention output:
        #   out = softmax(scale·score + mask) @ c_kv @ W_UV → wo → y


        # rope
        bsz, seqlen, _ = x.shape
        end_pos = start_pos + seqlen