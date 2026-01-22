import torch
import triton
import triton.language as tl

def torch_stand_attn(
    q:torch.Tensor, k:torch.Tensor, v:torch.Tensor
):
    D = q.shape[-1]
    attn_weights = torch.matmul(q, k.T) / (D ** 0.5)
    attn_weights = torch.softmax(attn_weights, dim=-1).matmul(v)
    return attn_weights

def torch_flash_attn(
    q:torch.Tensor, k:torch.Tensor, v:torch.Tensor
):  
    out = torch.zeros_like(q)
    L, D = q.shape
    sqrt_d = D ** 0.5

    for i in range(L):
        qi = q[i, :]
        m_i = torch.tensor(-float('inf'), device=q.device) 
        sum_i = torch.tensor(0.0, device=q.device)
        o_i = torch.zeros(D, device=q.device) # 保持 (D,) 形状

        for j in range(L):
            kj, vj = k[j, :], v[j, :]
            p_ij = torch.matmul(qi, kj.t()) / sqrt_d
            
            # 更新 m
            m_prev = m_i
            m_i = torch.maximum(m_prev, p_ij)

            # 更新 sum
            sum_prev = sum_i
            sum_i = sum_prev * torch.exp(m_prev - m_i) + torch.exp(p_ij - m_i)

            # 更新 O
            o_i = o_i * sum_prev / sum_i * torch.exp(m_prev - m_i) + vj * torch.exp(p_ij - m_i) / sum_i

        out[i, :] = o_i

    return out

@triton.jit
def flash_attn_kernel(
    q, k, v, out, 
    sm_scale,                           # 1/sqrt(D)
    stride_qm, stride_qk,               # Q 的步长
    stride_kn, stride_kk,               # K 的步长
    stride_vn, stride_vk,               # V 的步长
    stride_om, stride_ok,               # Out 的步长
    L, D,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr 
):
    pid = tl.program_id(0)  # axis=0

    # 设置 Q 的起始行索引
    rm = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    rk = tl.arange(0, BLOCK_K)                                      # Attn 中 D 维度是固定的，通常 block k直接设置为Dim
    q_ptr = q + (rm[:, None] * stride_qm + rk[None, :] * stride_qk) # 二维广播, [block_m, block_k] 大小
    q_m = tl.load(q_ptr, mask=rm[:, None] < L, other=0)               # 从显存加载到 SRAM

    # 初始化 Online Softmax 的中间变量
    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    sum_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    o_i = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32) # 注意：这里 o_i 存储的是加权累加值（未除以 sum_i）

    # 内层循环：遍历 K, V 的 Block
    for tile_n in range(0, L, BLOCK_N):
        rn = tile_n + tl.arange(0, BLOCK_N)
        
        # 加载 K 块：为了配合 tl.dot(Q, K.T)，我们直接加载转置后的 K
        # K 原始形状 [L, D]，我们要加载成 [D, BLOCK_N]
        k_ptr = k + (rk[:, None] * stride_kk + rn[None, :] * stride_kn)
        k_m = tl.load(k_ptr, mask=rn[None, :] < L, other=0)
        
        # 1. 计算 p_ij (Attention Score)
        # q_m: [BLOCK_M, BLOCK_K], k_m: [BLOCK_K, BLOCK_N] -> p_ij: [BLOCK_M, BLOCK_N]
        p_ij = tl.dot(q_m, k_m)
        p_ij *= sm_scale
        
        # 边界处理：如果 rn 越界，score 设为负无穷
        p_ij = tl.where(rn[None, :] < L, p_ij, -float('inf'))

        # 2. 更新 m_i (Max Score)
        m_prev = m_i
        m_curr = tl.max(p_ij, 1)
        m_i = tl.maximum(m_prev, m_curr)

        # 3. 计算缩放因子 (Online Softmax 核心)
        # alpha 是旧数据的修正，beta 是新数据的修正
        alpha = tl.exp(m_prev - m_i)
        p_exp = tl.exp(p_ij - m_i[:, None]) # [BLOCK_M, BLOCK_N]

        # 4. 更新 sum_i (分母累加)
        sum_i = sum_i * alpha + tl.sum(p_exp, 1)

        # 5. 更新 o_i (分子累加)
        # 先加载 V 块: [BLOCK_N, BLOCK_K]
        v_ptr = v + (rn[:, None] * stride_vn + rk[None, :] * stride_vk)
        v_m = tl.load(v_ptr, mask=rn[:, None] < L, other=0)

        # 修正旧的 o_i 并加上新的贡献
        o_i = o_i * alpha[:, None]
        o_i = tl.dot(p_exp.to(v_m.dtype), v_m, acc=o_i)

    # 循环结束后，统一除以 sum_i 得到最终均值
    o_i = o_i / sum_i[:, None]

    # 将结果写回显存
    out_ptr = out + (rm[:, None] * stride_om + rk[None, :] * stride_ok)
    tl.store(out_ptr, o_i, mask=rm[:, None] < L)
