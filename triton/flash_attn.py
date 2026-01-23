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

"""
v2 出发点：
- 并行受限：如果 batch_size 小或者 head 小，那么无法充分利用 SM
- 频繁 HBM 读写：我们的 v1 版本其实不是严格意义上的 v1 版本，原文的 v1 版本为了 backward 方便，实际上是 kv 外循环，q内循环
    这样的问题是：我们需要维护 output 矩阵，频繁从显存中读取、写入 output，这里的 io 开销太大（output 和 q 的大小一致）
- 非运算开销大：softmax中的exp、sum、div 属于 Non-Tensor Core 操作
下面是一个 v2 的实现
"""
@triton.jit
def flash_attn_kernel(
    q, k, v, out,
    stride_ql, stride_qd,
    stride_kl, stride_kd,
    stride_vl, stride_vd,
    stride_ol, stride_od,
    L: int, D: int,
    sm_scale, 
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    # 1. 定位当前程序处理的 Q 分块
    block_id = tl.program_id(0)
    q_block_row = block_id * BLOCK_M + tl.arange(0, BLOCK_M)
    q_block_col = tl.arange(0, BLOCK_D)

    # 计算 Q 指针并加载 Q (保持在 SRAM 中)
    q_ptr = q + (q_block_row[:, None] * stride_ql + q_block_col[None, :] * stride_qd)
    q_m = tl.load(q_ptr, mask=q_block_row[:, None] < L, other=0.0)

    # 2. 初始化 Online Softmax 统计量
    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32) 
    sum_i = tl.zeros([BLOCK_M], dtype=tl.float32)           
    # acc 存储分子部分的加权和，最后统一除以 sum_i
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)    

    # 3. 遍历 KV Block
    for tile_n in range(0, L, BLOCK_N):
        k_block_row = tile_n + tl.arange(0, BLOCK_N)
        
        # 加载 K (加载为 [BLOCK_D, BLOCK_N] 形状，即 K^T)
        k_ptr = k + (k_block_row[None, :] * stride_kl + q_block_col[:, None] * stride_kd)
        k_m = tl.load(k_ptr, mask=k_block_row[None, :] < L, other=0.0)

        # 计算 p_ij = (Q * K^T) * scale
        p_ij = tl.dot(q_m.to(tl.float16), k_m.to(tl.float16)) # 必须强制转为 fp16
        p_ij *= sm_scale
        # Mask 掉越界的序列部分
        p_ij = tl.where(k_block_row[None, :] < L, p_ij, -float('inf'))

        # --- Online Softmax 核心逻辑 ---
        # [block_m] 大小的，因为 online 流的是一个q和不同的k
        m_prev = m_i
        m_curr = tl.max(p_ij, 1)
        m_i = tl.maximum(m_prev, m_curr)

        # 计算修正因子
        alpha = tl.exp(m_prev - m_i)
        p_exp = tl.exp(p_ij - m_i[:, None])

        # 更新分母 (sum)
        sum_i = sum_i * alpha + tl.sum(p_exp, 1)

        # 更新分子 (acc)
        # 加载 V (形状 [BLOCK_N, BLOCK_D])
        v_ptr = v + (k_block_row[:, None] * stride_vl + q_block_col[None, :] * stride_vd)
        v_m = tl.load(v_ptr, mask=k_block_row[:, None] < L, other=0.0)
        
        # 这里是 v2 的精髓：先修正旧的 acc，再加新的贡献
        acc = acc * alpha[:, None]
        acc = tl.dot(p_exp.to(tl.float16), v_m.to(tl.float16), acc=acc)

    # 4. 循环结束，计算最终结果 O = acc / sum
    o_i = acc / sum_i[:, None]

    # 5. 写回显存
    out_ptr = out + (q_block_row[:, None] * stride_ol + q_block_col[None, :] * stride_od)
    tl.store(out_ptr, o_i, mask=q_block_row[:, None] < L)

def triton_flash_attn(q, k, v):
    # 确保输入是连续的，并且转换到 float16，因为 tl.dot 对 fp16 支持最好
    q = q.to(torch.float16)
    k = k.to(torch.float16)
    v = v.to(torch.float16)
    
    L, D = q.shape
    sm_scale = 1.0 / (D ** 0.5)
    out = torch.empty_like(q)

    # 硬件优化配置
    # BLOCK_M 和 BLOCK_N 通常选 64 或 128
    # BLOCK_D 必须是 D（假设 D 比较小如 64, 128）
    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_D = D 

    # 定义 Grid：在序列长度 L 方向上开启多少个并行 Program
    # 每个 Program 处理 BLOCK_M 行 Q
    grid = (triton.cdiv(L, BLOCK_M),)

    flash_attn_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
        out.stride(0), out.stride(1),
        L, D,
        sm_scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=4, # 并行执行的 Warp 数量
        num_stages=2 # 流水线阶段数（针对 H100/A100 的内存优化）
    )
    return out