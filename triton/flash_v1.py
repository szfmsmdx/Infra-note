import torch
import triton
import triton.language as tl
from transformers.models.qwen3 import Qwen3Model

def flash_attn_torch(
    q:torch.Tensor, k:torch.Tensor, v:torch.Tensor
):
    """
    q、k、v - 2D 矩阵
    """
    L, D = q.shape()
    for i in range(L):
        for j in range(D):
            qv, kv, vv = q[i,:], k.t()[:,j], v[i,:]
