import torch
import triton
import triton.language as tl

def flash_attn_torch(
    q:torch.Tensor, k:torch.Tensor, v:torch.Tensor
):
    """
    q、k、v - 2D 矩阵
    """
    L, D = q.shape()
    