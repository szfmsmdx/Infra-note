import torch
from torch.nn.modules.transformer import Transformer

class RMS_Norm(torch.nn.Module):
    def __init__(self, hidden_dim, eps=1e-6):
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.ones(hidden_dim))
        self.eps = eps

    def forward(self, x):
        # x : [B, L, D]
        x_mean = torch.mean(x ** 2, dim=-1, keepdim=True)
        return x / torch.sqrt(x_mean + self.eps) * self.gamma

class FFN(torch.nn.Module):
    def __init__(self, in_dim, out_dim, dropout_rate=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.Linear1 = torch.nn.Linear(self.in_dim, self.out_dim, bias=False)
        self.Linear2 = torch.nn.Linear(self.out_dim, self.in_dim, bias=False)
        self.act = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, x):
        return self.dropout(
            self.Linear2(self.act(self.Linear1(x)))
        )
    
class Self_Attention(torch.nn.Module):
    def __init__(self, in_dim, out_dim, num_head=8):
        super().__init__()
        assert out_dim % num_head == 0
        self.head_dim = out_dim // num_head  # 单个头输出维度
        self.num_head = num_head
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.q = torch.nn.Linear(in_dim, out_dim, bias=False)
        self.k = torch.nn.Linear(in_dim, out_dim, bias=False)
        self.v = torch.nn.Linear(in_dim, out_dim, bias=False)
        self.o = torch.nn.Linear(out_dim, out_dim, bias=False)
    
    def forward(self, x):
        B, L, _  = x.shape
        q, k, v, o = self.q(x), self.k(x), self.v(x), self.o(x)
        # split and reshape
        q = q.reshape(B, L, self.num_head, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, L, self.num_head, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, L, self.num_head, self.head_dim).permute(0, 2, 1, 3)

        # attention
        score = torch.softmax((q * k) / torch.sqrt(self.head_dim)) * v  # [B, H, L, H_D]

        # concat
        score_cat = score.permute(0, 2, 1, 3).reshape(B, L, self.out_dim)   # [B, L, D]

        # o_proj
        score_proj = o(score_cat)

        return score_proj