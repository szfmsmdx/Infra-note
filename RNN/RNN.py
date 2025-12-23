import torch
import torch.nn as nn
from typing import Dict
from Tokenizer import Tokenizer, build_vocab

class RNN_Cell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # 遵顼 torch lienar 的 (out, in) 形状
        self.weight_x = nn.Parameter(torch.empty(hidden_dim, input_dim))    
        self.weight_h = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.bias = nn.Parameter(torch.empty(hidden_dim))

        # 初始化
        nn.init.xavier_uniform_(self.weight_h)
        nn.init.orthogonal_(self.weight_x)
        nn.init.zeros_(self.bias)

    def forward(self, x, hidden_state):
        """
        x : [B, 1, input_dim]
        hidden_state : [B, hidden_dim]
        """
        hidden_state_next = torch.tanh(
            x @ self.weight_x.t() + hidden_state @ self.weight_h.t() + self.bias
        )
        return hidden_state_next

class RNN_Net(nn.Module):
    def __init__(self, vocab_size, input_dim, hidden_dim, pad_idx, eos_idx, device):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.pad_idx = pad_idx
        self.eos_idx = eos_idx
        self.device = device

        self.embedding = nn.Embedding(vocab_size, input_dim, padding_idx=pad_idx).to(device)
        self.cell = RNN_Cell(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
        self.lm_head = torch.nn.Linear(in_features=hidden_dim, out_features=vocab_size, bias=False).to(device)

    def step(self, token_idx, h):
        """
        单个 token 的一步前向
        token_idx : [B]
        h         : [B, H]
        """
        x = self.embedding(token_idx.to(self.device))
        h = self.cell(x, h)
        return h

    def encode(self, input_idx, h_0=None):
        """sample 的一步前向"""
        input_idx.to(self.device)
        B, T = input_idx.shape
        h = torch.zeros(B, self.hidden_dim, device=input_idx.device) if h_0 is None else h_0
        h_all = []

        for t in range(T):
            mask = (input_idx[:, t] != self.pad_idx)
            h_new = self.step(input_idx[:, t], h)
            h = torch.where(mask.unsqueeze(-1), h_new, h)
            h_all.append(h)

        return torch.stack(h_all, dim=1)
    
    def forward(self, input_idx, h_0 = None):
        """sample 的一步输出过程"""
        input_idx.to(self.device)
        hidden_state = self.encode(input_idx, h_0)
        logits = self.lm_head(hidden_state)
        return logits

    def generate(self, input_idx, max_new_token=20, temperature=1.0):
        input_idx.to(self.device)
        B = input_idx.size(0)
        h_all = self.encode(input_idx)
        h = h_all[:, -1, :]

        alive_mask = torch.ones(B, dtype=torch.bool, device=self.device)    # [B]
        generate_token = []
        for _ in range(max_new_token):
            logit = self.lm_head(h) / temperature
            next_token = torch.argmax(logit, dim=-1).to(self.device)    # [B]
            next_token = torch.where(
                alive_mask, next_token, torch.full_like(next_token, self.pad_idx)
            )
            alive_mask &= (next_token != self.eos_idx)

            h_new = self.step(next_token, h)
            h = torch.where(alive_mask.unsqueeze(-1), h_new, h) # unsqueeze 广播对齐

            generate_token.append(next_token.unsqueeze(1))
            if not torch.any(alive_mask):
                break

        return torch.cat(generate_token, dim=1)


if __name__ == "__main__":
    sentences = [
        "你好！",
        "今天天气真好。",
        "我喜欢学习。",
        "你好，世界！"
    ]
    vocab = build_vocab(sentences)
    print(vocab, sep='\n')
    embed_dim, hidden_dim = 16, 32
    tokenizer = Tokenizer(vocab)
    rnn = RNN_Net(
        vocab_size=tokenizer.vocab_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        pad_idx=tokenizer.pad_id,
        eos_idx=tokenizer.eos_id
    )

    input_idx = tokenizer.encode(sentences)
    new_token = rnn.generate(input_idx)
    generate = tokenizer.decode(new_token)
    print(generate)