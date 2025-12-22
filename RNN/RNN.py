import torch
from typing import Dict, List

class Embedding(torch.nn.Module):
    def __init__(self, vocab : Dict[str, int], embed_dim : int):
        super().__init__(Embedding)
        self.vocab = vocab
        self.embed_dim = embed_dim
        self.vocab_size = len(vocab)
        self.embedding = torch.nn.Embedding(
            self.vocab_size, self.embed_dim
        )

    def embed(self, tokens : List[str]):
        """tokens -> index -> vec"""
        # 用列表推导式代替 for 更加高效
        indices = [self.vocab[token] for token in tokens]
        indices = torch.tensor(indices, dtype=torch.long)
        return self.embedding(indices)


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__(Encoder)

    def forward(self, hidden_state, input):
        pass

class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__(Decoder)
    def forward(self, hidden_state, input):
        pass

class RNN(torch.nn.Module):
    def __init__(self):
        super().__init__(RNN)

    def forward(self):
        pass