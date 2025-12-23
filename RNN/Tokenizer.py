from typing import List, Dict
import torch

SPECIAL_VOCAB = [
    "<bos>", "<eos>", "<pad>", "<unknow>"
]

def build_vocab(sentences: List[str]):
    vocab = {}
    for idx, token in enumerate(SPECIAL_VOCAB):
        vocab[token] = idx

    offset = len(SPECIAL_VOCAB)

    vocab_set = set()
    for sentence in sentences:
        for char in sentence:
            vocab_set.add(char)

    for i, char in enumerate(sorted(vocab_set)):
        vocab[char] = i + offset

    return vocab

class Tokenizer():
    def __init__(self, vocab : Dict[str, int]):
        self.vocab = vocab  # token2id
        self.id2token = {i:t for t, i in vocab.items()}
        self.vocab_size = len(vocab)

        self.bos_id = vocab["<bos>"]
        self.eos_id = vocab["<eos>"]
        self.pad_id = vocab["<pad>"]
        self.unk_id = vocab["<unknow>"]

    def encode(self, texts:List[str], add_special_token=True):
        batch_idx = []
        for text in texts:
            idx = []
            if add_special_token:
                idx.append(self.bos_id)
            idx += [self.vocab.get(char, self.unk_id) for char in text]
            if add_special_token:
                idx.append(self.eos_id)
            batch_idx.append(idx)

        # padding
        max_len = max(len(idx) for idx in batch_idx)
        for i in range(len(batch_idx)):
            batch_idx[i] = batch_idx[i] + [self.pad_id] * (max_len - len(batch_idx[i]))

        return torch.tensor(batch_idx, dtype=torch.long)
    
    def decode(self, batch_idx:List[List[int]], skip_special_token=True):
        batch_tokens = []
        for idx in batch_idx:
            tokens = []
            for i in idx:
                token = self.id2token.get(int(i), "<unknow>")
                if skip_special_token and token in SPECIAL_VOCAB:
                    continue
                tokens.append(token)
            batch_tokens.append("".join(tokens))
        return batch_tokens
            