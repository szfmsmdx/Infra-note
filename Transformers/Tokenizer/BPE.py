# BPE toy tokenizer
import torch
import pickle
from collections import OrderedDict, defaultdict
from tqdm import tqdm

class BPE_Tokenizer():
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.token2id = OrderedDict()
        self.id2token = OrderedDict()
        self.merges = []

    def _pair_stats(self, tokens: list[str], stats: dict[tuple[str, str], int]):
        """维护 state : pair count 计数对"""
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            stats[pair] = stats.get(pair, 0) + 1
        return stats
    
    def _merge(self, tokens_list: list[list[str]]):
        """
        维护 self.merge : [token, token] token 合并规则
        返回
            new_token : 合并后的 token 
            new_token_list : token_list 经过 merge 规则合并后的 token
        """
        stats = {}
        for tokens in tokens_list:
            self._pair_stats(tokens, stats)
        
        if not stats:
            return None, tokens_list

        target_pair = max(stats, key=stats.get)

        new_token = target_pair[0] + target_pair[1]
        self.merges.append(target_pair)

        new_token_list = []
        for tokens in tokens_list:
            merged_token = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == target_pair:
                    merged_token.append(new_token)
                    i += 2
                else:
                    merged_token.append(tokens[i])
                    i += 1
            new_token_list.append(merged_token)

        return new_token, new_token_list
    
    def _build_vocab(self, tokens_list:list[list[str]]):
        """
        根据收敛的 tokens_list 构建 vocab
        """
        vocab = set(tok for tokens in tokens_list for tok in tokens)
        for i, token in enumerate(sorted(vocab)):
            self.id2token[i] = token
            self.token2id[token] = i
        
    def encode(self, text:str):
        """
        输入 text : str
        输出 idx
        """
        # 这里注意，因为我们的 merge 是按照最大频率的次序慢慢添加的所以greedy版本自然按照贪心实现
        tokens = list(text)
        for a, b in self.merges:
            merged_token = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
                    merged_token.append(a+b)
                    i += 2
                else:
                    merged_token.append(tokens[i])
                    i += 1

            tokens = merged_token
        return [self.token2id[t] for t in tokens]
    
    def decode(self, ids:list[int]):
        """ idx to text """
        tokens = [self.id2token[i] for i in ids]
        return "".join(tokens)
    
    def train(self, train_texts:list[str]):
        """
        合并更新 vocab
        """
        tokens_list = [list(text) for text in train_texts]
        # 给一个小一点的初始化否则小 vocab_size 设置会直接短路
        vocab = set()

        for _ in tqdm(range(self.vocab_size)):  # 这是轮数，最终 vocab_size是 build vocab实现的
            new_token, tokens_list = self._merge(tokens_list)
            if new_token is None:
                break
            vocab.add(new_token)

        self._build_vocab(tokens_list)

    def save(self, save_path):
        with open(save_path, "wb") as f:
            pickle.dump((self.token2id, self.id2token, self.merges, self.vocab_size), f)

    def load(self, save_path):
        with open(save_path, "rb") as f:
            self.token2id, self.id2token, self.merges, self.vocab_size = pickle.load(f)

if __name__ == "__main__":
    cn = open("/data3/szf/Infra-note/Transformers/Tokenizer/data/train-cn.txt", 'r', encoding='utf-8').read()
    en = open("/data3/szf/Infra-note/Transformers/Tokenizer/data/train-en.txt", 'r', encoding='utf-8').read()

    tokenizer = BPE_Tokenizer(vocab_size=200)
    
    tokenizer.train([cn, en])
    print(tokenizer.token2id)
    print(len(tokenizer.token2id))