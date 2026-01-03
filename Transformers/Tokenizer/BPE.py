# BPE toy tokenizer
import pickle
from collections import OrderedDict, defaultdict
from tqdm import tqdm
import re
from functools import lru_cache

SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\w+| ?[^\s\w]+|\s+(?!\S)|\s+"""

class BPE_Tokenizer():
    def __init__(self, vocab_size=2000):
        super().__init__()
        self.vocab_size = vocab_size
        self.token2id = OrderedDict()
        self.id2token = OrderedDict()
        self.merges = []    # ((id, id), id)
        self.merges_rank = {}   # ((id, id), rank) 越小
        self.special_tokens = {}

    def _pre_tokenize(self, text : str):
        """
        预先将 text 分词为 word level
        """
        return re.findall(SPLIT_PATTERN, text)
    
    def _get_stats(self, word_counts : dict[tuple[int], str]):
        """stats : (pair, int)"""
        stats = defaultdict(int)      
        for word_byte in word_counts:
            count = word_counts[word_byte]
            for i in range(len(word_byte) - 1):
                pair = word_byte[i], word_byte[i + 1]   # pair 是 (int, int)
                stats[pair] += count
        return stats

    def _merge_token(self, word_count : dict[tuple[int], int], stats):
        """
        word_count : byte level dict
        """
        target_pair = max(stats, key=stats.get)
        new_token = self.id2token[target_pair[0]] + self.id2token[target_pair[1]]
        new_token_idx = len(self.id2token)
        rank = len(self.merges_rank)

        # 更新词表
        self.id2token[new_token_idx] = new_token
        self.token2id[new_token] = new_token_idx
        self.merges.append((target_pair, new_token_idx))
        self.merges_rank[target_pair] = rank
        del stats[target_pair]

        # 更新计数器
        new_word_count = defaultdict(int)
        for word_idx, count in word_count.items():
            # 修改 word_idx 中 target_pair 部分
            if target_pair[0] not in word_idx:
                new_word_count[word_idx] = count    # 用原来的
                continue

            new_word_idx = []
            i = 0
            while i < len(word_idx):
                if i + 1 < len(word_idx) and (word_idx[i], word_idx[i + 1]) == target_pair:
                    if new_word_idx:    # 删左边
                        left_token_idx = new_word_idx[-1]
                        stats[(left_token_idx, word_idx[i])] -= count
                        if stats[(left_token_idx, word_idx[i])] == 0:   # 如果是 0 则删除防止内存泄漏
                            del stats[(left_token_idx, word_idx[i])]
                        stats[(left_token_idx, new_token_idx)] += count
                    if i + 2 < len(word_idx):
                        right_token_idx = word_idx[i + 2]
                        stats[(target_pair[1], right_token_idx)] -=count
                        if stats[(target_pair[1], right_token_idx)] == 0:
                            del stats[(target_pair[1], right_token_idx)]
                        stats[new_token_idx, right_token_idx] += count
                    new_word_idx.append(new_token_idx)
                    i += 2
                else:
                    new_word_idx.append(word_idx[i])
                    i += 1

            new_word_count[tuple(new_word_idx)] = count

        return new_word_count, stats

    def train(self, trian_text : list[str]):
        # word_count[word_idx : tuple[int], count : int]
        self.merges.clear()
        self.merges_rank.clear()
        self._encode_word.cache_clear()
        
        word_count = defaultdict(int)
        for text in trian_text:
            words = self._pre_tokenize(text)
            for word in words:
                word_idx = tuple(word.encode("utf-8"))    # encode 完是id!!!
                word_count[word_idx] += 1

        # 初始化词表, 因为 word byte 的范围是 0-255，所以我们直接用 0-255 进行初始化
        for i in range(256):
            self.id2token[i] = bytes([i])
            self.token2id[bytes([i])] = i

        # merge
        stats = self._get_stats(word_count)
        for _ in tqdm(range(self.vocab_size - len(self.token2id))):
            word_count, stats = self._merge_token(word_count, stats)

    @lru_cache(maxsize=10000)
    def _encode_word(self, word_ids : tuple[int]):
        word_ids = list(word_ids)
        while len(word_ids) >= 2:   # 说明可能可以合并
            stats = {}
            for i in range(len(word_ids) - 1):
                pair = (word_ids[i], word_ids[i + 1])
                if pair in self.merges_rank:
                    stats[pair] = self.merges_rank[pair]

            if not stats:
                break

            pair_to_merge = min(stats, key=stats.get)   # 越小优先级越高
            rank = self.merges_rank[pair_to_merge]  # 空间换时间，不用继续遍历原表格
            new_id = self.merges[rank][1]

            # 替换
            new_ids = []
            i = 0
            while i < len(word_ids):
                if i < len(word_ids) - 1 and (word_ids[i], word_ids[i + 1]) == pair_to_merge:
                    new_ids.append(new_id)
                    i += 2 
                else:
                    new_ids.append(word_ids[i])
                    i += 1
            word_ids = new_ids

        return tuple(word_ids)  # 转成原格式 tuple

    def encode(self, text: str, add_special_tokens=True):
        res = []
        words_list = self._pre_tokenize(text)
        for word in words_list:
            word_ids = tuple(word.encode("utf-8"))
            res.extend(self._encode_word(word_ids))
        
        if add_special_tokens:
            # 更加通用的 Special Token 注入逻辑
            bos_id = self.special_tokens.get("<bos>") or self.special_tokens.get("[CLS]")
            eos_id = self.special_tokens.get("<eos>") or self.special_tokens.get("[SEP]")
            
            if bos_id is not None:
                res = [bos_id] + res
            if eos_id is not None:
                res.append(eos_id)
                
        return res
                
        return res
    def decode(self, ids: list[int], skip_special_tokens=True):
        byte_list = b""
        decoded_text = ""
        for i in ids:
            if i in self.id2token:
                token = self.id2token[i]
                if isinstance(token, bytes):
                    byte_list += token
                elif isinstance(token, str):    # 说明是特殊字符
                    if not skip_special_tokens:
                        decoded_text += byte_list.decode("utf-8", errors='replace')
                        byte_list = b"" # 清空缓冲区
                        decoded_text += token
                    else:
                        pass
                        
        decoded_text += byte_list.decode("utf-8", errors='replace')
        return decoded_text

    def save(self, save_path):
        with open(save_path, "wb") as f:
            pickle.dump((self.token2id, self.id2token, self.merges, self.vocab_size, self.special_tokens), f)

    def load(self, save_path):
        with open(save_path, "rb") as f:
            data = pickle.load(f)
            if len(data) == 5:
                self.token2id, self.id2token, self.merges, self.vocab_size, self.special_tokens = data
            else:
                self.token2id, self.id2token, self.merges, self.vocab_size = data
                self.special_tokens = {}
        for i, (pair, new_id) in enumerate(self.merges):
            self.merges_rank[pair] = i

        self._encode_word.cache_clear()

    def add_special_token(self, tokens : list[str]):
        for token in tokens:
            if token not in self.special_tokens and token not in self.token2id:
                new_id = len(self.id2token)
                self.special_tokens[token] = new_id
                self.id2token[new_id] = token 
                self.token2id[token] = new_id

if __name__ == "__main__":
    # cn = open("./data/train-cn.txt", 'r', encoding='utf-8').read()
    # en = open("./data/train-en.txt", 'r', encoding='utf-8').read()

    tokenizer = BPE_Tokenizer()
    
    # tokenizer.train([cn, en])
    # print(tokenizer.token2id)
    print(len(tokenizer.token2id))

    # tokenizer.save("./tokenizer.pt")
    tokenizer.load("./tokenizer.pt")
    print(tokenizer.vocab_size)

    prompt = "今天天气真不错！"
    encode_idx = tokenizer.encode(prompt)
    print(prompt, encode_idx)
    decode_text = tokenizer.decode(encode_idx)
    print(prompt, "------ after decode:", decode_text)

    tokenizer.add_special_token(["<pad>", "<eos>"])
    print(f"Encode T5 style: {tokenizer.encode('今天')}") # output: [..., <eos>_id]