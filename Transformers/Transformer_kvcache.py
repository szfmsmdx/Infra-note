import torch
from math import sqrt, log
from Config import T5Config
from kvcache import KVcache

from transformers.models.qwen2.modeling_qwen2 import Qwen2Model

def casual_mask(seq_len: int):
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return torch.zeros(seq_len, seq_len).masked_fill(mask, -1e9)

class T5PositionEmbedding(torch.nn.Module):
    def __init__(self, num_head, num_buckets=32, max_distance=128):
        super().__init__()
        self.num_head = num_head
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        # [bucket, num_head] : dim 设置为 num_head 原因是这里作为每个头的偏执 bias, 也算一种多头位置编码
        self.embedding = torch.nn.Embedding(self.num_buckets, self.num_head)    # [bucket, H]

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        """
        输入: relative_position (Tensor)
        输出: bucket_ids (Tensor)
        """
        # embedding 不接受负数，所以这里要考虑 offset 的问题    
        assert num_buckets % 4 == 0
        length = num_buckets // 2     # 一个区间的长度
        log_len = length // 2        # 使用 log 的长度
        abs_relative_position = torch.abs(relative_position)
        bucket_ids = torch.where(
            abs_relative_position <= log_len,
            abs_relative_position, 
            torch.where(
                abs_relative_position < max_distance,
                log_len + torch.round(torch.log(abs_relative_position - log_len)),
                length - 1
            )
        )
        bucket_ids = bucket_ids + torch.where(relative_position > 0, length, 0)
        return bucket_ids

    @staticmethod
    def _t5_relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        """更加平滑、均匀的桶生成方式"""
        num_buckets //= 2 # 单向桶数，例如 16
        res = 0
        n = -relative_position # T5 习惯：计算目标相对于当前的偏移
        
        # 2. 处理正负半区 (未来/过去)
        res += (n < 0).to(torch.long) * num_buckets
        n = torch.abs(n)

        max_exact = num_buckets // 2 # 精确区边界，例如 8
        is_small = n < max_exact
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / 
            log(max_distance / max_exact) * (num_buckets - max_exact)
        ).to(torch.long)

        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
        res += torch.where(is_small, n, val_if_large)
        return res
    
    def forward(self, q_len, k_len):
        """生成这个 seq 对应的 relative id 矩阵"""
        seq_q = torch.arange(0, q_len, device=self.embedding.weight.device) # 保持设备一致
        seq_k = torch.arange(0, k_len, device=self.embedding.weight.device)
        relative_id = seq_q[:, None] - seq_k[None, :]
        relative_bucket_id = self._relative_position_bucket(relative_id, self.num_buckets, self.max_distance).long()    # [L, L]
        position_bias = self.embedding(relative_bucket_id)  # [L, L, H]
        position_bias = position_bias.permute(2, 0, 1).unsqueeze(0)
        return position_bias

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
    def __init__(self, in_dim, hidden_dim, dropout_rate=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        self.Linear1 = torch.nn.Linear(self.in_dim, self.hidden_dim, bias=False)
        self.Linear2 = torch.nn.Linear(self.hidden_dim, self.in_dim, bias=False)
        self.act = torch.nn.ReLU()

    def forward(self, x):
        return self.Linear2(self.act(self.Linear1(x)))
    
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
    
    def forward(self, x, position_embedding=None, mask=None, pask_kv_cache=None):
        B, L, _  = x.shape
        q, k, v = self.q(x), self.k(x), self.v(x)

        # split and reshape
        # shape : [B, H, L, D]
        q = q.reshape(B, L, self.num_head, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, L, self.num_head, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, L, self.num_head, self.head_dim).permute(0, 2, 1, 3)

        if pask_kv_cache:
            k_cache, v_cache = pask_kv_cache
            k = torch.cat([k_cache, k], dim=-2)
            v = torch.cat([v_cache, v], dim=-2)

        # attention
        # score : [B, H, L, L]
        attn_score = torch.matmul(q, k.permute(0, 1, 3, 2)) / sqrt(self.head_dim)
        if position_embedding is not None:
            attn_score = attn_score + position_embedding    # 这里和 T5 的embedding长度对应
        if mask is not None:
            mask = mask.view(1, 1, mask.size(-2), mask.size(-1)).to(x.device)
            attn_score += mask
        score = torch.softmax(attn_score, dim=-1).matmul(v)

        # concat
        score_cat = score.permute(0, 2, 1, 3).reshape(B, L, self.out_dim)   # [B, L, D]

        # o_proj
        score_proj = self.o(score_cat)

        return score_proj, (k, v)

class Cross_Attention(torch.nn.Module):
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
    
    def forward(self, x, memory, position_embedding=None, mask=None, memory_cache=None):
        """
        x : [B, Lt, D] (Lt is L of target)
        memory : [B, Ls, D] (Ls if L of source)
        return attn_score
        """
        B, Lt, _ = x.shape
        q = self.q(x)
        q = q.reshape(B, Lt, self.num_head, self.head_dim).permute(0, 2, 1, 3)  # [B, H, Lt, D]

        if memory_cache:
            k, v = memory_cache
        else:
            B, Ls, _ = memory.shape
            k, v = self.k(memory), self.v(memory)
            k = k.reshape(B, Ls, self.num_head, self.head_dim).permute(0, 2, 1, 3)
            v = v.reshape(B, Ls, self.num_head, self.head_dim).permute(0, 2, 1, 3)

        # attention
        # score : [B, H, Lt, Ls]
        attn_score = torch.matmul(q, k.permute(0, 1, 3, 2)) / sqrt(self.head_dim)
        if position_embedding is not None:
            attn_score += position_embedding
        if mask is not None:
            mask = mask.view(1, 1, mask.size(-2), mask.size(-1)).to(x.device)
            attn_score += mask
        score = torch.softmax(attn_score, dim=-1).matmul(v)

        # concat
        score_cat = score.permute(0, 2, 1, 3).reshape(B, Lt, self.out_dim)   # [B, Lt, D]

        # o_proj
        score_proj = self.o(score_cat)

        return score_proj, (k, v)
    
class Encode_Layer(torch.nn.Module):
    def __init__(self, model_dim, num_head, ffn_dim, dropout_rate=0.1):
        super().__init__()
        self.model_dim = model_dim      # Attn 的 in_dim 和 out_dim 是一样的
        self.num_head = num_head
        self.ffn_dim = ffn_dim
        self.dropout_rate = dropout_rate

        self.dropout = torch.nn.Dropout(p=self.dropout_rate)
        self.attn_norm = RMS_Norm(self.model_dim)
        self.self_attn = Self_Attention(self.model_dim, self.model_dim, self.num_head)
        self.mlp_norm = RMS_Norm(self.model_dim)
        self.mlp = FFN(self.model_dim, self.ffn_dim, self.dropout_rate)

    def forward(self, x, position_embed=None):
        attn_norm_x = self.attn_norm(x)
        attn_x = self.dropout(self.self_attn(attn_norm_x, position_embedding=position_embed)) + x
        mlp_norm_x = self.mlp_norm(attn_x)
        mlp_x = self.mlp(mlp_norm_x) + attn_x
        return mlp_x

class Encoder(torch.nn.Module):
    def __init__(
            self, num_layers, vocab_size, model_dim, num_head, ffn_dim, dropout_rate = 0.1
        ):
        super().__init__()
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.model_dim = model_dim
        self.num_head = num_head
        self.ffn_dim = ffn_dim
        self.dropout_rate = dropout_rate

        self.embedding = torch.nn.Embedding(vocab_size, self.model_dim)
        self.encode_layers = torch.nn.ModuleList([
            Encode_Layer(model_dim, num_head, ffn_dim, dropout_rate) 
            for _ in range(num_layers)
        ])
        self.norm = RMS_Norm(self.model_dim)
        self.position_embed = T5PositionEmbedding(self.num_head, 32, max_distance=int(2 ** 10))

    def forward(self, input_ids, attention_mask=None):
        """
        input_ids: [B, L]
        attention_mask: [B, L] (1 代表有效, 0 代表 padding)
        """
        batch_size, seq_len = input_ids.shape
        
        if attention_mask is not None:
            extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_mask = (1.0 - extended_mask) * -1e9
        else:
            extended_mask = 0

        position_embedding = self.position_embed(seq_len)
        x = self.embedding(input_ids)
        
        for layer in self.encode_layers:
            x = layer(x, position_embed=position_embedding + extended_mask)
        x = self.norm(x)
        return x

class Decode_Layer(torch.nn.Module):
    def __init__(self, model_dim, num_head, ffn_dim, dropout_rate=0.1):
        super().__init__()
        self.model_dim = model_dim
        self.num_head = num_head
        self.ffn_dim = ffn_dim
        self.dropout_rate = dropout_rate

        self.dropout = torch.nn.Dropout(p=self.dropout_rate)
        self.self_attn_norm  = RMS_Norm(self.model_dim)
        self.self_attn = Self_Attention(self.model_dim, self.model_dim, self.num_head)
        self.cross_attn_norm = RMS_Norm(self.model_dim)
        self.cross_attn = Cross_Attention(self.model_dim, self.model_dim, self.num_head)
        self.mlp_norm = RMS_Norm(self.model_dim)
        self.mlp = FFN(self.model_dim, self.ffn_dim, self.dropout_rate)

    def forward(self, x, memory, position_embedding, casual_mask, pask_kv_cache=None, memeory_cache=None):
        attn_norm_x = self.self_attn_norm(x)
        attn_x, pask_kv_cache = self.dropout(self.self_attn(attn_norm_x, mask=casual_mask, pask_kv_cache=pask_kv_cache)) + x

        cross_norm_x = self.cross_attn_norm(attn_x)
        cross_x, memeory_cache = self.dropout(self.cross_attn(cross_norm_x, memory, position_embedding, memeory_cache=memeory_cache)) + attn_x

        mlp_norm_x = self.mlp_norm(cross_x)
        mlp_x = self.dropout(self.mlp(mlp_norm_x)) + cross_x
        return mlp_x, pask_kv_cache, memeory_cache

class Decoder(torch.nn.Module):
    def __init__(
            self, num_layers, vocab_size, model_dim, num_head, ffn_dim, dropout_rate = 0.1
        ):
        super().__init__()
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.model_dim = model_dim
        self.num_head = num_head
        self.ffn_dim = ffn_dim
        self.dropout_rate = dropout_rate

        self.embedding = torch.nn.Embedding(vocab_size, self.model_dim)
        self.decode_layers = torch.nn.ModuleList([
            Decode_Layer(model_dim, num_head, ffn_dim, dropout_rate) 
            for _ in range(num_layers)
        ])
        self.norm = RMS_Norm(self.model_dim)
        self.position_embed = T5PositionEmbedding(self.num_head, 32, max_distance=int(2 ** 10))

    def forward(self, input_ids, memory, decode_cache_list:dict=None, memory_cache_list:dict=None):
        if input_ids.size(1) != 1:
            mask = casual_mask(input_ids.size(1))
        else:
            mask = None
        x = self.embedding(input_ids)

        for i, layer in enumerate(self.decode_layers):
            pask_kv_cache = decode_cache_list.get(i, None) if decode_cache_list else None
            memory_cache = memory_cache_list.get(i, None) if memory_cache_list else None
            x, pask_kv_cache_i, memory_cache_i = layer(x, memory, None, mask, pask_kv_cache, memory_cache)
            if decode_cache_list:
                decode_cache_list[i] = pask_kv_cache_i
            if memory_cache_list:
                memory_cache_list[i] = memory_cache_i
        x = self.norm(x)
        return x

class Model(torch.nn.Module):
    def __init__(self, config:T5Config):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config.num_layers, config.vocab_size, config.model_dim, 
                               config.num_head, config.ffn_dim, config.dropout_rate)
        self.decoder = Decoder(config.num_layers, config.vocab_size, config.model_dim, 
                               config.num_head, config.ffn_dim, config.dropout_rate)
        self.lm_head = torch.nn.Linear(config.model_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.decoder.embedding.weight

    def forward(self, source_ids, target_ids):
        attention_mask = (source_ids != self.config.pad_token_id).long()
        
        memory = self.encoder(source_ids, attention_mask=attention_mask)
        decoder_output = self.decoder(target_ids, memory)
        logits = self.lm_head(decoder_output)
        return logits

    def generate(
            self, 
            source_ids, 
            use_cache=False, 
            max_new_token=50
        ):
        # init kv_cache
        cache_manager = KVcache()

        # prefill 


        # decode