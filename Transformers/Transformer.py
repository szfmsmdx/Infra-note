import torch
from torch.nn.modules.transformer import Transformer
from math import sqrt, log

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
    
    def forward(self, seq_len):
        """生成这个 seq 对应的 relative id 矩阵"""
        seq = torch.arange(0, seq_len, device=self.embedding.weight.device) # 保持设备一致
        relative_id = seq[:, None] - seq[None, :]   # [L, L]
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
    
    def forward(self, x, position_embedding=None):
        B, L, _  = x.shape
        q, k, v = self.q(x), self.k(x), self.v(x)
        # split and reshape
        # shape : [B, H, L, D]
        q = q.reshape(B, L, self.num_head, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, L, self.num_head, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, L, self.num_head, self.head_dim).permute(0, 2, 1, 3)

        # attention
        # score : [B, H, L, L]
        attn_score = torch.matmul(q, k.permute(0, 1, 3, 2)) / sqrt(self.head_dim)
        if position_embedding is not None:
            attn_score = attn_score + position_embedding
        score = torch.softmax(attn_score, dim=-1).matmul(v)

        # concat
        score_cat = score.permute(0, 2, 1, 3).reshape(B, L, self.out_dim)   # [B, L, D]

        # o_proj
        score_proj = self.o(score_cat)

        return score_proj
    
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

    def forward(self, input_ids):
        """input_ids : [B, L]"""
        position_embedding = self.position_embed(input_ids.size(1))
        x = self.embedding(input_ids)
        for layer in self.encode_layers:
            x = layer(x, position_embedding)
        x = self.norm(x)
        return x

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
    
    def forward(self, x, memory, position_embedding=None, mask=None):
        """
        x : [B, Lt, D] (Lt is L of target)
        memory : [B, Ls, D] (Ls if L of source)
        return attn_score
        """
        B, Lt, _ = x.shape
        B, Ls, _ = memory.shape
        q, k, v = self.q(x), self.k(memory), self.v(memory)
        # split and reshape
        q = q.reshape(B, Lt, self.num_head, self.head_dim).permute(0, 2, 1, 3)  # [B, H, Lt, D]
        k = k.reshape(B, Ls, self.num_head, self.head_dim).permute(0, 2, 1, 3)  # [B, H, Ls, D]
        v = v.reshape(B, Ls, self.num_head, self.head_dim).permute(0, 2, 1, 3)  # [B, H, Ls, D]

        # attention
        # score : [B, H, Lt, Ls]
        attn_score = torch.matmul(q, k.permute(0, 1, 3, 2)) / sqrt(self.head_dim)
        if position_embedding is not None:
            attn_score += position_embedding
        if mask is not None:
            mask = mask.view(1, 1, mask.size(-2), mask.size(-1))
            attn_score += mask
        score = torch.softmax(attn_score, dim=-1).matmul(v)

        # concat
        score_cat = score.permute(0, 2, 1, 3).reshape(B, Lt, self.out_dim)   # [B, Lt, D]

        # o_proj
        score_proj = self.o(score_cat)

        return score_proj

class Decode_Layer(torch.nn.Module):
    def __init__(self, model_dim, num_head, ffn_dim, dropout_rate=0.1):
        super().__init__()
        self.model_dim = model_dim
        self.num_head = num_head
        self.ffn_dim = ffn_dim
        self.dropout_rate = dropout_rate

        self.dropout = torch.nn.Dropout(p=self.dropout_rate)
        self.self_attn_norm  = RMS_Norm(self.model_dim)
        self.self_attn = Cross_Attention(self.model_dim, self.model_dim, self.num_head)
        self.cross_attn_norm = RMS_Norm(self.model_dim)
        self.cross_attn = Cross_Attention(self.model_dim, self.model_dim, self.num_head)
        self.mlp_norm = RMS_Norm(self.model_dim)
        self.mlp = FFN(self.model_dim, self.ffn_dim, self.dropout_rate)

    def forward(self, x, memory, position_embedding, casual_mask):
        attn_norm_x = self.self_attn_norm(x)
        attn_x = self.dropout(self.self_attn(attn_norm_x, attn_norm_x, mask=casual_mask)) + x

        cross_norm_x = self.cross_attn_norm(attn_x)
        cross_x = self.dropout(self.cross_attn(cross_norm_x, memory, position_embedding)) + attn_x

        mlp_norm_x = self.mlp_norm(cross_x)
        mlp_x = self.dropout(self.mlp(mlp_norm_x)) + cross_x
        return mlp_x

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

    def forward(self, input_ids, memory):
        mask = casual_mask(input_ids.size(1))
        x = self.embedding(input_ids)
        for layer in self.decode_layers:
            x = layer(x, memory, None, mask)
        x = self.norm(x)
        return x

if __name__ == "__main__":
    torch.manual_seed(42)
    
    # 全局超参数
    V_SIZE = 100
    D_MOD = 32
    N_LAY = 2
    N_H = 4
    D_FF = 128
    
    print("--- [总攻测试] 启动 T5 Encoder-Decoder 联合演习 ---")
    
    # 实例化双子星
    encoder = Encoder(num_layers=N_LAY, vocab_size=V_SIZE, model_dim=D_MOD, num_head=N_H, ffn_dim=D_FF)
    decoder = Decoder(num_layers=N_LAY, vocab_size=V_SIZE, model_dim=D_MOD, num_head=N_H, ffn_dim=D_FF)
    
    # 模拟数据
    src_ids = torch.randint(0, V_SIZE, (1, 10)) # 源语言长度 10
    tgt_ids = torch.randint(0, V_SIZE, (1, 5))  # 目标语言长度 5
    
    encoder.eval()
    decoder.eval()
    
    try:
        # 第一步：Encoder 编码
        with torch.no_grad():
            memory = encoder(src_ids)
            print(f"1. Encoder 完工，特征形状: {memory.shape}") # [1, 10, 32]
            
            # 第二步：Decoder 解码
            output = decoder(tgt_ids, memory)
            print(f"2. Decoder 完工，输出形状: {output.shape}") # [1, 5, 32]
            
        if output.shape == (1, 5, D_MOD):
            print("\n✅ 全链路形状匹配成功！")
        else:
            print(f"❌ 形状异常: {output.shape}")
            
    except Exception as e:
        print(f"❌ 运行崩溃，报错信息: {e}")

    # 验证 Decoder 的独立性（再次检查掩码）
    print("\n--- [终极验证] Decoder 跨序列干扰检查 ---")
    with torch.no_grad():
        # 修改 tgt_ids 的最后一个词
        tgt_ids_mod = tgt_ids.clone()
        tgt_ids_mod[0, -1] = (tgt_ids[0, -1] + 1) % V_SIZE
        
        out1 = decoder(tgt_ids, memory)
        out2 = decoder(tgt_ids_mod, memory)
        
        # 前 4 个词的输出不应受第 5 个词的影响
        diff = (out1[:, :-1, :] - out2[:, :-1, :]).abs().max().item()
        if diff < 1e-6:
            print(f"✅ 完美！Decoder 在多层嵌套下依然保持因果性 (Diff: {diff:.2e})")
        else:
            print(f"❌ 警告：多层堆叠导致因果泄露！Diff: {diff:.4f}")