import torch
from torch.nn.modules.transformer import Transformer
from math import sqrt, log

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

if __name__ == "__main__":
    torch.manual_seed(42)
    
    # 1. 模拟超参数
    VOCAB_SIZE = 100
    D_MODEL = 32
    N_LAYERS = 3
    N_HEADS = 4
    D_FF = 128
    BATCH = 2
    SEQ_LEN = 8

    print(f"--- 开始测试 Encoder (Layers={N_LAYERS}) ---")
    encoder = Encoder(N_LAYERS, VOCAB_SIZE, D_MODEL, N_HEADS, D_FF)

    # 2. 模拟输入数据 [B, L]
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
    
    # 3. 前向传播
    try:
        output = encoder(input_ids)
        print(f"✅ 前向传播成功！输出形状: {output.shape}") # 期望 [2, 8, 32]
        
        # 验证输出是否有 NaN (检查 Norm 和 Log 稳定性)
        if torch.isnan(output).any():
            print("❌ 警告：输出包含 NaN！")
        else:
            print("✅ 数值稳定性检查通过 (无 NaN)")
            
    except Exception as e:
        print(f"❌ 运行崩溃: {e}")

    # 4. 【关键测试】浅拷贝验证
    # 检查第一层和第二层的权重内存地址是否相同
    layer0_ptr = id(encoder.encode_layers[0].self_attn.q.weight)
    layer1_ptr = id(encoder.encode_layers[1].self_attn.q.weight)
    if layer0_ptr == layer1_ptr:
        print("❌ 严重错误：检测到层之间共享权重（浅拷贝 Bug）！")
    else:
        print("✅ 层独立性检查通过（各层参数独立）")

    # 5. 梯度回传测试 (验证计算图是否闭环)
    print("\n--- 梯度回传测试 ---")
    loss = output.mean()
    loss.backward()
    
    # 检查 Embedding 层是否有梯度
    if encoder.embedding.weight.grad is not None:
        print("✅ 梯度成功回传至 Embedding 层")
    else:
        print("❌ 错误：梯度丢失，请检查残差连接或 forward 逻辑")

    print("\n🎉 Encoder 阶段性测试完成！")