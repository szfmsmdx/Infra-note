import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCharCNN(nn.Module):
    def __init__(self, num_chars, char_embed_dim, kernel_sizes, num_filters, output_dim):
        """
        Args:
            num_chars: 字符表大小 (比如 26个字母 + 标点 = 100)
            char_embed_dim: 每个字符的初始向量维度 (比如 16)
            kernel_sizes: 卷积核的宽度列表 (比如 [2, 3, 4] 代表看 bigram, trigram...)
            num_filters: 每个卷积核有多少个 (比如 50 个)
            output_dim: 最终输出的单词向量维度 (比如 128)
        """
        super().__init__()
        
        # 1. 字符嵌入层 (Lookup Table)
        # Input: [Batch, Word_Len] -> Output: [Batch, Word_Len, Char_Dim]
        self.char_embedding = nn.Embedding(num_chars, char_embed_dim)
        
        # 2. 卷积层列表 (ModuleList)
        # 我们需要并行的多个卷积核，分别处理不同长度的特征
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=char_embed_dim, # 输入通道 = 字符维度
                out_channels=num_filters,   # 输出通道 = 特征探测器的数量
                kernel_size=k               # 窗口宽度 (2, 3, or 4)
            ) for k in kernel_sizes
        ])
        
        # 3. 最终投影层 (把拼接后的特征压缩到指定维度)
        total_filters = num_filters * len(kernel_sizes)
        self.linear = nn.Linear(total_filters, output_dim)

    def forward(self, x):
        # x shape: [Batch, Max_Word_Len] (比如 batch=2, 单词最长=7)
        
        # --- Step 1: Character Embedding ---
        x = self.char_embedding(x) 
        # Shape: [Batch, Word_Len, Char_Dim]
        
        # --- Step 2: Transpose for Conv1d ---
        # PyTorch 的 Conv1d 要求输入是 [Batch, Channels, Length]
        # 所以要把 Char_Dim 换到中间去
        x = x.permute(0, 2, 1) 
        # Shape: [Batch, Char_Dim, Word_Len]
        
        # --- Step 3: Convolution + ReLU + MaxPooling ---
        conved_outputs = []
        for conv in self.convs:
            # A. 卷积: 滑动窗口提取特征
            # input: [Batch, Char_Dim, Word_Len]
            # output: [Batch, Num_Filters, Output_Len]
            conv_out = conv(x)
            
            # B. 激活: ReLU
            conv_out = F.relu(conv_out)
            
            # C. 池化: Max-Over-Time Pooling
            # 我们不管单词有多长，只取每个 Filter 响应最大的那个值
            # kernel_size=conv_out.shape[2] 意味着池化整个长度
            pooled = F.max_pool1d(conv_out, kernel_size=conv_out.shape[2])
            # Shape: [Batch, Num_Filters, 1]
            
            # 压缩掉最后那个 1 维度
            pooled = pooled.squeeze(2) 
            # Shape: [Batch, Num_Filters]
            
            conved_outputs.append(pooled)
            
        # --- Step 4: Concatenation (拼接) ---
        # 把不同宽度的卷积核提取的特征拼在一起
        # [Batch, Num_Filters * len(kernel_sizes)]
        final_feature = torch.cat(conved_outputs, dim=1)
        
        # --- Step 5: Projection (投影) ---
        # 映射到你想要的最终维度 (比如 Word2Vec 的维度)
        output = self.linear(final_feature)
        
        return output

# ================= 运行测试 =================
if __name__ == "__main__":
    # 1. 模拟配置
    BATCH_SIZE = 2
    MAX_WORD_LEN = 8  # 假设单词最长 8 个字符 (比如 "playing" 加 padding)
    NUM_CHARS = 30    # 假设只有小写字母 + pad
    CHAR_DIM = 16     # 字符向量维度
    KERNELS = [2, 3, 4] # 分别看 2-gram, 3-gram, 4-gram
    FILTERS = 5       # 每个核只有 5 个探测器 (为了打印方便看)
    OUTPUT_DIM = 10   # 最终输出单词向量维度
    
    model = SimpleCharCNN(NUM_CHARS, CHAR_DIM, KERNELS, FILTERS, OUTPUT_DIM)
    
    # 2. 模拟输入 (Batch=2)
    # word 1: [1, 2, 3, 4, 0, 0, 0, 0] (假设是 "abcd" + pad)
    # word 2: [5, 6, 7, 8, 9, 10, 0, 0] (假设是 "efghij" + pad)
    dummy_input = torch.randint(0, NUM_CHARS, (BATCH_SIZE, MAX_WORD_LEN))
    
    print(f"Input Shape: {dummy_input.shape}") # [2, 8]
    
    # 3. 前向传播
    word_vector = model(dummy_input)
    
    print("\n--- Model Architecture ---")
    print(model)
    
    print("\n--- Output ---")
    print(f"Output Shape: {word_vector.shape}") # 应该是 [2, 10]
    print("得到的单词向量 (前2个维度):")
    print(word_vector[:, :2])