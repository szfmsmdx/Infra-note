# LLM.int8()
首先，llm.int8() 是 W8A8 dynamic 量化策略，在计算的过程中执行了以下三步：
- 分离离群值
- 混合精度计算：
	- 离群部分——FP16 进行矩阵乘法
	- 非离群——量化为 int8 进行高效 vector-wise 矩阵乘
- FP16 部分和 int8 结果 dequant 整合成 fp16 精度的输出

LLM.int8() 方法的主要目的是在不降低性能的情况下降低大模型的应用门槛，使用了 LLM.int8() 的 BLOOM-176B 比 FP16 版本慢了大约 15% 到 23%，结果如下所示：

|精度|参数量|硬件|延迟 (ms/token，BS=1)|延迟 (ms/token，BS=8)|延迟 (ms/token，BS=32)|
|---|---|---|---|---|---|
|bf16|176B|8xA100 80GB|239|32|9.9|
|int8|176B|4xA100 80GB|282|37.5|10.2|
|bf16|176B|14xA100 40GB|285|36.5|10.4|
|int8|176B|5xA100 40GB|367|46.4|oom|
|fp16|11B|2xT4 15GB|11.7|1.7|0.5|
|int8|11B|1xT4 15GB|43.5|5.3|1.3|
|fp32|3B|2xT4 15GB|45|7.2|3.1|
|int8|3B|1xT4 15GB|312|39.1|10.2|

## 量化实践（BNB）
！`bitsandbytes` 是基于 CUDA 的主要用于支持 LLM.int8() 的库，它是`torch.nn.modules`的子类
1. 首先导入模块，初始化fp16 模型并保存

```text
import torch
import torch.nn as nn
from bitsandbytes.nn import Linear8bitLt

fp16_model = nn.Sequential(
    nn.Linear(64, 64),
    nn.Linear(64, 64)
).to(torch.float16).to(0)
torch.save(fp16_model.state_dict(), "model.pt")
```

2. 初始化 int8 模型并加载保存的weight，此处标志变量`has_fp16_weights`非常重要。默认情况下，它设置为`True`，用于在训练时使能 Int8/FP16 混合精度。

```text
int8_model = nn.Sequential(
    Linear8bitLt(64, 64, has_fp16_weights=False),
    Linear8bitLt(64, 64, has_fp16_weights=False)
)

int8_model.load_state_dict(torch.load("model.pt"))
```

此时还未进行量化操作，可以查看weight值

```text
int8_model[0].weight
Parameter containing:
Parameter(Int8Params([[ 0.1032,  0.0544,  0.1021,  ..., -0.0465,  0.1050,  0.0687],
            [ 0.0083,  0.0352,  0.0540,  ..., -0.0931, -0.0224,  0.0541],
            [ 0.0476,  0.0220, -0.0803,  ...,  0.1031,  0.1134,  0.0905],
            ...,
            [-0.0523, -0.0858,  0.0330,  ...,  0.1122, -0.1082,  0.1210],
            [ 0.0045, -0.1019,  0.0072,  ..., -0.1069, -0.0417,  0.0365],
            [-0.1134,  0.0032, -0.0742,  ..., -0.1142, -0.0374,  0.0915]]))
```

之后将模型加载到GPU上，此时发生量化操作：

```text
int8_model = int8_model.to(0) # Quantization happens here
```

此时可以查看weight值，可以看到值已经传到GPU上并转为 INT8 类型。

```text
int8_model[0].weight
Parameter containing:
Parameter(Int8Params([[ 105,   55,  104,  ...,  -47,  107,   70],
            [   9,   36,   56,  ...,  -96,  -23,   56],
            [  49,   23,  -82,  ...,  105,  116,   92],
            ...,
            [ -54,  -89,   34,  ...,  116, -112,  126],
            [   5, -107,    8,  ..., -112,  -44,   38],
            [-116,    3,  -76,  ..., -117,  -38,   94]], device='cuda:0',
           dtype=torch.int8))
```

获取 FP16 权重以便在 FP16 中执行离群值的矩阵乘，可见与原始值比较接近

```text
(int8_model[0].weight.CB * int8_model[0].weight.SCB) / 127
tensor([[ 0.1032,  0.0532,  0.1018,  ..., -0.0453,  0.1017,  0.0682],
        [ 0.0088,  0.0348,  0.0548,  ..., -0.0926, -0.0219,  0.0546],
        [ 0.0482,  0.0223, -0.0802,  ...,  0.1012,  0.1102,  0.0897],
        ...,
        [-0.0531, -0.0861,  0.0333,  ...,  0.1118, -0.1064,  0.1228],
        [ 0.0049, -0.1036,  0.0078,  ..., -0.1080, -0.0418,  0.0370],
        [-0.1140,  0.0029, -0.0744,  ..., -0.1128, -0.0361,  0.0916]],
       device='cuda:0')
```

最后比较 fp16 模型与 int8 模型输出结果

```text
input_ = torch.randn(8, 64, dtype=torch.float16)
hidden_states_int8 = int8_model(input_.to(0))
hidden_states_fp16 = fp16_model(input_.to(0))
print(torch.max(hidden_states_fp16 - hidden_states_int8))
```

可以最大的绝对误差为

```text
tensor(0.0098, device='cuda:0', dtype=torch.float16, grad_fn=<MaxBackward1>)
```

另外，我们也可以直接加载预训练完成的模型为 int8 类型，方式如下：

```text
from transformers import LlamaForCausalLM
model = LlamaForCausalLM.from_pretrained(args.base_model, 
                                         load_in_8bit=args.load_8bit,
                                         torch_dtype=torch.float16, 
                                         device_map={"auto"}, )
```
# Smooth Quant
主要思路是：模型规模增大时，token变化范围比weight变化范围更大，所以通过引入超参数 s，将部分激活量化难度转移给 weight，核心是解决**计算吞吐的问题**
所以他的 W4A8 的量化，dynamic 和 static 都支持的量化方法

对于激活的范围，文章指出同一 token，不同 channel 差异很大，不同 token，离群值基本都出现在那几个 channel，所以通过引入平滑因子 s 让 $Y=XW=X/s\cdot Ws$ 

为了减少激活的量化难度，其实可以 $s_j = max(|X_j|),j=1,\cdots,C_j$ ，也就是每个 channel 对应的最大值，但是这样会让weight比较难以量化，所以引入转移强度 α，使得 $s_j=max(|X_j|)^\alpha / max(|W_j|)^{1-\alpha}$ ，其中 α 可以手动调整，当 α 为 1 时实际上就是比较暴力的最大值，将难度全部转移给了 weight

得到smooth变换之后的 activation 和 weight 矩阵，可以再采用 per-token 或 per-tensor 的量化方式，又根据激活值是基于先验数据获得或者执行时获得分为动态量化和静态量化，作者提出了三个层次的量化方式，其中O1~O3的计算延迟依次减少。

|Method|Weight|Activation|
|---|---|---|
|W8A8|per-tensor|per-tensor dynamic|
|ZeroQuant|group-wise|per-token dynamic|
|LLM.int8|per-channel|per-token dynamic+FP16|
|Outlier Suppression|per-tensor|per-tensor static|
|SmoothQuant-O1|per-tensor|per-token dynamic|
|SmoothQuant-O2|per-tensor|per-tensor dynamic|
|SmoothQuant-O3|per-tensor|per-tensor static|

平滑因子S的计算也可以融合到前层 Layernorm/Softmax 中。

# AWQ
和 smooth quant 的思路比较类似：AWQ指出大模型的权重重要性不同，由输入激活分布控制，维护这 1% 的关键权重能更好地维护模型的能力

注意，AWQ 是weight only的W4A16 静态static 量化策略，主要还是解决 **memory wall** 的问题

首先AWQ说明1%权重需要被保护，那么如果重要通道采用 fp16 原始权重保存，那么 kernel 会比较难写，所以有没有办法优化这一块，这里采取了 smooth quant 的方法，做 scaling

scaling推导，参考文章：[(7 封私信) 深入理解AWQ量化技术 - 知乎](https://zhuanlan.zhihu.com/p/697761176)

和smooth的结论一致，当你的 s 过大的时候，其实是将激活的量化难度转移给 weight，所以我们当然希望能够找到最优的系数，由于 quant 不可导，所以引入 α 转化为空间搜索，后面和 smooth 就类似了

# GPTQ
>参考文章：[(7 封私信) 一文搞懂大模型量化技术：GGUF、GPTQ、AWQ - 知乎](https://zhuanlan.zhihu.com/p/1899107168172630461)

GPTQ算法核心来源于 OBQ(Optimal Brain Quantization)， 而OBQ的思路主要来自于OBS(Optimal Brain Surgeon)。

## OBS
在 OBS 中，假设我们抹去一个权重 $w_q$ ，那么我们应该填上一个补偿 $\delta_q$ 应用在剩余权重上使得误差被抵消，OBS 的做法是：$w_q = \text{argmin}_q \frac{w_q^2}{[H^{-1}]_{qq}}, \delta_q = -\frac{w_q}{[H_{qq}^{-1}]} \cdot H_{:,q}^{-1}$ 

## OBC
此时OBS进行剪枝的复杂度太高，求解 $H^{-1}$ 的复杂度为$O((d_{row}\cdot d_{col})^3)$ ，在加上逐元素就是 4 次的复杂度

OBC 做出了优化：
- 每行是独立的，且相关的黑塞矩阵都是 $H=2XX^T$ 
- 更新Hessian逆矩阵有更高效的算法

那么实际上 OBC 做剪枝就是一个 for k 的操作减去 k 个行权重，每次 for 的过程中也是贪心去找该行从小到大损失的一个权重元素

## OBQ
OBQ 将他推广到量化中，OBQ的公式是这样的：$w_q = \text{argmin}_q \frac{(\text{quant}(q) - w_q)^2}{[H^{-1}]_{qq}}, \delta_q = -\frac{w_q - \text{quant}(q)}{[H_{qq}^{-1}]} \cdot H_{:,q}^{-1}$ ，但 OBQ 的复杂度是比较高的$O(d_{row}*d_{col}^3)$ ，采取一个贪心的策略：
- 在所有未量化的权重中，寻找那个量化后产生误差最小的权重
- 量化该权重，并立即利用上述公式更新同一行中所有剩余的 FP16 权重，以补偿误差

总体而言，OBQ的贡献在于：
- 建立了基于二阶信息（Hessian）的量化补偿数学模型。
- 证明了通过微调未量化权重可以极大地补偿量化损失。
- 提供了 Weight-by-weight 量化的精度基准。

## GPTQ
GPTQ 在 OBQ 的基础上进一步优化：
- GPTQ提出顺序进行量化和贪心量化的效果是差不多的，那么实际上你就可以把贪心的串行策略优化为并行，且每行独立并共享黑塞矩阵，那么黑塞矩阵只需要被更新一次即可
- GPTQ 引入了 **Lazy Batch-Updates** 技术，该技术通过分块 (batching) 的方式减少内存访问次数，从而显著提高计算效率。

GPTQ 是 weight only 的 static quant方法

# llama.cpp系列
命名通常遵循 `Q比特_变体` 的格式：
- **Q**：代表 Quantization（量化）。
- **数字**：代表主要的比特数（如 4 代表 4-bit）。
- **变体 (0/1/K/S/M/L)**：
    - **0/1**：早期版本，表示是否带有额外的偏置或特定的对齐。
    - **K**：源自作者 k-quants 的优化，引入了更精细的分级量化。
    - **S/M/L**：Small / Medium / Large，表示在同一比特下，模型不同层（如 Attention vs MLP）保留精度的程度。

## legacy quant
Q8_0 (8-bit)
- **具体内容**：每 32 个权重为一个块（Block）。每个块存储一个 FP16 的缩放系数（Scale）和 32 个 Int8 的权重。
- **公式**：$w = q \cdot \text{scale}$
- **特点**：几乎无损，计算开销极小。常用于存下激活值（W8A8 的变体）以保证精度。

Q4_0 & Q4_1 (4-bit)
- **Q4_0**：每 32 个权重一组，存储一个 FP16 scale 和 32 个 4-bit 权重。对称量化。
- **Q4_1**：在 Q4_0 基础上增加了一个 FP16 的 **Minimum（最小值）**。
    - **公式**：$w = q \cdot \text{scale} + \text{min}$
    - **意义**：非对称量化。对于分布偏移的权重，精度比 Q4_0 好，但计算速度稍慢。

## K-Quants 策略 (现代主流)

这是目前 `llama.cpp` 的主力。其核心贡献是**混合比特（Mixed Precision）**：模型中对精度敏感的层（如 Attention 的 $V$ 矩阵）比特数高一点，不敏感的层（如 MLP 的下投影）比特数低一点。

### 命名含义 (以 Q4_K_M 为例)
- **Q4**：基础目标是 4-bit。
- **K**：使用了 K-Quants 的优化算法。
- **M (Medium)**：权重分配方案。
    - **S (Small)**：全部使用 4-bit。
    - **M (Medium)**：部分关键层使用 6-bit，其余 4-bit。
    - **L (Large)**：更多层使用 6-bit。

### 核心内容：分级缩放 (Super-Block)
K-Quants 引入了“超级块”概念（通常包含 256 个权重）：
1. 将 256 个权重分成 16 个小块（每块 16 个值）。
2. **二级缩放**：
    - 每个小块有自己的缩放因子。
    - 这些缩放因子本身又被量化（例如量化为 6-bit 或 4-bit）。
    - 这种分层机制极大地压制了离群值（Outliers）带来的误差。