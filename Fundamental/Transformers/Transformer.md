# Norm
## 几种basic norm 对比
> 参考文章：[# 彻底搞懂：Batch Norm, Layer Norm, Instance Norm & Group Norm](https://zhuanlan.zhihu.com/p/19150825562)

简单来说，什么 Norm 对应跨什么维度，具体可以看下面这张经典的图：
![[Norm.png]]
### Batch Norm
对于 Batch Norm，就是跨 batch，对于上面的图片，比如针对某个 channel，我们对这个 batch 中的所有样本的同一个通道（同一个 channel）的所有位置（所有 H \* W）进行归一化

### Layer Norm
同样对于 Layer Norm，这里就是对于这个样本的所有通道当做整体做一个 Norm

### Instance Norm
Instance 的粒度更小，在 Layer 的基础上，把每个 channel 也作为单独的整体，Norm一下

### Group Norm
Group Norm 和 Instance Norm 类似，把几个 通道作为一个  Group，一个 Group 作为一个整体 Norm

# 位置编码
> 参考文章：[位置编码](https://blog.csdn.net/qq_41897558/article/details/137297209)

对于 Transformer 模型来说，由于纯粹的 Attention 模块是无法捕捉输入顺序的，所以为了解决这个问题，一般有两个办法：
1. 将位置信息融入到输入中 —— 绝对位置编码
2. 想办法微调 Attention，让他有分辨不同位置 Token 的能力 —— 相对位置编码
## 绝对位置编码
一般来说绝对位置编码是直接加入到输入 x 中，也就是说对于输入的第 k 个向量 $x_k$ 加入位置 k 的编码向量 $p_k$ 得到最终输入 $x_k + p_k$ 

### 训练类编码
训练类其实就是把位置编码当做一个可训练的参数进行训练，对于 max_seq_len = 128, dim=256 的例子来说，初始化一个 128x256 的矩阵参与训练

但是这种方式的缺点就是他没有**外推性**（训练和预测时的输入长度不一样，train short、test long，导致模型泛化性下降）

### 正弦位置编码（三角函数类）
这也是 Attn ... Need 文章提出的：
$$
p_{k,2i} = \sin\left( \frac{k}{10000^{2i/d}} \right),\quad \\ 
p_{k,2i+1} = \cos\left( \frac{k}{10000^{2i/d}} \right)
$$

由于他存在显示的生成规律，所以具有一定的外推性，另外还有个规律，根据三角函数的合角、差角公式，例如：$\sin(\alpha+\beta)=\sin\alpha\cos\beta+\cos\alpha\sin\beta$  可以得到，偶数位（2i）位置的位置编码和前、后位置都存在一定的关联，奇数位（2i+1）同理

### 递归式
顾名思义，是一种类似于 RNN 的位置编码，由于 RNN 的结构原因，天然能够学习到位置信息，那么我们是否可以考虑在输入后先接一层 RNN 学习这种位置信息然后再接入Transformer

FLOATER类方案，例如文章 [Learning to Encode Position for Transformer with Continuous Dynamical Model](https://arxiv.org/abs/2003.09229) 基于微分方程（ODE）的方式来建模位置编码，FLOATER 也属于地柜模型，所以这种基于神经网络的微分方程也称为神经微分方程

这种递归式编码具有比较好的外推性

## 相对位置编码
相对位置编码考虑的是算 Attention 的过程中考虑当前位置和被 Attention 位置的**相对距离**，下面是几种常见的相对位置编码

### 经典式
根据 Google 的文章：[Self-Attention with Relative Position Representations](https://arxiv.org/abs/1803.02155) 
一般认为，相对位置编码是由绝对位置编码启发而来。考虑一般的带绝对位置编码的 Attention： 
$$ \begin{cases} \mathbf{q}_i = (\mathbf{x}_i + \mathbf{p}_i) \mathbf{W}_Q \\ \mathbf{k}_j = (\mathbf{x}_j + \mathbf{p}_j) \mathbf{W}_K \\ \mathbf{v}_j = (\mathbf{x}_j + \mathbf{p}_j) \mathbf{W}_V \\ a_{i,j} = \operatorname{softmax}\left( \mathbf{q}_i \mathbf{k}_j^\top \right) \\ \mathbf{o}_i = \sum_j a_{i,j} \mathbf{v}_j \end{cases} $$

其中 $\operatorname{softmax}$ 对 $j$ 那一维归一化，所有向量均为行向量。 我们初步展开 $\mathbf{q}_i \mathbf{k}_j^\top$： 
$$ \begin{aligned} \mathbf{q}_i \mathbf{k}_j^\top &= (\mathbf{x}_i + \mathbf{p}_i) \mathbf{W}_Q \mathbf{W}_K^\top (\mathbf{x}_j + \mathbf{p}_j)^\top \\ &= (\mathbf{x}_i \mathbf{W}_Q + \mathbf{p}_i \mathbf{W}_Q)(\mathbf{W}_K^\top \mathbf{x}_j^\top + \mathbf{W}_K^\top \mathbf{p}_j^\top) \end{aligned} $$ 为了引入相对位置信息，Google 的做法是： 
- 去掉查询（query）中的位置项 $\mathbf{p}_i \mathbf{W}_Q$， （消去绝对位置编码）
- 将键（key）中的 $\mathbf{p}_j \mathbf{W}_K$ 替换为一个**二元相对位置向量** $\mathbf{R}_{i,j}^K$， 从而得到新的注意力得分： $$ a_{i,j} = \operatorname{softmax}\left( \mathbf{x}_i \mathbf{W}_Q \left( \mathbf{x}_j \mathbf{W}_K + \mathbf{R}_{i,j}^K \right)^\top \right) $$ 同时，在输出计算中，也将值（value）中的 $\mathbf{p}_j \mathbf{W}_V$ 替换为另一个相对位置向量 $\mathbf{R}_{i,j}^V$： $$ \mathbf{o}_i = \sum_j a_{i,j} \left( \mathbf{x}_j \mathbf{W}_V + \mathbf{R}_{i,j}^V \right) $$ 所谓**相对位置编码**，是指将原本依赖于绝对坐标对 $(i, j)$ 的向量 $\mathbf{R}_{i,j}^K$ 和 $\mathbf{R}_{i,j}^V$，改为**仅依赖于相对距离 $i - j$**。通常还会对相对距离进行截断，以适应任意长度的序列： $$ \begin{aligned} \mathbf{R}_{i,j}^K &= \mathbf{p}^K \left[ \operatorname{clip}(i - j,\, p_{\min},\, p_{\max}) \right] \\ \mathbf{R}_{i,j}^V &= \mathbf{p}^V \left[ \operatorname{clip}(i - j,\, p_{\min},\, p_{\max}) \right] \end{aligned} $$ 其中 $\operatorname{clip}(d, p_{\min}, p_{\max}) = \min(\max(d, p_{\min}), p_{\max})$。 这样一来，只需预定义或学习有限个位置嵌入（对应从 $p_{\min}$ 到 $p_{\max}$ 的相对距离），即可处理任意长度的输入序列。无论 $\mathbf{p}^K$、$\mathbf{p}^V$ 采用可学习参数还是基于三角函数的固定编码（如正弦位置编码），都能有效建模相对位置关系。

### XLNET

对于上述注意力得分的展开：

$$
\mathbf{q}_i \mathbf{k}_j^\top = 
\mathbf{x}_i \mathbf{W}_Q \mathbf{W}_K^\top \mathbf{x}_j^\top +
\mathbf{x}_i \mathbf{W}_Q \mathbf{W}_K^\top \mathbf{p}_j^\top +
\mathbf{p}_i \mathbf{W}_Q \mathbf{W}_K^\top \mathbf{x}_j^\top +
\mathbf{p}_i \mathbf{W}_Q \mathbf{W}_K^\top \mathbf{p}_j^\top
$$

XLNET 对相对位置 $p_j$ 直接替换为相对位置向量 $R_{i-j}$ ，对于自身的位置编码 $p_i$ 替换为两个不同的可训练参数，此时注意力得分改进为

$$
\mathbf{x}_i \mathbf{W}_Q \mathbf{W}_K^\top \mathbf{x}_j^\top +
\mathbf{x}_i \mathbf{W}_Q \mathbf{W}_K^\top \mathbf{R}_{i-j}^\top +
\mathbf{u} \mathbf{W}_Q \mathbf{W}_K^\top \mathbf{x}_j^\top +
\mathbf{v} \mathbf{W}_Q \mathbf{W}_K^\top \mathbf{R}_{i-j}^\top
$$

进一步改进：
- 由于相对位置编码 $\mathbf{R}_{i-j}$ 的表示空间可能与 token embedding 不同，**为其分配独立的投影矩阵** $\mathbf{W}_{K,R}$，替代原来的 $\mathbf{W}_K$；
- 向量 $\mathbf{u} \mathbf{W}_Q$ 和 $\mathbf{v} \mathbf{W}_Q$ 可直接吸收到 $\mathbf{u}$、$\mathbf{v}$ 中（即令 $\mathbf{u} \leftarrow \mathbf{u} \mathbf{W}_Q$，$\mathbf{v} \leftarrow \mathbf{v} \mathbf{W}_Q$），因此最终形式为：

$$
\mathbf{x}_i \mathbf{W}_Q \mathbf{W}_K^\top \mathbf{x}_j^\top +
\mathbf{x}_i \mathbf{W}_Q \mathbf{W}_{K,R}^\top \mathbf{R}_{i-j}^\top +
\mathbf{u} \mathbf{W}_K^\top \mathbf{x}_j^\top +
\mathbf{v} \mathbf{W}_{K,R}^\top \mathbf{R}_{i-j}^\top
$$

此外，**Transformer-XL 在 value 向量中完全去掉了位置偏置**，即直接使用：

$$
\mathbf{o}_i = \sum_j a_{i,j} \, \mathbf{x}_j \mathbf{W}_V
$$
自此开始位置编码就只存在于 Attention 中，而不加到 V 上了

### T5

[T5 模型文章](https://arxiv.org/abs/1910.10683) 
在文章中的相对位置编码则更为简单，对于注意力得分的展开式，我们可以观察得到其实这是四组注意力结合：
1. 输入 - 输入
2. 输入 - 位置
3. 位置 - 输入
4. 位置 - 位置
T5 认为输入和位置应该是解耦的，所以 2、3 两项不应该存在，而“位置 - 位置”的组合 $P_jW_QW_K^\top p_j^\top$  是一个只依赖于 (i,j) 的标量，所以我们可以将他作为参数训练出来，简化为：
$$
x_i W_Q W_K^\top x_j^\top + \beta_{i,j}
$$
值得注意的是， T5 做了一个转换（分桶处理），相对位置是 i-j, 实际上用的是 f(i - j) ，映射关系如下：

|  $i-j$   |  0  |  1  | ... |  8  |  9  | ... |
| :------: | :-: | :-: | :-: | :-: | :-: | :-: |
| $f(i-j)$ |  0  |  1  | ... |  8  |  8  | ... |

简单来说，就是做了一个归纳，靠近位置分配粒度比较细致，稍远的位置分配的粒度更粗

## RoPE 旋转位置编码

> 参考文章：[十分钟读懂旋转编码](https://www.zhihu.com/tardis/zm/art/647109286?source_id=1003)

