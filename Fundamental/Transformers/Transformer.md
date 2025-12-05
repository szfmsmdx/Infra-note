# 背景知识
## 统计机器翻译（SMT）
### 噪声信道模型
原始的噪声信道模型假定：
- 源语言（被翻译）中的信宿 $f$ , 是由目标句子（经过翻译后的句子）的信源 $e$ 经过含噪声的信道编码后得到的
- 现在的过程是，在已知 $f$ 的情况下，我们想将信宿翻译回信源，想知道他是那个 $e$  发出的，那么自然，概率为 $p(e|f)$ 
	- 但是直接构建 $p(e|f)$ 在当时（1990s）无法实现，这是一个句子到句子级别的建模
	- 此外 $p(e|f)$ 建模只能利用双语语料，数据比较稀少
- 利用贝叶斯可以将原概率转化为 $p(e|f)=\frac{p(f|e)p(e)}{p(f)}$ ,其中：
	- $p(e|f)$ 是我们想求的（译文给定 f 推断 e）
	- $p(f|e)$ 更容易建：如果原文是 e，翻译成 f 的概率
	- $p(e)$ 是语言模型：句子 e 自然出现的概率
	- $p(f)$ 是常数，因为“输入的 f 已经给定”
- 利用贝叶斯清洗完的优势：
	1. $p(f|e)$ 好训练，利用一些短语就可以训练
	2. $p(e)$ 是判断一个句子是否能自然出现（判断是否是自然语言），n-gram好判断
- 于是：
	- $p(f|e)$ 称之为翻译模型
	- $p(e)$ 称之为语言模型


### Seq2Seq(Encoder-Decoder) 

> 参考[这篇文章](https://zhuanlan.zhihu.com/p/194308943)

更详细的RNN还是参考这里的笔记 [[RNN]] 

主要的想法还是利用两个RNN，一个构建语义，一个从语义开始翻译，复杂一点的Seq2Seq包括Teach Forcing、Attention 和 Beam Search
- Teach Forcing：在训练时以一定概率使用目标输出来进行训练
- Attention：见下文
- Beam Search（集束搜索）：参考[这篇文章](https://zhuanlan.zhihu.com/p/114669778) ，注意 Beam_search并不是将搜索树的每个节点都展开为beam_size个子节点，而是维护搜索树的每层节点个数为 beam_size个，那么自然搜索空间应该是 $B\times T\times V$ 
	- $Total=V+(T−1)⋅B⋅V=V⋅[1+(T−1)B]$ ，搜索空间应该看的是搜索过的节点个数而不是看最终的剩余

# Attention
Attention最早提出追溯到[这篇论文](https://arxiv.org/pdf/1409.0473) 
本文笔记参考于这篇文章[使用详细的例子来理解RNN中的注意力机制-CSDN博客](https://blog.csdn.net/u011984148/article/details/99439993) 
![[Attention in RNN.png]]
博客中详细叙述了工作流程，注意的点是：
- 其实这里相当于网络内部的隐藏状态 $s_i$ 在每个时间步动态去查询对应的上下文状态
- FC 层是所有时间步、所有上下文共用的投影层

## History
### Self-Attention


### Hard Attention

### Global vs Local Attention

## Transformers
Attention被发扬光大实际上是著名的 [Attention is All You Need]([Attention is All you Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)) 这篇论文的影响
Transformer 是目前第一个只使用自注意力机制的模型


### Scaled Dot-Product Attention

### Multi-Head Attention

### Masking

## Position Embedding
- 绝对位置编码：
    - Sinusoidal (原始论文)。
    - Learnable Embedding (BERT, GPT-2)。
- 相对位置编码：
    - T5 的相对位置偏置。
    - RoPE (Rotary Positional Embedding)：目前最主流 (Llama)。通过复数旋转矩阵注入位置信息，具有极好的外推性。
    - ALiBi：不加 Pos Embedding，直接在 Attention Score 上加惩罚，外推性极强。