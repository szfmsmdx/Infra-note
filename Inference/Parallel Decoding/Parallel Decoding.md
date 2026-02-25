# 并行解码
首先，并行解码是什么？

传统的 LLM 生成是自回归的，生成第 n 个词必须由 n - 1 个词出来，并行解码的思路是打破每次只写一个词的限制，通常分为两类：
- 投机采样：Draft model先盲猜几个词，再由大模型进行校验
- 非自回归生成：修改模型结构，让模型本身具备能够输出多个位置的能力

## 常用方法

| 方案分类 | 代表模型/技术 | 技术特点 |
| :--- | :--- | :--- |
| 投机采样<br>(Speculative Sampling) | DeepSeek-V3, SpecInfer | 使用辅助小模型预测，主模型验证，无损。 |
| 多头并行<br>(Multi-Head Parallelism) | Medusa, Eagle, Hydra | 在主模型顶层加多个“Medusa Heads”，每个头预测未来不同步长的词。 |
| Prompt Lookup<br>(提示查找解码) | Prompt Lookup Decoding | 不用小模型，直接从 Prompt 的历史片段中找重复模式进行匹配。 |
| 静止/早期停止<br>(Early Exiting) | LayerSkip, CALM | 预测信心足够时，跳过模型高层直接输出。 |

## Parallel Decoding
### Medusa
#### 动机
传统自回归每生成一个 token 需要一次完整的 forward
- 算力闲置（Memory-Bound）： 现代 GPU 算力极强，但 LLM 解码通常是“内存受限”的。生成一个 token 时，GPU 大部分时间在搬运模型权重，计算单元其实在“打瞌睡”。
- 串行瓶颈： 如果我们要生成 100 个 token，就必须串行走 100 次模型，这成了推理延迟的最大来源。

#### 贡献
Medusa 在冻结权重的 LLM 最后一个隐藏层之上，平行添加了 $k$ 个解冻的“Medusa Heads”（通常是简单的 MLP 层）。
- Base Head： 负责预测第 $t+1$ 个 token。
- Medusa Head 1： 负责预测第 $t+2$ 个 token。
- Medusa Head n： 负责预测第 $t+n+1$ 个 token。

##### 树状注意力机制（Tree-based Attention）
这是 Medusa 的精髓。既然模型给出了多个候选 token，我们就把这些候选组合成一棵“路径树”。
1. 并行验证： 模型在一次正向传播中，同时处理这棵树上的所有路径。
2. 因果掩码（Causal Mask）： 通过特殊的掩码设计，确保树中的每个节点只能看到它的前驱节点。
3. 单次确认： 即使模型预测了未来 5 个位置，如果实际验证发现只有前 3 个符合概率要求，则接受前 3 个，剩下的丢弃。

#### 不足
- **训练依赖：** Medusa Heads 需要通过微调（Fine-tuning）来学习如何预测未来。虽然作者提出了 **Medusa-1**（冻结骨干只训头）和 **Medusa-2**（联合训练），但这依然增加了部署前的工程量。
- **内存开销：** 树状注意力虽然快，但会增加 KV Cache 的占用。如果显存本就捉襟见肘，Medusa 可能会导致 Batch Size 下降。
- **Top-P/Top-K 采样难题：** 在树状验证中处理随机性采样（Sampling）比贪婪搜索（Greedy Search）要复杂得多，虽然 Medusa 引入了典型的接受方案，但在极高随机性需求下，加速比会严重退化。