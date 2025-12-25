> huggingface tokenizer api 地址：[Tokenizers](https://huggingface.co/docs/tokenizers/index)

tokenizer 的作用是语言模型的预备部分，针对输入文本进行切分，通常有以下几种方式：
1. 基于 **字** 的切分
2. 基于 **词** 的切分
3. 基于 **sub word** 的切分
4. 基于 **byte** 的切分
整体的切分流程包括：
> 文本归一化 -> 预切分 -> 基于分词模型的切分 -> 后处理

按 token 粒度分类：

| 级别                                         | 典型方法           | 说明               | 代表分词器                                    |
| ------------------------------------------ | -------------- | ---------------- | ---------------------------------------- |
| 基于字（Character-level）                       | 直接按字符切分        | 词表小、序列长、能覆盖任意词   | char-RNN tokenizer（早期）                   |
| 基于词（Word-level）                            | 预先定义词表、按词切     | 语义强、词表大、OOV问题严重  | Jieba / spaCy / BERT-word tokenizer      |
| Subword 级（Subword-level）                   | 统计合并/概率模型/语言模型 | 词表适中、序列适中、OOV鲁棒  | BPE / WordPiece / Unigram（SentencePiece） |
| 基于字节（Byte-level）                           | 直接按 byte 处理    | 语言无关、彻底无 OOV、最鲁棒 | GPT-2 byte-BPE / Tiktoken                |
| 基于 Unicode 码点（Codepoint-level，可视作 char 变体） | 按 Unicode 码点切分 | 处理 Emoji/复杂字符更合理 | 现代 char tokenizer 可能使用                   |
| 基于字节对但不合并 subword（Raw byte-level）          | 直接 0-255 作为词表  | 彻底语言无关，但语义极弱     | 罕用，仅理论存在                                 |

# 切分流程

tokenizer 包括训练和推理两个阶段：
- 训练：从语料库学习如何切分
- 推理：给一个句子切割成不同的 token

## 归一化

这是最基础的文本清洗，包括删除多余的换行和空格，转小写，移除音调等。例如：

```text
input: Héllò hôw are ü?
normalization: hello how are u?
```

## 预分词

预分词阶段会把句子切分成更小的“词”单元。可以基于空格或者标点进行切分。 不同的tokenizer的实现细节不一样。例如:

```text
input: Hello, how are you?

pre-tokenize:
[BERT]: [('Hello', (0, 5)), (',', (5, 6)), ('how', (7, 10)), ('are', (11, 14)), ('you', (16, 19)), ('?', (19, 20))]
[GPT2]: [('Hello', (0, 5)), (',', (5, 6)), ('Ġhow', (6, 10)), ('Ġare', (10, 14)), ('Ġ', (14, 15)), ('Ġyou', (15, 19)), ('?', (19, 20))]
[t5]: [('▁Hello,', (0, 6)), ('▁how', (7, 10)), ('▁are', (11, 14)), ('▁you?', (16, 20))]
```

可以看到BERT的tokenizer就是直接基于空格和标点进行切分。 GPT2也是基于空格和标签，但是空格会保留成特殊字符“Ġ”。 T5则只基于空格进行切分，标点不会切分。并且空格会保留成特殊字符"▁"，并且句子开头也会添加特殊字符"▁"。

## 基于分词模型的切分

这里指的就是不同分词模型具体的切分方式。分词模型包括：BPE，WordPiece 和 Unigram 三种分词模型。

## 后处理

后处理阶段会包括一些特殊的分词逻辑，例如添加Special token：[CLS], [SEP]等。

# 分词方式
## 基于字、词的分词

|指标|基于字的切分（Character-level）|基于词的切分（Word-level）|
|---|---|---|
|适用语言|适用于无明显词界限的语言，如中文、日文等|适用于有明显空格分隔的语言，如英语、法语等|
|OOV问题|不存在OOV问题|存在OOV问题|
|计算效率|计算效率较低，token 数量多|计算效率较高，token 数量少|
|语义信息|信息较为[细粒度](https://zhida.zhihu.com/search?content_id=260527376&content_type=Article&match_order=1&q=%E7%BB%86%E7%B2%92%E5%BA%A6&zhida_source=entity)，但较为局限|信息较为抽象，能较好保留单词语义|
|对形态变化的适应性|适应性强，能够处理复合词或变化形态|需要词形[预处理](https://zhida.zhihu.com/search?content_id=260527376&content_type=Article&match_order=1&q=%E9%A2%84%E5%A4%84%E7%90%86&zhida_source=entity)，难以处理形态变化丰富的语言|
|应用场景|适用于形态丰富、多语言混合的情况|适用于语言之间有明显词界限的任务|
>**OOV**: 在训练模型时，模型只会处理词汇表中的单词，对于词汇表之外的单词赋值为特殊的符号（UNK）

显而易见的就是根据字词来分，明显模型学不会一些组合逻辑（比如说 embeddings 和 embed 他们是有关系的，但是基于字、词的分词方式来看他们是割裂的）

## 基于 subword 的分词

Subword的基本切分原则是：
- 高频词依旧切分成完整的整词
- 低频词被切分成有意义的子词，例如 dogs => [dog, ##s]

基于Subword的切分可以实现：
- 词表规模适中，解码效率较高
- 不存在UNK，信息不丢失
- 能学习到词缀之间的关系

基于Subword的切分包括：BPE，WordPiece 和 Unigram 三种分词模型。

### BPE

词表构建流程（训练流程）
1. 初始化：将语料按字符（或更小的初始 units）拆开，作为初始序列。
2. 统计频率：统计所有相邻 token pair 的出现次数。
3. 合并：选频率最高的 pair (A, B) 合并为新 token `AB`，加入词表。
4. 更新语料表示：把语料中所有 `A B` 替换为 `AB`。
5. 迭代：重复 2–4，直到达到预设 vocab size 或合并次数上限。

分词（推理）流程
- 采用最长匹配（max-merge priority）：在已有 merges 规则下，把输入文本不断按可能合并的最长 subword替换，直到无法继续合并。

复杂度
- 训练：`O(#merges × 语料长度)`，工程实现常用**优先队列 + 频率哈希**加速。
- 推理：`O(n)` 级 greedy merge，n=输入字符数。

### BBPE
BPE 的变体，但是更细粒度的分割，输入空间是 raw bytes(0-255)

构建流程：
1. 初始化：把语料转成 byte 序列，每个 byte 作为初始 token（词表大小 256）。
2. 统计 & 合并：与 BPE 完全相同的逻辑，统计相邻 byte pair 频率并合并。
3. 加入新 token：合并后的 token 仍然是byte subsequence（如 `0xE4 0xB8 → 0xE4B8` 这样的组合）。
4. 迭代至目标 vocab size（GPT-2 约 50k）。

分词流程
- 先把输入文本UTF-8 编码成 bytes，然后在 byte merges 规则上做最长可合并匹配。
- 彻底无 OOV（任何语言/符号/乱码都有 UTF-8 byte 兜底）。

复杂度
- 训练/推理与 BPE 一致，只是初始 tokens=256 bytes。

### Unigram

>Unigram分词与BPE和WordPiece不同，是基于一个大词表逐步裁剪成一个小词表。 通过Unigram语言模型计算删除不同subword造成的损失来衡量subword的重要性，保留重要性较高的子词。

定义：把 tokenization 视作概率语言模型的最优路径选择问题，而不是贪心合并。

词表构建流程
1. 生成候选词表：从语料中提取大量可能的 subwords（通常用频率或字符 n-gram生成，规模 >> 目标 vocab）。
2. 训练 Unigram LM：把语料看成 token 序列，训练每个 token 的出现概率（极大似然估计）。
3. 剪枝：
    - 计算每个 token 对语料 likelihood 的贡献。
    - 删除贡献最小的 token（按比例或固定数量）。
4. 迭代 2–3，直到词表缩减到目标 vocab size。

分词（推理）流程
- 使用 Viterbi 或 N-best DP：
    1. 输入文本（字符序列）→ DP 计算所有可能 subword 组合的概率。
    2. 选全局概率最高的切分路径。
- 也支持N-best 采样（用于生成多样性切分，但推理常用 1-best）。

复杂度
- 训练：`O(候选词表 × 语料长度)` 级迭代剪枝。
- 推理：`O(n × k)` DP，n=输入长度，k=候选 subwords 平均匹配分支数。

### WordPiece
与 BPE 类似，只是在合并 pair 的过程中的策略不是出现的频率而是出现的 互信息

词表构建流程
1. 初始化：把语料按字符切分，并保留完整词（用于计算概率增益）。
2. 统计 pair 增益：对每个相邻 pair `(A, B)` 计算：
    **$$score=freq(AB)/(freq(A)×freq(B))score = freq(AB) / (freq(A) × freq(B))score=freq(AB)/(freq(A)×freq(B))$$**
    近似 PMI 或合并带来的语言模型 likelihood 增益。
3. 选择最高 score 的 pair 进行合并，加入词表。
4. 更新语料表示，重复直到达到目标 vocab size（BERT-base 30,522）。
    

 分词（推理）流程
- 采用基于词表的最长匹配：
    1. 先按空格切成 words
    2. 每个 word 内部尝试从左到右做最长 subword 词表匹配
    3. 匹配不到则拆成字符或 `[UNK]`（因此存在 OOV）

复杂度
- 训练：`O(#merges × 语料长度)`
- 推理：`O(n)` greedy longest-match，n=word 内字符数

# 评价指标

对于 tokenizer 来说，我们当然需要一些指标进行量化

1. 词表与编码能力：
	1.  $$CR = \frac{原始字符}{Token数}$$ 越大说明每个 Token 承载的信息越多
 
	2.  $$UNK\%=\frac{无法编码的Token}{总 Token}$$ 越小越好
2. 语义与建模质量
	1. PPL：同一个模型使用不同 tokenizer 产生的 token 序列，越低越好
3. 工程效率
	1. Token序列长度（影响后续 $O(n^2)$ 的Attention计算效率）
	2. tokenizer 编码速度、内存占用
