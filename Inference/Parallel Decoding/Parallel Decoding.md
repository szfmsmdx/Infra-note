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

