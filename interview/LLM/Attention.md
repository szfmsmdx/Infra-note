# Attention 变种
基础知识参考 [[Transformer#几种 Attention 对比]]


# 稀疏 Attention
## MLA

## DSA

# 分布式 Attention

# 为什么现代LLM都采用 Decoder-only 架构？
1. 语言建模就是自回归的，根据前面的语料预测下一个 token，这和 decoder only Transformer 是一致的，从这个角度来看其实encoder结构更加适合分类任务
2. 现代LLM希望统一所有任务，一个model做翻译、问答、推理、总结各种任务，decoder-only是可以实现的，GPT 范式就是将任务统一成 prompt-continuation这个结构，但如果是 encoder-decoder结构的话，其实就需要task-specific formatting
3. scale up更简单，因为他的结构和任务说明了我们只需要 next token prediction即可，不需要标注、task specific dataset，所以data scale up会非常快

