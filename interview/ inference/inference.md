# 为什么 prefill 不太用 cuda graph？
首先说一下 prefill 和 decode 阶段的区别，在常用的推理框架比如 vllm、sglang 都支持 continuous batching，那么：
- prefill 的输入序列长度应该是所有请求的长度和
- decode 阶段输入长度就是请求个数
那么，prefill 阶段的长度变化范围是更大的

其次，cuda graph 就是针对 tensor shape 一致（shape不一致会导致地址变化）、seq len变长的过程，且需要所有的算子支持CUDA Graph Capture  
所以 cuda graph 是有成本的，他启动会占用一些显存资源，然后存 graph 也是有成本，这个成本是 O(n) 随着序列长度递增的

那么分别来看，对于 decode 来说，他的变化范围有限，prefill 变化长度太宽，开了 cuda graph的收益并不明显  
举个例子比如 prefill 是超长文本 100K+，那么 cuda graph 要开到 100K+，但这样的超长序列一方面开销太大，另一方面为了少量请求单开收益太低

## prefill 能不能用？
其实也是可以用的，一种可行的方案是分段构造 cuda graph，比如我们对 FFN 采取 cuda graph，Attention 保持 eager  
现有的框架 vllm 已经支持 prefill 阶段的 cuda graph 了，采用的是 PIECEWISE。其运行逻辑是：在Attention或其他不支持CUDA Graph的操作中保持eager模式，而其余所有操作则被纳入CUDA Graph中执行。

# chunk prefill 有什么弊端？
1. 反复读入 kvcache
2. 可能同时受限计算强度和访存
3. kernel 启动增大 ttft

# KV cache减少显存占用的方法
1. 稀疏化和剪枝：通过识别并保留重要 token
    这里主要的思路是根据 Attention sink现象，在一些语义不相关的 token Attention计算的时候，经常会把 Attention score分配给头部靠前的几个 token，那么画热力图会看到有一条线的现象就是 Attention sink，根据 Attention sink有了以下改进
    - H2O：根据注意力累计得分动态剔除贡献小的 token，保留重要的 H2 token
    - streamingLLM，保留最开始的几个token和维护最近的 L 大小的窗口，维持你长文本对话的能力
2. 量化：把 kv cache 量化到 int4、int8 大小
3. 架构和算子优化
   - PageAttention
   - GQA、MLA
    等都是这个思路

# 为什么 gemm、transpose 这些算子需要 shm？
1. 从内存效率看：GEMM、transpose这些算子有内存复用的空间，以 GEMM 举例，你做 AxB 的时候，行列实际上会多次参与运算，每次从 HBM 读会占满带宽，cuda Core要等数据，所以采用 tiling，将一小块先搬到 shm，实现空间复用
2. 从合并访存来看：以 transpose 为例，转置的操作应该是连续读一行，然后写一列，但是写一列的过程是不连续的，所以你可以通过先从 HBM 里面读一块内容到 shm 里面，然后在 shm 里操作完再写回

# 端侧大模型有哪些限制？哪里优化？
先说一下限制，主要有几点：
1. 显存带宽：移动端的统一内存带宽是远比数据中心 GPU 低的
2. 算力功耗：持续高频输出可能会导致温度高、降频等问题
3. 内存容量：移动端系统可能和其他 APP 贡献统一内存，那么对 KV cache来说是有竞争关系

优化：
1. KV cache管理，比如 PageAttention 等，尽量打满不浪费
2. 量化压缩内存、提高带宽效率
3. 算子融合和内存复用：比如 rms norm和后续的FFN可以融合，否则你要先写，写完了在读在计算

# 实际过程中为什么 GPU 调度比 CPU 更加不平衡？

# 以 EP 为背景 prefill 和 decode 各自应该选用 EP16 还是 EP8，原因呢？

# 如何根据 prefill 和 decode 确定最大的 batch size？

# 业务上如果 ttft 没有满足，那么应该怎么解决？
## 哪怕增加了 P 节点也没有用怎么办

# prompt 分多个 chunk，顺序有要求吗？

# 算子调优思路？