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
- GPU启动的kernel是非抢占式的要等运行完，所以会出现一些负载倾斜的问题
- CPU控制权在OS kernel，内核有全局视野，但是GPU控制权在用户态框架上面，需要自己设计一些调度策略，以及GPU上下文切换的成本比较高，所以倾斜于少切换，最终导致长任务对资源垄断

# 以 EP 为背景 prefill 和 decode 各自应该选用 EP16 还是 EP8，原因呢？
- prefill应该是EP8，因为 prefill 是 compute-bound，少EP可以减少通信成本，降低TTFT
- decode那就采用 EP16 去减轻内存吞吐的压力，提升 TBT/TPOT

# 如何根据 prefill 和 decode 确定最大的 batch size？

# 业务上如果 ttft 没有满足，那么应该怎么解决？
先考虑是不是prefill算力不够，先增加P节点

## 哪怕增加了 P 节点也没有用怎么办
首先的话，那应该就不是算力的问题，可能的原因有：
- 被长序列给阻塞了，可以采取 chunk prefill 优化短 seq 的 TTFT，或者采取其他一些调度策略
- 通信问题，all2all 或者 nccl 阻塞
  - 可能是 EP 的某个节点卡了，那么可以试试减少 EP，减少通信成本

# prompt 分多个 chunk，顺序有要求吗？
会的，因为Attention计算是自回归的，需要依赖之前的信息，如果打散的话，那么可能会读到不应该被读到的token

# 算子调优思路？
- nsys：看 Kernel 在时间轴上的分布、启动延迟、CPU-GPU 同步点、PCIe 拷贝耗时。用来判断是不是“排队”或“启动”慢了
- ncu：看寄存器使用率、Shared Memory 冲突、Tensor Core 利用率、DRAM 吞吐量

主要关注指标：
- DRAM Throughput (GB/s)：实际显存带宽占用。
- Compute Throughput (TFLOPS)：实际算力占用（区分 FP16/INT8 Tensor Core）。
- Memory Throughput / Compute Throughput ：相对于硬件理论峰值的百分比。
- Stall Reasons：比如 memory_throttle (显存堵了), sync_dependency (等数据), dispatch_stall (没喂饱)。
- Occupancy (活跃线程束占比)：是否填满了 SM。

然后可以看 ncu 中的 roofline，看自己算子的位置：
- 处在左侧倾斜部分，那么是 mem-bound，要考虑 mem 层面的优化，比如 shm、算子融合、量化等操作，主要是提高访存效率，要么少读，要么读了多复用
- 处在右侧那么是 compute-bound，那么考虑 tensor-core 使用或者其他一些方法
- 如果里两个 wall 都很远的话，可能是 launch 开销比较大，也可能是里面一些分支判断比较多，可以考虑用 mask 什么的

# 说一下 FlashAttention？
参考：[[Transformer#FlashAttn]]
#### online softmax


首先说一下 FlashAttention 动机，对于 Attention 计算，我们要先算出 score matrix 再写到 HBM，然后做 softmax，读出来写回去，最后乘上 V，大量的 HBM 读写导致实际是 mem-bound
flashAttn 干的事是 tile 分块，在 SRAM / shm / register 里面做 online softmax 和局部累加，把中间结果尽量留到片上，最后写会必要的输出

所以他的技术点在于：
1. tile
2. online softmax
3. fuse，把QK、softmax，PV fuse为一个kernel

真正大头的 dense matmul 部分，尤其是 $QK^T$  和后面的 $P\times V$ ，如果数据类型和 tile 形状合适，通常优先走 Tensor Core；而 softmax、mask、scale、max/sum reduction、index 计算、边界处理、online normalization 这些标量或 reduction 逻辑，更多是在 CUDA core 上做。也就是说，Tensor Core 负责“高吞吐矩阵乘”，CUDA core 负责“控制流、规约、逐元素、搬运协同”

- v1提出
- v2优化并行策略，解决 v1 在长序列下的性能瓶颈
- v3支持Hopper架构和FP8格式
- v4适配Blackwell架构，适配FP4