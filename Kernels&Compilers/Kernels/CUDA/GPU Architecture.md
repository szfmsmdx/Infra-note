> 参考文章：
> [(26 封私信 / 80 条消息) 【CUDA 入门必知必会】GPU 体系结构与编程模型 - 知乎](https://zhuanlan.zhihu.com/p/1991985685964560041)


# 架构历史简表

|架构代号|发布年份|工艺制程|典型产品|核心创新|
|---|---|---|---|---|
|Tesla|2006–2008|90–65nm|G80 (GeForce 8800)|首个统一着色器架构，开启通用 GPU 计算（GPGPU）|
|Fermi|2010|40nm|GTX 480, Tesla C2050|引入 L1/L2 缓存、ECC 内存，强化 HPC 支持|
|Kepler|2012|28nm|GTX 680, K80|能效优化，SMX 流式多处理器，动态并行|
|Maxwell|2014|28nm → 20nm|GTX 980|超高能效比，SMM 设计，大幅降低功耗|
|Pascal|2016|16nm FinFET|GTX 1080, P100|首推 NVLink、HBM2（P100），支持半精度（FP16）|
|Volta|2017|12nm|V100|首代 Tensor Core，专为 AI 加速设计|
|Turing|2018|12nm|RTX 2080, T4|RT Core（光追） + Tensor Core，开启实时光线追踪|
|Ampere|2020|7nm (数据中心) / 8nm (消费)|A100, RTX 3090|第二代 RT/Tensor Core，支持稀疏训练，FP64 增强|
|Hopper|2022|TSMC 4N (4nm)|H100, H200|Transformer Engine，DPX 指令，NVLink 4.0，万亿级模型支持|
|Blackwell|2024|TSMC 4NP / 3nm|B100, B200, RTX 5090|双芯片设计、第五代 Tensor Core、FP4/FP6、TMEM、解耦并行、RAS 引擎|
|Rubin（即将）|2026（预计）|2nm 或更先进|Rubin GPU + Vera CPU|继 Blackwell 之后的新一代，集成自研 ARM CPU（Olympus 架构）|
