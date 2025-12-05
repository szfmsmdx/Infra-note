虽然 Transformer 是主流，但理解 RNN 能让你深刻理解“序列依赖”和“状态空间”，且最新的 **Mamba/RWKV** 架构本质上是 RNN 的文艺复兴。

#### 1. 基础架构与数学

- **Vanilla RNN**：$h_t = \tanh(W h_{t-1} + U x_t)$。
- **BPTT (Backpropagation Through Time)**：随时间反向传播的推导，理解梯度如何在时间步中流动。
- **梯度消失/爆炸 (Vanishing/Exploding Gradient)**：
    - _数学本质_：连乘效应（雅可比矩阵的特征值是否大于 1）。
    - _解决方法_：Gradient Clipping（截断）、正交初始化、Gating 机制。

#### 2. 门控机制 (Gating Mechanism)
- **LSTM (Long Short-Term Memory)**：
    - 遗忘门、输入门、输出门、**Cell State (核心)**。
    - 理解为什么加法更新（$C_t = f_t C_{t-1} + i_t \tilde{C}_t$）能构建“梯度高速公路”，解决梯度消失。
- **GRU (Gated Recurrent Unit)**：
    - 简化版 LSTM，合并了 Cell State 和 Hidden State，理解重置门和更新门的作用。
#### 3. 序列到序列 (Seq2Seq)
- **Encoder-Decoder 范式**：定长向量瓶颈（Context Vector Bottleneck）的产生原因。
- **Teacher Forcing**：训练时使用真实标签作为下一时刻输入 vs. 推理时使用预测值（Exposure Bias 问题）。
- **双向 RNN (Bi-RNN)**：利用未来信息（这也是 BERT Masked LM 的灵感来源）。

#### 4. RNN 的现代变体 (Linear RNN / SSM) - **前沿专家方向**
- **并行化难题**：为什么 RNN 训练慢（时序依赖，无法像 Transformer 那样并行）。
- **RWKV / Mamba**：如何结合 RNN 的推理效率（O(1) 显存）和 Transformer 的训练并行性。