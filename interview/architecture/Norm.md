### 1. Pre Norm 和 Post Norm 区别？
看一下表达式：
- Pre norm： $x_{t+1}=x_t+F_t(Norm(x_t))$ 
- Post norm： $x_{t+1}=Norm(x_t+F_t(x_t))$ 
区别在于——残差链接的信息，pre norm 始终保留了最原始的输入，导致模型其实是越叠越宽的（参考：[(25 封私信 / 76 条消息) Pre-norm和Post-norm的一些个人理解 - 知乎](https://zhuanlan.zhihu.com/p/1992629405449733321?share_code=hOoN5WB1RHsN&utm_psn=1993754586033436239)），而post norm有一个作用是：每次 post norm就削弱一次恒等分支的权重（每次将sublayer（残差）和主干分支拉回到同一范围内），而 pre norm相当于 $x_0+F_1(Norm(x_0))+F_2(Norm(x_0))+\cdots$ ，这相当于让模型变宽，对于所谓越深的模型，会削弱 sublayer 的贡献

### 2. 几种 Norm 对比
参考 [[Transformer]]

### 3. 