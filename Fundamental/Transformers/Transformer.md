# Norm
## 几种basic norm 对比
> 参考文章：[# 彻底搞懂：Batch Norm, Layer Norm, Instance Norm & Group Norm](https://zhuanlan.zhihu.com/p/19150825562)

简单来说，什么 Norm 对应跨什么维度，具体可以看下面这张经典的图：
![[Norm.png]]
### Batch Norm
对于 Batch Norm，就是跨 batch，对于上面的图片，比如针对某个 channel，我们对这个 batch 中的所有样本的同一个通道（同一个 channel）的所有位置（所有 H \* W）进行归一化

### Layer Norm
同样对于 Layer Norm，这里就是对于这个样本的所有通道当做整体做一个 Norm

### Instance Norm
Instance 的粒度更小，在 Layer 的基础上，把每个 channel 也作为单独的整体，Norm一下

### Group Norm
Group Norm 和 Instance Norm 类似，把几个 通道作为一个  Group，一个 Group 作为一个整体 Norm

## 工业界 norm 研究
### RMS Norm