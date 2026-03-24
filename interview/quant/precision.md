# 说一下 FP4？
> 参考这篇文章： https://zhuanlan.zhihu.com/p/1936783804216873571

FP4 一般来说是 E2M1，NV 的 Blackwell 架构等硬件都是支持 FP4 这个数据类型的  
一般来说常见的 FP 格式有 MXFP4 和 NVFP4  

## MXFP4
MX 是 micro scaling缩写，MXFP4 在存的时候元素用 E2M1，数值计算根据 IEEE 规则来看：
- E > 0 时： $value = (-1)^s \times 2 ^{E-Bias}\times {(1 + M)}, Bias=1$ 
- E = 0 时： 规定参与计算的 E = 1
所以 E2M1 的数值映射应该如下：

| 二进制 (S-E-M) | 计算过程 ( $2^{E-1} \times (1+M)$    或特殊处理) | 最终数值 |
| :---------- | :-------------------------------------- | :--- |
| `0 00 0`    | 零 (Zero)                                | 0.0  |
| `0 00 1`    | 非正规数: $2^{0} \times 0.5$    (注1)        | 0.5  |
| `0 01 0`    | $2^{1-1} \times 1.0 = 2^0 \times 1$     | 1.0  |
| `0 01 1`    | $2^{1-1} \times 1.5 = 2^0 \times 1.5$   | 1.5  |
| `0 10 0`    | $2^{2-1} \times 1.0 = 2^1 \times 1$     | 2.0  |
| `0 10 1`    | $2^{2-1} \times 1.5 = 2^1 \times 1.5$   | 3.0  |
| `0 11 0`    | $2^{3-1} \times 1.0 = 2^2 \times 1$     | 4.0  |
| `0 11 1`    | $2^{3-1} \times 1.5 = 2^2 \times 1.5$   | 6.0  |

此外，MXFP4 是 per group 量化的，group size=32，scale用 8bit存，E8M0 范围是 \[ $2^{-127}$ , $2^{127}$] 

## NVFP4
1. 和 MXFP4 差不多，但 NVFP4 的精度更好，因为 NVFP4 的 scale 是 FP8_E4M3 的，粒度更细
2. 但 FP8 本身取值范围很小表示范围很粗，所以 NVFP4 还加了个 per tensor的scale（FP32）
3. NVFP4 的 group size是16也更细

![[NVFP4.png]]

最新的[英伟达 B 系列显卡](https://zhida.zhihu.com/search?content_id=261376133&content_type=Article&match_order=1&q=%E8%8B%B1%E4%BC%9F%E8%BE%BE+B+%E7%B3%BB%E5%88%97%E6%98%BE%E5%8D%A1&zhida_source=entity)支持直接用 NVFP4 计算，也就是推理时，将 activation 从 16 bit 动态量化到 NVFP4 与同为 NVFP4 的 weight 进行 GEMM 计算，之后用 16 bit 输出。由于 MXFP4 和 NVFP4 的编码一致，只是 scale 和 group size 不同，提前将 scale 做好转换即可，所以 MXFP4 也可直接计算。

# int4 可能比 int8 快吗？
主要还是从带宽等角度来回答：
1. int4 相比 int8 数据搬运量会减少一半，带宽压力下降
2. 同样的 tensor core int4 的并行度可能更高，可以放更多元素
3. cache、SRAM利用率更高

# 对比一下 fp 和 int？
直觉上来看，fp 其实更适合 LLM 权重的量化，因为 fp 格式去存储是两边稀疏中间稠密，且表示范围更大，但是也不是说分布合适，fp就一定比int表征误差更小，实际上到了低精度阶段，fp 和 int 的差距并不大，低精度都是 per group，fp 的稠密区已经很难和group分布对上了。

低精度 int 比 fp好是因为对旋转算法更加亲和，int 和 fp 在 8bit 的时候区别才比较大，动态范围和甜区都差很多
	为什么说低精度 int 比 fp 对旋转算法更加亲和呢？因为旋转量化的本质就是让分布更加均匀，旋转量化本质是做了一个正交变换而矩阵模长不变

## 一个简单场景题
对于有异常值的情况，数据分布越均匀越好，在量化到 int8、fp8、int4 都没有问题，但是量化到 fp4 缺不是如此，为什么？

1. int 是均匀量化：int8、int4 通过旋转等方法让数据分布均匀，那么就能将数值映射到均匀间隔的整数桶，如果有异常值，那么 scale 被拉大，大部分数据被压在很少几个 bucket
2. fp8 指数位表示的范围够用
3. fp4 做旋转或者其他操作让数据分布均匀时会发生一个问题：能量被扩散到更多维度，那么更多的值会落在中等幅度区间，但 fp4 没有足够指数表示大区间，同时尾数也比较粗糙，那么很多值被 round 到同一个浮点数，反而误差更大

