# Profile

对于一个写好的模型，首先我们要知道他在那些地方是有问题的，这里我们使用 `torch.profile` 这个工具

```python
def run_profile():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    tokenizer = BPE_Tokenizer.load("../Transformers/Tokenizer/tokenizer.pt")
    config = LlamaConfig(tokenizer)
    config.num_layers = 16 
    config.dim = 512
    config.num_heads = 16
    config.intermediate_size = 1024
    config.num_kv_heads = 4
    model = LlamaGPT(config).to(device).bfloat16() # 使用 FP16，更符合推理场景
    
    bsz = 4
    seq_len = 256
    input_ids = torch.randint(0, config.vocab_size, (bsz, seq_len)).to(device)

    print("Warming up...")
    with torch.no_grad():
        for _ in range(5):
            model(input_ids)

    print("Profiling...")
    with profile(
        activities=[
            ProfilerActivity.CPU, 
            ProfilerActivity.CUDA
        ], 
        record_shapes=True,
        profile_memory=True  # 建议开启，可以看到算子的内存分配情况
    ) as prof:
        with record_function("model_inference"):
            with torch.no_grad():
                model(input_ids)

    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))

    prof.export_chrome_trace("trace.json")
    print("Trace saved to trace.json. Open chrome://tracing to view.")
```

一个简单实例，详细的 profile 操作可以参考：[144_推理时延优化：Profiling与瓶颈分析 - 使用PyTorch Profiler诊断推理延迟，优化矩阵运算的独特瓶颈-阿里云开发者社区](https://developer.aliyun.com/article/1684085)

首先，我们要给每个需要被检测的操作打上 profile 标签，例如：

```python
class LlamaLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = LlamaAttention(config)
        self.mlp = LlamaMLP(config.dim, config.intermediate_size)
        self.attention_norm = RMSNorm(config.dim)
        self.ffn_norm = RMSNorm(config.dim)

    def forward(self, x, freqs_cis, mask=None, kv_cache=None, start_pos=0):
        # h = x + self.attention(self.attention_norm(x), freqs_cis, mask, kv_cache, start_pos)
        # out = h + self.mlp(self.ffn_norm(h))
        with record_function("LlamaLayer_Total"):			# <-打上标签
            with record_function("Attention_Block"):
                h = x + self.attention(self.attention_norm(x), freqs_cis, mask, kv_cache, start_pos)
            with record_function("MLP_Block"):
                out = h + self.mlp(self.ffn_norm(h))
        return out
```

然后运行profile脚本，结果如下：

![[profile.png]]

其中 aten 是底层C++库（A Tensor Library），我们看打上标签的，比如 `RMS Norm` ，他的一些信息为：

- self cpu%：这个大算子自己占用的 cpu 时间占用整个 profile 的百分比
    - 值很高意味着 python 逻辑开销大，在进行费张量计算的操作（循环、逻辑判断等）
- self cpu：大算子自己占用的 cpu 时间（不包括下面的子算子）
- cpu total：大算子总的 cpu 时间
    - 衡量一个大模块在 cpu 调度层面的贡献
- cpu time avg：CPU total / # of Calls
    - 判断模型是否正确以及模块的平均贡献

cuda同理

- CPU Mem / Self CPU Mem：算子申请的 CPU 内存（通常是内存条上的空间）
    - 在推理阶段，如果这里有值，通常是数据在进行 `CPU <-> GPU` 的搬运（Host-to-Device）
- CUDA Mem / Self CUDA Mem：算子申请的显存（GPU 内存）
    - 正值表示算子申请了新的空间
    - 负值表示算子释放了额外的空间
    - 如果频繁正值说明操作中创建了 `new_tensor` 可能会加剧显存碎片并降低速度

总的来说，我们应该关注：

-  `CUDA total`：锁定哪个模块（比如 `RMSNorm`）最慢。
-  `# of Calls`：确定它是高频小算子还是低频大算子。
-  `Self CUDA`：当你写好 Triton 算子后，对比它和原来 `aten` 算子组合的 Self 时间。
-  `CUDA Mem`：检查你的优化是否成功减少了临时内存的申请。

在我们这里，我们看到，这里 RMS_Norm 显然不正常：

- 一方面，他的 cuda/cpu time avg 比矩阵乘还要高
- 另一方面，他的显存申请也比较高

所以我们首先考虑优化 RMS Norm这个算子

## RMS Norm

### 分析

```python
def forward(self, x):
        # norm_x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # return (norm_x.float()).type_as(x) * self.weight
        with record_function("OP: RMSNorm"):
            norm_x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            return (norm_x.float()).type_as(x) * self.weight
```

我们先看这里的`forward`写法，为什么会慢？

- 这里 norm_x 实际进行了
    - pow
    - mean
    - add
    - rsqrt
    - mul

五步操作，每个操作是一个小 kernel，那么就是大算子的调用次数 * 5，每个小 kernel 需要额外进行 CPU、GPU的通信开销，以及后续的 float 、type_as等操作，都是这个 rms norm算子慢的原因之一

### CUDA 优化
>首先我们规定传入的 Tensor 的连续的 2D 向量，也就是（Batch_size * seq_len, model_dim）维度的向量

#### RMS_Norm_v1

首先我们写一版能跑起来的 CUDA 代码，看看里面的逻辑是什么：
```c++
__global__ void rms_norm_kernel(
	float* output,
	const float* input,
	const float* weight,
	int dim,
	float eps
){
	int row = blockIdx.x; // 当前 block 在grid中的索引
	const float* input_row = input + row * dim; // 指针偏移
	float* out_row = output + row * dim; // 输出指针偏移
	
	if (threadIdx.x == 0){
		float sum = 0.0f; // 线程级别私有变量
		
		for (int i = 0; i < dim; ++i){
			sum += input_row[i] * input_row[i]; //不能换成 pow，否则每个线程调用一次非常慢
		}
		
		float rms = rsqrt(sum / dim + eps);
		for(int i = 0; i < dim; ++i){
			out_row[i] = (input_row[i] * rms) * weight[i];
		}
	}
};
```
先看看这一版的效果怎么样，编译完跑一下结果，(测试数据是`bsz, seq_len, hidden_size = 4, 1024, 512`, 比较的Pytorch原生版本和手写CUDA版本速度，下同)：
![[rms_v1.png]]
可以看到，我们手写的版本甚至比原生 torch 还要慢，分析一下，是哪里有问题呢：
- 线程没调用：我们看到，首先 v1 版本只有线程 0 在干活，所以首先我们要使用多线程把显卡给调动起来
- IO开销大：其次，这里sum实际上每个线程在规约得到 sum 后是共用的，所以我们考虑对这些频繁取用的引入 shared mem，减少 io 开销
- 算法复杂度：此外从复杂度考虑，这里如果要求到最终的 sum，我们的版本是 O(n) 的复杂度，但实际上我们可以用分治的思想给他优化成 O(n) 的

#### RMS_Norm_v2

根据上面三点思考，我们写出 V2 版本：
```cpp
__global__ void rms_norm_kernel_v2(
float* output,
const float* input,
const float* weight,
int dim,
float eps
){
	int row = blockIdx.x;
	int col = threadIdx.x;
	
	const float* x_row = input + row * dim; // 这也是私有变量，不过指针的开销非常小
	float* out_row = output + row * dim;
	
	__shared__ float rms; // share memory
	__shared__ float s_diff_sq[BLOCK_SIZE];
	
	float val = 0.0f;
	if (col < dim){
		s_diff_sq[col] = x_row[col] * x_row[col]; // 存平方
	} else s_diff_sq[col] = 0.0;
	
	__syncthreads(); // 线程同步
	
	// 规约 reduce
	for(int stride = blockDim.x / 2; stride > 0; stride /= 2){
		if(col < stride){
			s_diff_sq[col] += s_diff_sq[col + stride];
		}
		__syncthreads(); // 等待线程同步;
	}
	
	
	if(col == 0){
		float sum = s_diff_sq[0] / dim;
		rms = rsqrt(sum + eps);
	}
	
	__syncthreads(); // 同步，等 0 算完
	if(col < dim){
		out_row[col] = x_row[col] * rms * weight[col];
	}
};
```

看一下结果：
![[rms_v2.png]]

嗯，这次速度快不少，甚至精度也高了一些，还有没有可以进一步优化的地方呢？

#### RMS_Norm_v3

这里简单介绍一下 GPU 的 warp：
- Warp 是线程调度和执行的基本单位
- 每个 warp 包含 32 个连续的线程（即 threadIdx.x 连续的 32 个线程）
- GPU 的 SM（Streaming Multiprocessor）以 SIMT（Single Instruction, Multiple Thread） 方式执行：一个 warp 中的所有线程在同一时刻执行同一条指令，但操作各自的数据

warp 是 GPU 硬件层面的并行单元，所有优化（如 `__shfl_*` 指令都是围绕 `warp`）实现的
进一步，利用 `__shfl_down_sync` 可以实现相邻 32 线程的寄存器级别的数值取用，所以我们可以进一步压缩 IO 空间，以 warp 为最小块

此外我们刚刚的代码有一个问题，如果分发的线程数比 dim 少怎么办？这也是一个算法设计的缺陷

针对上面两个问题，主体的解决方案思路为：
- 以 blockDim，一个 block 的线程个数为主导，通过 blockDim 的循环取用元素
	- 写法类似如下，也就是 0 线程负责 0、blockDim、2 * blockDim 、 ... 等
```cpp
for (int i = tid; i < dim; i += blockDim.x){
	float v = x_row[i];
	sum += v * v; // 加到自己的 sum 寄存器中
}
```

- 当然我们还是想利用 warp，warp 是连续的 32线程，且 `BLOCK_SIZE`  最大为 1024，所以我们可以开一个 32 大小的 float 数组，数组的每个 float 放一个 warp 规约的sum
- 紧接着，我们把 warp放到前32个连续线程中，再次利用 warp 规约前32个线程就实现了最终的功能

代码如下：
```cpp
__global__ void rms_norm_kernel_v3(
	float* output,
	const float* input,
	const float* weight,
	int dim,
	float eps
){
	/*
	优化思路:
		1. 处理非 2幂次维度：找到大于 dim 的最小2幂次
		2. 优化reduce过程中写入share_mem：使用 warp 寄存器操作
	*/
	
	  
	int row = blockIdx.x;
	int tid = threadIdx.x;
	
	const float* x_row = input + row * dim;
	float* out_row = output + row * dim;
	
	float sum = 0.0f;
	
	// 处理线程和dim不对应的情况
	// 线程 0 处理 0、blockDim、2*blockDim ...
	// 线程 1 处理 1+0、1+blockDim、1+2*blockDim ...
	// ... 归到 blockDim 大小
	for (int i = tid; i < dim; i += blockDim.x){
		float v = x_row[i];
		sum += v * v; // 加到自己的 sum 寄存器中
	}
	
	// block 内规约
	__shared__ float warp_sum[32]; // warp 是 block 内共享，block 内所有线程都能看到，存32个float
	int lane = tid % 32; // warp 内索引
	int wid = tid / 32; // warp idx
	
	sum = warpReduceSum(sum); // 每32个线程进行规约，也就是规约到 warp 个数
	
	if (lane == 0) warp_sum[wid] = sum; // 由 0 号线程干活, 这里 wid 之所以不会越界是因为 block_size最多1024
	__syncthreads();
	
	// warp 规约，这里需要连续的32，所以取前 32 个
	// blockDim < 1024 / 32 < 32, 所以对应的第一个warp wid，这时候他们的 lane 对应的就是 wid
	// 就是把 warp 每个 wid 写入到前32个线程的 sum中
	sum = (tid < blockDim.x / 32) ? warp_sum[lane] : 0.0f; // 归到前 32 线程
	if (wid == 0)
		sum = warpReduceSum(sum); // 前32线程规约
	
	__shared__ float s_rms;
	if(tid == 0) // share mem方便后面存取，由 0 号进行计算
		s_rms = rsqrt(sum / dim + eps);
	
	__syncthreads();
	
	for (int i = tid; i < dim; i += blockDim.x){
		output[row * dim + i] = (x_row[i] * s_rms) * weight[i];
	}
};
```

看一下结果
![[rms_v3.png]]
嗯，这次比刚刚快了一倍，还有没有进一步优化的空间呢？
下面介绍的是向量化版本

#### RMS_Norm_v4
首先我们知道， rms_norm 是一个 mem_bound 算子，所以我们要尽可能地去优化他的访存，对于 torch，内部每个 Tensor 是以 16 字节进行对齐的
- dtype ： fp16， 每个元素占 2 字节
- dytpe ： fp32，每个元素占 4 字节
- 无论哪种 `dtype`，PyTorch 分配的内存**起始地址**通常都是 16 字节、64 字节甚至 128 字节对齐的

|PyTorch dtype (Python)|C++ 类型（CUDA）|每个元素大小（bytes）|别名|
|---|---|---|---|
|`torch.float16`|`half`|2|fp16|
|`torch.float32`|`float`|4|fp32|
|`torch.float64`|`double`|8|fp64|
|`torch.int8`|`int8_t`|1|-|
|`torch.int16`|`int16_t`|2|-|
|`torch.int32`|`int32_t`|4|-|
|`torch.int64`|`int64_t`|8|-|
|`torch.bfloat16`|`__nv_bfloat16`|2|bf16|
|`torch.uint8`|`uint8_t`|1|-|

好，那么对于CUDA调用 GPU 来说，这里我们介绍一个 float4 指令，float4 指令可以理解为内存解读指令：
- 取出 16 Byte 的内存地址
- 每 4 Byte 解读为 fp32 的元素

>关于更多 float4 的理解可以参考 [(26 封私信 / 80 条消息) cuda编程中，转为float4是什么？ - 知乎](https://www.zhihu.com/question/574968879)

CUDA对应的内置向量读取格式：

|类型|元素类型|元素数量|总字节数|对齐要求|
|---|---|---|---|---|
|`char4`|signed char|4|4|4 字节|
|`uchar4`|unsigned char|4|4|4 字节|
|`short2`|short|2|4|4 字节|
|`ushort2`|unsigned short|2|4|4 字节|
|`int2`|int|2|8|8 字节|
|`float2`|float|2|8|8 字节|
|`int4`|int|4|16|16 字节|
|`float4`|float|4|16|16 字节|


相当于是一种更高效的访存指令，优化了一开始的四次指令发射

对应代码为：
```cpp
__global__ void rms_norm_kernel_v4(
	float* output,
	const float* input,
	const float* weight,
	int dim,
	float eps
){
	int block_idx = blockIdx.x;
	int tid = threadIdx.x;
	
	// input 和 output 本质是 float 指针，我们强转成 float4 指针
	// float4 是一次性处理 4 个 float 的内置类型
	const float4* x_vec = (const float4*)(input + block_idx * dim);
	const float4* w_vec = (const float4*)weight;
	float4* out_vec = (float4*)(output + block_idx * dim);
	
	int vec_dim = dim / 4; // 向量化的维度是原来的 1/4
	float sum = 0.0f;
	
	// 1. 向量化读取与平方累加
	for (int i = tid; i < vec_dim; i += blockDim.x) {
		float4 data = x_vec[i];
		// dim 能被 4 整除得到 vec_dim，且内存地址必须是 16 字节对齐（Tensor 一般是 16字节的）
	
		sum += data.x * data.x;
		sum += data.y * data.y;
		sum += data.z * data.z;
		sum += data.w * data.w;
	}
	
	  
	
	// 2. Block 级规约（沿用 V4 的逻辑）
	__shared__ float s_warp_sums[32];
	int lane = tid % 32;
	int wid = tid / 32;
	
	sum = warpReduceSum(sum);
	if (lane == 0) s_warp_sums[wid] = sum;
	__syncthreads();
	
	  
	sum = (tid < 32) ? s_warp_sums[lane] : 0.0f;
	if (wid == 0) sum = warpReduceSum(sum);
	
	  
	__shared__ float s_rms;
	if (tid == 0) s_rms = rsqrtf(sum / dim + eps);
	__syncthreads();
	
	  
	
	// 3. 向量化写回
	for (int i = tid; i < vec_dim; i += blockDim.x) {
		float4 data = x_vec[i];
		float4 w = w_vec[i];
		float4 res;
		res.x = data.x * s_rms * w.x;
		res.y = data.y * s_rms * w.y;
		res.z = data.z * s_rms * w.z;
		res.w = data.w * s_rms * w.w;
		out_vec[i] = res; // 一次写回 16 字节
	}
}
```

结果如下：
![[rms_v4.png]]

可以看到，比 v3 版本稍微快一点，当然这是在一个小 batch 的情况下了