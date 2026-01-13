# 加入 Cache

针对我们的 ToyT5Model，如果我们想对他进行加速，首先考虑的应该是加入 kvcache，对于 kvcache，应该加在哪里呢？仔细想想，有这两个部分：
1. 参与 Decoder Layer cross attn 运算的 memory，这部分由 Encoder 算完就不动了，在推理时每一层的 memory 还要重新做一次 $W_k$ 和 $W_v$ 的转换，这一步就没必要了
2. Decoder 自己 self attn 中之前 token kv cache
所以我们对 self attn 和 cross attn 进行修改：

>cross attn v0

```python
def forward(self, x, memory, position_embedding=None, mask=None):
	"""
	x : [B, Lt, D] (Lt is L of target)
	memory : [B, Ls, D] (Ls if L of source)
	return attn_score
	"""
	
	B, Lt, _ = x.shape
	B, Ls, _ = memory.shape
	q, k, v = self.q(x), self.k(memory), self.v(memory)
	
	# split and reshape
	q = q.reshape(B, Lt, self.num_head, self.head_dim).permute(0, 2, 1, 3) # [B, H, Lt, D]
	k = k.reshape(B, Ls, self.num_head, self.head_dim).permute(0, 2, 1, 3) # [B, H, Ls, D]
	v = v.reshape(B, Ls, self.num_head, self.head_dim).permute(0, 2, 1, 3) # [B, H, Ls, D]
	
	  
	
	# attention
	# score : [B, H, Lt, Ls]
	attn_score = torch.matmul(q, k.permute(0, 1, 3, 2)) / sqrt(self.head_dim)
	if position_embedding is not None:
		attn_score += position_embedding
	
	if mask is not None:
		mask = mask.view(1, 1, mask.size(-2), mask.size(-1)).to(x.device)
	attn_score += mask
	score = torch.softmax(attn_score, dim=-1).matmul(v)
	  
	# concat
	score_cat = score.permute(0, 2, 1, 3).reshape(B, Lt, self.out_dim) # [B, Lt, D]
	
	# o_proj
	score_proj = self.o(score_cat)
	
	return score_proj
```

>self attn v0
```python
def forward(self, x, position_embedding=None):
	B, L, _ = x.shape
	q, k, v = self.q(x), self.k(x), self.v(x)
	
	# split and reshape
	# shape : [B, H, L, D]
	q = q.reshape(B, L, self.num_head, self.head_dim).permute(0, 2, 1, 3)
	k = k.reshape(B, L, self.num_head, self.head_dim).permute(0, 2, 1, 3)
	v = v.reshape(B, L, self.num_head, self.head_dim).permute(0, 2, 1, 3)
	
	  
	
	# attention
	# score : [B, H, L, L]
	attn_score = torch.matmul(q, k.permute(0, 1, 3, 2)) / sqrt(self.head_dim)
	if position_embedding is not None:
		attn_score = attn_score + position_embedding
	score = torch.softmax(attn_score, dim=-1).matmul(v)
	
	  
	# concat
	score_cat = score.permute(0, 2, 1, 3).reshape(B, L, self.out_dim) # [B, L, D]
	
	  
	
	# o_proj
	score_proj = self.o(score_cat)
	
	  
	return score_proj
```

加入 attn 之后：
>cross attn v1

```python
def forward(self, x, memory, position_embedding=None, mask=None, memory_cache=None):
	"""
	x : [B, Lt, D] (Lt is L of target)
	memory : [B, Ls, D] (Ls if L of source)
	return attn_score
	"""
	
	B, Lt, _ = x.shape
	q = self.q(x)
	q = q.reshape(B, Lt, self.num_head, self.head_dim).permute(0, 2, 1, 3) # [B, H, Lt, D]
	
	  
	
	if memory_cache:
	k, v = memory_cache
	else:
	B, Ls, _ = memory.shape
	k, v = self.k(memory), self.v(memory)
	k = k.reshape(B, Ls, self.num_head, self.head_dim).permute(0, 2, 1, 3)
	v = v.reshape(B, Ls, self.num_head, self.head_dim).permute(0, 2, 1, 3)
	
	  
	
	# attention
	# score : [B, H, Lt, Ls]
	attn_score = torch.matmul(q, k.permute(0, 1, 3, 2)) / sqrt(self.head_dim)
	if position_embedding is not None:
	attn_score += position_embedding
	if mask is not None:
	mask = mask.view(1, 1, mask.size(-2), mask.size(-1)).to(x.device)
	attn_score += mask
	score = torch.softmax(attn_score, dim=-1).matmul(v)
	
	  
	
	# concat
	score_cat = score.permute(0, 2, 1, 3).reshape(B, Lt, self.out_dim) # [B, Lt, D]
	
	  
	
	# o_proj
	score_proj = self.o(score_cat)
	
	  
	
	return score_proj, (k, v)
```

对应的 self attn：
>self attn v1
```python
def forward(self, x, position_embedding=None, mask=None, pask_kv_cache=None):
	B, L, _ = x.shape
	q, k, v = self.q(x), self.k(x), self.v(x)
	
	
	# split and reshape
	# shape : [B, H, L, D]
	q = q.reshape(B, L, self.num_head, self.head_dim).permute(0, 2, 1, 3)
	k = k.reshape(B, L, self.num_head, self.head_dim).permute(0, 2, 1, 3)
	v = v.reshape(B, L, self.num_head, self.head_dim).permute(0, 2, 1, 3)
	
	  
	if pask_kv_cache:
		k_cache, v_cache = pask_kv_cache
		k = torch.cat([k_cache, k], dim=-2)
		v = torch.cat([v_cache, v], dim=-2)
	
	
	# attention
	# score : [B, H, L, L]
	attn_score = torch.matmul(q, k.permute(0, 1, 3, 2)) / sqrt(self.head_dim)
	if position_embedding is not None:
		attn_score = attn_score + position_embedding # 这里和 T5 的embedding长度对应
	if mask is not None:
		mask = mask.view(1, 1, mask.size(-2), mask.size(-1)).to(x.device)
	attn_score += mask
	score = torch.softmax(attn_score, dim=-1).matmul(v)
	
	  
	
	# concat
	score_cat = score.permute(0, 2, 1, 3).reshape(B, L, self.out_dim) # [B, L, D]
	
	
	# o_proj
	score_proj = self.o(score_cat)
	
	
	return score_proj, (k, v)
```

看一下 test 结果
```python
# 实验配置
batch_size = 4
input_len = 256
gen_len = 1024

config.num_layers = 12
config.model_dim = 768
config.num_head = 12
config.ffn_dim = 3072


"""
📦 模型参数占用显存: 785.48 MB
🚀 开始高压测试 [Batch: 4, 生成长度: 1024]
------------------------------------------------------------
【无缓存全量模式】:
  > 总耗时: 29.98s | 每 Token 均摊: 29.28ms
  > 峰值分配 (含权重): 2535.67MB
  > 系统预留 : 19692.00MB
------------------------------------------------------------
【KV Cache 增量模式】:
  > 总耗时: 9.40s | 每 Token 均摊: 9.18ms
  > 峰值分配 (含权重): 1993.89MB
  > 系统预留 : 1458.00MB
------------------------------------------------------------
🔥 加速比: 3.19x
"""
```

可以看到大概快了 3.2 倍左右

# cat 优化
进一步观察有没有还能够优化的地方？
对比原始版本的 attn，主要的 generate 瓶颈在于 token 的 cat 拼接后继续参与运算，这个 cat 操作需要系统分配巨大的显存确保能够成功连接，所以我们可以考虑手动先分配好这个内存占用。总的来说优化思路有这三点：
1. 空间压缩：先分配再填入，空间换时间
2. 计算压缩：kv Cache 量化
3. 算子压缩：FlashAttention / PageAttention 等通过 tiling 技术减少显存读写

这里只实现空间压缩部分，其余会在其他章节有涉及：
> 无优化

```python
if pask_kv_cache:
	k_cache, v_cache = pask_kv_cache
	k = torch.cat([k_cache, k], dim=-2)
	v = torch.cat([v_cache, v], dim=-2)
```

>优化后：

kvcache 类预先分配
```python
class KVcache():
	def __init__(
	self,
	num_layers,
	batch_size,
	num_head,
	seq_len,
	head_dim,
	device
	):

	"""
	self.encoder_cache : {
		idx_layer : (memory_k_cache, memory_v_cache)
	}
	self.decoder_cache : {
		idx_layer : (token_k_cache, token_v_cache)
	}
	"""
	
	self.num_layers = num_layers
	self.encoder_cache = defaultdict(tuple)
	self.decoder_cache = defaultdict(tuple)
	
	# 预分配 decoder
	for i in range(num_layers):
		self.decoder_cache[i] = (
		torch.zeros((batch_size, num_head, seq_len, head_dim), device=device), # k
		torch.zeros((batch_size, num_head, seq_len, head_dim), device=device), # v
		)
```

```python
if pask_kv_cache: # [B, H, L, D]
	k_buffer, v_buffer = pask_kv_cache
	k_buffer[:, :, step : step + L, :] = k
	v_buffer[:, :, step : step + L, :] = v
	k = k_buffer[:, :, :step + L, :]
	v = v_buffer[:, :, :step + L, :]
```

# 结果

```
配置：
batch_size = 16
input_len = 256
gen_len = 512
config.num_layers = 16
config.model_dim = 512
config.num_head = 16
config.ffn_dim = 1024

📦 模型参数占用显存: 340.16 MB

🚀 开始高压测试 [Batch: 16, 生成长度: 512]
------------------------------------------------------------
【无缓存全量模式】:
  > 总耗时: 25.92s | 每 Token 均摊: 50.62ms
  > 峰值分配 (含权重): 1565.77MB
  > 系统预留 (接近smi): 23548.00MB
------------------------------------------------------------
【KV Cache 增量模式】:
  > 总耗时: 6.34s | 每 Token 均摊: 12.38ms
  > 峰值分配 (含权重): 1249.84MB
  > 系统预留 (接近smi): 940.00MB
------------------------------------------------------------
🔥 加速比: 4.09x
```
在这个配置下，模型加速比还是比较可观的，同时对于 cat 临时分配带来的 GPU 显存压力也有不小的缓解效果