>项目连接: [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)
>参考文章：[2025最快下手vLLM的项目-nanovllm源码解读](https://zhuanlan.zhihu.com/p/1925484783229698084)

# 项目结构
```
nanovllm/
├── engine/                    #   推理引擎核心
│   ├── llm_engine.py         #   └── 总协调器，驱动整个推理流程
│   ├── scheduler.py          #   └── 智能调度器，决定执行顺序
│   ├── block_manager.py      #   └── KV缓存内存管理 (PagedAttention核心)
│   ├── model_runner.py       #   └── 单GPU上的模型执行器
│   └── sequence.py           #   └── 请求序列的数据结构
├── layers/                    # ⚙️ 神经网络层实现
│   ├── attention.py          #   └── FlashAttention + KV缓存管理
│   ├── sampler.py            #   └── 从logits采样生成token
│   ├── linear.py             #   └── 支持张量并行的线性层
│   ├── layernorm.py          #   └── RMS LayerNorm
│   ├── rotary_embedding.py   #   └── 旋转位置编码 (RoPE)
│   ├── activation.py         #   └── 激活函数 (SiLU)
│   └── embed_head.py         #   └── 词嵌入和语言模型头
├── models/                    #  ️ 具体模型架构
│   └── qwen3.py              #   └── Qwen3模型完整实现
├── utils/                     #   工具模块
│   ├── context.py            #   └── 全局上下文状态管理
│   └── loader.py             #   └── 模型权重加载器
├── config.py                 # ⚙️ 配置管理
├── llm.py                   #   用户接口入口
└── sampling_params.py       #   采样参数定义
```

# 整体架构图
```text
                          用户接口层
                        ┌─────────┐
                        │   LLM   │
                        └────┬────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   LLMEngine     │  ← 总指挥官
                    │                 │
                    │ ┌─────────────┐ │
                    │ │ Scheduler   │ │  ← 调度器
                    │ │ BlockManager│ │  ← 内存管家  
                    │ │ ModelRunner │ │  ← 执行引擎
                    │ └─────────────┘ │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Model + Layers  │
                    │                 │
                    │ Qwen3Model      │  ← 模型主体
                    │ Attention       │  ← 注意力层
                    │ Sampler         │  ← 采样器
                    └─────────────────┘
```

# 关键数据流
## 数据流图
```text
用户输入 → Tokenizer → Sequence → Scheduler → ModelRunner → Model → Sampler → 输出

详细展开：
prompts     token_ids    Sequence     scheduled    input_ids    logits   token_ids    decoded_text
  ↓            ↓           ↓           seqs          ↓           ↓          ↓            ↓
"Hello"  →  [123,45]  →  Seq#1    →  [Seq#1]   →  tensor   →  tensor  →    67      →  " world"
"Hi"     →  [89,12]   →  Seq#2    →  [Seq#2]   →  [...]    →  [...]   →    23      →  " there"
```

## 数据结构转换过程
```text
# 1. 用户输入转换为Sequence对象
prompts = ["Hello", "How are you?"]
sampling_params = SamplingParams(temperature=0.8, max_tokens=50)

# 2. 每个prompt变成一个Sequence
sequences = []
for prompt in prompts:
    token_ids = tokenizer.encode(prompt)  # "Hello" → [123, 45, 67]
    seq = Sequence(token_ids, sampling_params)
    sequences.append(seq)

# 3. Scheduler决定哪些序列一起执行
scheduled_seqs, is_prefill = scheduler.schedule()
# 可能返回: ([seq1, seq2], True)  表示批量处理2个序列的prefill

# 4. ModelRunner准备模型输入
if is_prefill:
    input_ids, positions = prepare_prefill(scheduled_seqs)
    # input_ids: [123, 45, 67, 89, 12, 34]  # 拼接所有序列
    # positions: [0, 1, 2, 0, 1, 2]        # 每个token在序列中的位置
else:
    input_ids, positions = prepare_decode(scheduled_seqs)
    # input_ids: [67, 34]                   # 每个序列的最后一个token
    # positions: [3, 3]                    # 下一个位置（这里是为了指向计算出的新kv存储的位置，所以对于decode来说是下一个位置）

# 5. 模型计算得到logits
logits = model(input_ids, positions)
# logits shape: [batch_size, vocab_size]

# 6. Sampler生成下一个token
next_tokens = sampler(logits, temperatures)
# next_tokens: [89, 56]  # 为每个序列生成下一个token
```

# vLLM 工作流程

>以 bench.py 为例

### 初始化
-> `LLM()`  初始化一个 LLM 对象：
- 加载 config 配置
- 创建多个 `ModelRunner` 进程实现多线程（避免 Python 的GIL全局解释锁）
- 创建 `Scheduler` 管理器对象

具体而言
```python
def __init__(self, model, **kwargs):
	config_fields = {field.name for field in fields(Config)}
	config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
	config = Config(model, **config_kwargs)
	self.ps = []
	self.events = []
	# CUDA 强制要求 -> spawn 启动方法会创建完全干净的新py解释器，确保每个进程能够独立、安全初始化CUDA环境
	ctx = mp.get_context("spawn")
	for i in range(1, config.tensor_parallel_size):
		event = ctx.Event() # 子进程同步事件对象
		process = ctx.Process(target=ModelRunner, args=(config, i, event)) # 创建子进程，target是进程启动后的类或函数
		process.start() # 启动子进程
		self.ps.append(process) # 方便后续的进程管理
		self.events.append(event) # 用于主进程和子进程间的通信
	self.model_runner = ModelRunner(config, 0, self.events)
	self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
	config.eos = self.tokenizer.eos_token_id
	self.scheduler = Scheduler(config)
	atexit.register(self.exit)
```

关于 torch.multiprocessing 可以参考文章：[torch.multiprocessing](https://blog.csdn.net/weixin_42764932/article/details/132090185)

#### ModelRunner
```python
def __init__(self, config: Config, rank: int, event: Event | list[Event]):
	self.config = config
	hf_config = config.hf_config
	self.block_size = config.kvcache_block_size
	self.enforce_eager = config.enforce_eager
	self.world_size = config.tensor_parallel_size
	self.rank = rank
	self.event = event
	
	dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank) # 分布式训练组
	torch.cuda.set_device(rank)
	default_dtype = torch.get_default_dtype()
	torch.set_default_dtype(hf_config.torch_dtype)
	torch.set_default_device("cuda")
	self.model = Qwen3ForCausalLM(hf_config) # 自己重新写的支持并行的 Qwen3ForCausalLM
	load_model(self.model, config.model)
	self.sampler = Sampler()
	self.warmup_model()
	self.allocate_kv_cache()
	if not self.enforce_eager:
		self.capture_cudagraph()
	torch.set_default_device("cpu")
	torch.set_default_dtype(default_dtype)
	
	if self.world_size > 1:
		if rank == 0:
			self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
			dist.barrier()
		else:
			dist.barrier()
			self.shm = SharedMemory(name="nanovllm")
			self.loop()
```

关于 ModelRunner 的初始化过程，可以看到

#### Scheduler