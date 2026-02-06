>参考文章：
>[(6 封私信) PyTorch分布式训练基础：掌握torch.distributed及其通信功能 - 知乎](https://zhuanlan.zhihu.com/p/692668388)
>[分布式通信包 - torch.distributed — PyTorch 2.9 文档 - PyTorch 文档](https://docs.pytorch.ac.cn/docs/stable/distributed.html#backends)

# 入门
## 多线程创建
### torchrun

首先我们从一段代码开始看：
```python
import os
import torch
import torch.distributed as dist

def main():
    # 1. 初始化进程组
    dist.init_process_group(
        backend="nccl",      # 通信后端
        init_method="env://",# 初始化方式, 让 torch 去环境变量里面读
    )

    # 2. 获取当前进程的“身份”
    rank = dist.get_rank()          # 当前逻辑进程编号
    world_size = dist.get_world_size()  # 设置的总进程数
    local_rank = int(os.environ["LOCAL_RANK"])

    # 3. 绑定当前进程使用的 GPU
    torch.cuda.set_device(local_rank)

    # 4. 构造一个张量（每个 rank 初始值不同）
    x = torch.tensor([rank], device="cuda", dtype=torch.float32)

    # 5. All-Reduce：所有进程一起做规约
    dist.all_reduce(x, op=dist.ReduceOp.SUM)

    # 6. Barrier：全局同步
    dist.barrier()

    if rank == 0:
        print(f"world_size = {world_size}")

    print(f"rank {rank}: after all_reduce, x = {x.item()}")

    # 7. 销毁进程组
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

我们可以看到，进行 dist 通信首先需要初始化对应的进程组，这需要：
- 选择通信后端——决定了用什么协议进行通信
- 选择初始化方式——决定怎么找到彼此并建立连接
	- 一般比较常用的是 env:// 从环境变量中读取，配合 torchrun 会自动设置以下必须信息：
		- MASTER_ADDR=127.0.0.1 ： 主节点 IP
		- MASTER_PORT=29500 ： 主节点监听端口
		- WORLD_SIZE ： 总进程数
		- RANK = 0：当前进程的全局 rank
	- 此外还有就是通过共享文件系统 file://
		- 所有进程通过读写一个共享文件来同步连接信息
		- 速度比较慢
	- 以及通过 tcp://ip:port 显式指定主节点的方式
		- 比如 ： init_method="tcp://192.168.1.100:23456"
		- 所有节点连接到这个 IP.PORT 来注册自己
		- 且需要手动保证 rank0 的程序先启动
- 执行流程为（以 env 初始化为例）：
	- 所有进程从环境变量中知道自己从 localhost:29500 集合
	- 所有进程尝试连接这个地址
	- rank0 充当协调者收集所有进程的信息
	- 一旦 world_size 个进程都连接上，那么 init_process_group 返回成功
	- 此后一些分布式操作就可以通过 nccl 协议进行高效通信

随后可以进行一些分布式操作，比如：
- all reduce
- all gather
- broadcast
最后销毁线程组

注意，init_process_group 只是注册通信，他没有进程进程的创建或者是其他的操作，创建是通过 torchrun 脚本实现的：
```bash
OMP_NUM_THREADS=2 torchrun --nproc_per_node=4 --master_port=29501 example.py
```

这里我指定的 OMP_NUM_THREADS 每个进程的线程个数，torchrun做的事情是：
- 创建了 4 个子进程
- 并对每个子进程设置好对应的环境变量
- 然后每个进程都运行一次 example 脚本

### mp
当然我们也可以选择自己创建，torch 提供了对应的 api
```python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank, world_size):
    # 1. 初始化进程组
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:29501",
        rank=rank,
        world_size=world_size,
    )

    # 2. 绑定 GPU（一进程一卡）
    torch.cuda.set_device(rank)

    # 3. 构造一个张量
    x = torch.tensor([rank], device="cuda", dtype=torch.float32)

    # 4. All-Reduce
    dist.all_reduce(x, op=dist.ReduceOp.SUM)

    # 5. Barrier
    dist.barrier()

    if rank == 0:
        print(f"world_size = {world_size}")

    print(f"rank {rank}: after all_reduce, x = {x.item()}")

    # 6. 销毁进程组
    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = 4

    # 必须在 main 里
    mp.set_start_method("spawn", force=True)

    # 创建进程
    mp.spawn(
        worker,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )
```

我们可以通过 mp.spawn 来创手动建多进程，首先要进行设置，怎么创建？set_start_method 方法有三种
- fork
	- Unix 默认，复制父进程的全部资源
	- 启动很快，且资源继承完整
	- CUDA 运行时不支持（会导致子进程 GPU 错误）
	- 多进程会共享一个 CUDA 上下文，导致：
		- GPU 内存冲突
		- 随机崩溃
- spawn
	- 创建新的 python 解释器，继承必要的模块
	- 安全隔离，兼容 CUDA
	- 但是启动比较慢
- forkserver
	- 预启动一个单线程服务进程，按需 fork 子进程

### 通信
创建完线程，我们也操作了 AllReduce 的 sum 操作，接下来看看进程间的点对点通信
- dist.send(tensor, dst)：将张量发送到目标进程 `dst` (rank 编号)
- dist.recv(tensor, src)：从源进程 `src` 接收张量并填充到当前的 `tensor` 中
- 接收和发送必须成对出现，否则会卡死

#### 集合通信
集合通信是指**一组进程**（通常是全组）共同参与的通信行为。根据数据流向，主要分为以下几种：
- **Broadcast**: 一个发，所有人收。用于同步模型初始参数。
- **Scatter**: 一个进程把一个大的 Tensor 劈开，分发给所有人。
- **Gather**: 所有人把各自的小 Tensor 交给某一个进程，拼成一个大的。
- **All-Gather**: 所有人把自己的数据给所有人。常用于获取所有显卡的全局信息（如验证集结果汇总）。
- **Reduce**: 所有人按某种规则（如求和）把数据汇总给某一个人。
- **All-Reduce**: 所有人汇总后，每个人都拿到最终的汇总结果。这是 **DDP 更新梯度**的最核心操作。
- **Reduce-Scatter**: 汇总后，再把结果劈开分给所有人。这是 **FSDP** 的核心基础。

常用的一些方法包括：

| 方法                                               | 关键参数                                                     | 用法 / 语义说明                                                            |
| ------------------------------------------------ | -------------------------------------------------------- | -------------------------------------------------------------------- |
| `dist.broadcast(tensor, src)`                    | `tensor`（in-place）  <br>`src`（源 rank）                    | **单点 → 全体**：把 `src` rank 上的 `tensor` 广播到所有进程，其它 rank 的同名 tensor 会被覆盖 |
| `dist.all_reduce(tensor, op)`                    | `tensor`（in-place）  <br>`op`（SUM / MAX / MEAN 等）         | **全体 → 全体**：所有 rank 的 tensor 参与规约，结果在每个 rank 上都一样（DDP 梯度同步核心）        |
| `dist.reduce(tensor, dst, op)`                   | `tensor`（in-place）  <br>`dst`（目标 rank）  <br>`op`         | **全体 → 单点**：只有 `dst` 拿到规约结果，其它 rank 的 tensor 无意义                     |
| `dist.all_gather(output_list, input_tensor)`     | `output_list`（长度=world_size）  <br>`input_tensor`         | **全体 → 全体（拼接）**：收集所有 rank 的 input，每个 rank 都得到完整列表                    |
| `dist.gather(input_tensor, gather_list, dst)`    | `input_tensor`  <br>`gather_list`（仅 dst 提供）  <br>`dst`   | **全体 → 单点（拼接）**：只在 `dst` 上得到所有 rank 的数据                              |
| `dist.scatter(output_tensor, scatter_list, src)` | `output_tensor`  <br>`scatter_list`（仅 src 提供）  <br>`src` | **单点 → 全体（分发）**：`src` 把 list 按 rank 分发给各进程                           |
| `dist.reduce_scatter(output, input_list, op)`    | `output`  <br>`input_list`（长度=world_size）  <br>`op`      | **全体 → 全体（先规约再切分）**：每个 rank 只拿到规约结果的一段（ZeRO / FSDP 常用）               |
| `dist.all_to_all(output_list, input_list)`       | `output_list`  <br>`input_list`                          | **全体 ↔ 全体（重排通信）**：每个 rank 给每个 rank 发一块数据（MoE、token routing）          |
| `dist.barrier()`                                 | 无                                                        | **同步点**：所有 rank 在此阻塞，直到全部到达（调试/阶段对齐用）                                |

# DP

DP的工作流如下：
1. 主设备（通常是 cuda:0）持有原始模型副本
2. 输入一个 batch（如 256 张图）
3. 将 batch 拆成 2 份（每份 128 张），分别发送到 GPU 0 和 GPU 1
4. 在每张卡上复制一份完整的模型
5. 每张卡独立做：
    - 前向传播 → 得到 loss
    - 反向传播 → 计算梯度
6. 将所有卡上的梯度 gather 到主卡（cuda:0）
7. 在主卡上求梯度平均值
8. 用平均梯度更新主卡上的模型参数
9. 把更新后的模型参数 broadcast 回其他 GPU

DP 到其他卡上的相当于是工作副本，被 DP 包裹的 model 在计算损失的时候汇总到主卡，并 backward 的时候会回传梯度求平均，然后分发到副卡更新权重，最终的model也是取主卡上的

## DP的通信

DP的通信不走 NCCL，NCCL是专门为了多GPU集体通信的高性能库，DP更多的是采用类似 Tensor.copy_()、torch.cat() 之类的操作完成的隐式通信，底层依赖 CUDA 的点对点（P2P）内存拷贝比如（cudaMemcpyPeer），可能走NVLink 或者 PCIe

### 通信路径

当 DP 执行如下操作时：
```python
# 假设 input 分成 chunks，发往不同 GPU
replicas = replicate(module, device_ids)      # 复制模型
scattered = scatter(input, device_ids)        # 分发输入
outputs = parallel_apply(replicas, scattered, ...)  # 并行前向
gathered = gather(outputs, output_device=0)   # 收集输出到主卡
```

其中 `scatter` 和 `gather` 的核心是：

- 调用 `tensor.to(target_device)` → 触发 GPU 间内存拷贝
- 如果系统支持 CUDA P2P（Peer Access），PyTorch 会使用 `cudaMemcpyPeerAsync`
    - 这时数据可直接通过 NVLink（如果硬件支持）或 PCIe 传输
    - 无需经过主机内存（zero-copy）

(检查是否启用 P2P)
```python
import torch 
print(torch.cuda.can_device_access_peer(0, 1)) # GPU0 能否直连 GPU1？
```

怎么看 DP 的具体通信方式呢？
- `nvidia-smi topo -m` 
	- 输出中 GPU 之间标记为 NV1 / NV2 -> 支持 NVLink
	- 标记为 PIX -> 通过 PCIe 连接
- nsys 抓取：`nsys profile -t cuda,nvtx python train.py` 
	- 在 `.qdrep` 文件中有 `cudaMemcpyPeer`  表示 DP 在做 GPU 间的显存拷贝（P2P）
	- 看不到 nccl* 证明没用 nccl（DP不支持初始化group肯定用不到 nccl）

#### P2P
CUDA 的 P2P 允许：
- GPU 之间相互读写显存
- 无需经过 host mem 中转
- 底层通过 NVLink 或 PCIe 传输数据


# DDP
>参考文章：
>[(5 封私信 / 2 条消息) [原创][深度][PyTorch] DDP系列第一篇：入门教程 - 知乎](https://zhuanlan.zhihu.com/p/178402798)
>[(5 封私信 / 2 条消息) [原创][深度][PyTorch] DDP系列第二篇：实现原理与源代码解析 - 知乎](https://zhuanlan.zhihu.com/p/187610959)
>[(5 封私信 / 2 条消息) [原创][深度][PyTorch] DDP系列第三篇：实战与技巧 - 知乎](https://zhuanlan.zhihu.com/p/250471767)
>


首先说一下DDP和DP的区别

| 维度   | DP                                      | DDP                                       |
| ---- | --------------------------------------- | ----------------------------------------- |
| 进程模型 | 单进程多线程                                  | 多进程（每个 GPU 一个进程）                          |
| 通信方式 | 主 GPU 汇总梯度（Gather → Reduce → Broadcast） | All-Reduce（点对点高效同步）                       |
| 扩展性  | 仅限单机                                    | 支持多机多卡                                    |
| 性能   | 较差（主 GPU 成瓶颈，受 GIL 限制）                  | 更高（负载均衡，无 GIL 限制）                         |
| 内存使用 | 主 GPU 内存压力大                             | 各 GPU 负载均衡                                |
| 代码改动 | 极简（一行 `DataParallel`）                   | 较复杂（需初始化进程组、Sampler、启动方式等）                |
| 启动方式 | 直接 `python train.py`                    | 需用 `torchrun --nproc_per_node=N train.py` |

## Ring-Reduce

![[Ring-Reduce.png]]

可以看到：
- 各**进程**独立计算梯度。
- 每个进程将梯度依次传递给下一个进程，之后再把从上一个进程拿到的梯度传递给下一个进程。循环n次（进程数量）之后，所有进程就可以得到全部的梯度了。
- 每个进程只跟自己上下游两个进程进行通讯，极大地缓解了参数服务器的通讯阻塞现象

### DDP官方实践

DDP官方的最佳实践是每张卡对应一个单独的GPU模型（即一个进程），那比如两个机子，每个机子八张卡，并行数就是 2x8=16，当然我们也可以给每个进程多张卡，总的来说有几种情况：
1. 每个进程一张卡。这是DDP的最佳使用方法。
2. 每个进程多张卡，复制模式。一个模型复制在不同卡上面，每个进程都实质等同于DP模式。这样做是能跑得通的，但是，速度不如上一种方法，一般不采用。
3. 每个进程多张卡，并行模式。一个模型的不同部分分布在不同的卡上面。例如，网络的前半部分在0号卡上，后半部分在1号卡上。这种场景，一般是因为我们的模型非常大，大到一张卡都塞不下batch size = 1的一个模型。

## DistributedSampler

这是 DDP 专用的 dataloader，我们先用一段代码来看 DP 和 DDP 的调用区别
### DP example
```python
def train():
    # 2. 设置单进程可见的卡（比如你有 4 张，我们用 0,1,2,3）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 准备数据 (MNIST)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    # DP 的 Batch Size 是分配到所有卡上的，所以可以设大一点
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    model = TinyCNN().to(device)

    # 3. 核心：使用 DataParallel 包装模型
    if torch.cuda.device_count() > 1:
        print(f"检测到 {torch.cuda.device_count()} 张显卡，开启 DP 模式...")
        # device_ids 默认为所有可见卡
        model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            # 注意：使用了 DataParallel 后，原始模型在 model.module 中
            print(f"Batch {batch_idx}: Loss = {loss.item():.4f}")
            if batch_idx > 50: break # 演示用，跑 50 个 batch 就停

    print("DP 训练演示完成！")
```

这里的流程是：
- train_loader = DataLoader(...) 是懒启动，在具体 for 的时候才从磁盘加载数据
	- 读取过程主要是通过 CPU 加载，如果 num_workers > 0 那么通过异步去读取，然后通过 to 拿到主GPU上
- 主GPU接收到完整 batch Tensor后自动 scatter 这个大 Tensor 到所有 GPU 上

### DDP example

对比 DP，DDP 的例子如下：
```python
my_trainset = torchvision.datasets.CIFAR10(root='./data', train=True)
# 新增1：使用DistributedSampler，DDP帮我们把细节都封装起来了。用，就完事儿！
#       sampler的原理，后面也会介绍。
train_sampler = torch.utils.data.distributed.DistributedSampler(my_trainset)
# 需要注意的是，这里的batch_size指的是每个进程下的batch_size。也就是说，总batch_size是这里的batch_size再乘以并行数(world_size)。
trainloader = torch.utils.data.DataLoader(my_trainset, batch_size=batch_size, sampler=train_sampler)


for epoch in range(num_epochs):
    # 新增2：设置sampler的epoch，DistributedSampler需要这个来维持各个进程之间的相同随机数种子
    trainloader.sampler.set_epoch(epoch)
    # 后面这部分，则与原来完全一致了。
    for data, label in trainloader:
        prediction = model(data)
        loss = loss_fn(prediction, label)
        loss.backward()
        optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
        optimizer.step()
```

这里由于 DDP 是多进程的，所以流程如下：
- 创建train_dataset：只是初始化 dataset 对象，没有涉及到具体读取的过程
- 创建采样器 train_sampler：DistributedSampler 根据当前进程的rank自动划分数据集，此时也没有开始读取数据只是生成了索引表
- 创建 dataloader，这时这里的 batch size 是这个进程的 batch size，或者直接记忆 DataLoader 的 batch size是针对进程而言的
- for epoch 里面，真正开始读取 trainloader 的时候每个 GPU 开始读他的 batch，这时 num_workers > 0 的话，多线程异步读取可以进一步进行加速

所以其实 DDP 可以理解为 DP 的 plus 版本，具体的训练思路都是差不多的

## DDP 实现
### 准备

首先，DDP 分发模型到不同的机器（进程上）需要我们了解模型的大致构成，一个 nn.module 的属性大概可以分为两组，主要掌握这四个东西：
- self.training：模型是否在训练状态
- self.\_modules：模型的下属模块，类似于迭代地定义了self.training、self.\_modules 等内容
- self.\_parameters：模型的参数，默认 require_gridiant=True
- self.\_buffers：模型的缓冲区，与参数不同的是默认 require_gridiant=False，通常用来保存不需要梯度，但又作为模型状态的一部分存储的张量

所以，在网络开始前向传递之前，由一个比如cuda：0节点会把模型的buffer广播给其他节点以维护状态的统一

### 实现思路

大体的实现思路是通过 hook 来实现的，也就是说，DDP的梯度更新策略可以理解为：给某个阶段的参数打上了 backward hook，那么当反向传播时，触发到这个参数的梯度更新时，就开始执行 RingAllReduce策略进行梯度的节点之间的更新操作

### 执行流程
#### 初始化阶段
1. 准备进程通信组，建立连接，注意如果参与链接的进程不够进程组的大小，那么所有进程会卡在这一步，一直到进程数够进程组的大小
2. DDP 初始化： model = DDP(model) 
	1. 把 parameters、buffers 从 master 节点传到其他节点（master节点其实不是特别合适因为 DDP 是去中心化的，但是意思是这么个意思）
	2. 如果一个节点有多张卡那么每张卡也创建模型，总而言之是以进程为单位的创建
	3. parameters 进行分组，每个组称为一个 bucket，临近的parameter会被分到同一个 bucket
		1. 为了加速，在梯度通讯时先计算，得到梯度的 bucket 会立即通讯，而不是等梯度算完再通信，算是 overlap
	4. 创建管理器 reducer，给每个 parameter 注册 hook
	5. 为 sync_batchnorm做准备


#### 训练阶段

![[DDP Train.png]]




# FSDP
