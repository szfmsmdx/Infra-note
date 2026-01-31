>参考文章：
>[(6 封私信) PyTorch分布式训练基础：掌握torch.distributed及其通信功能 - 知乎](https://zhuanlan.zhihu.com/p/692668388)
>[分布式通信包 - torch.distributed — PyTorch 2.9 文档 - PyTorch 文档](https://docs.pytorch.ac.cn/docs/stable/distributed.html#backends)
>

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
4. 在每张卡上复制一份完整的模型（是的，模型被复制了！）
5. 每张卡独立做：
    - 前向传播 → 得到 loss
    - 反向传播 → 计算梯度
6. 将所有卡上的梯度 gather 到主卡（cuda:0）
7. 在主卡上求梯度平均值
8. 用平均梯度更新主卡上的模型参数
9. 把更新后的模型参数 broadcast 回其他 GPU

DP 到其他卡上的相当于是工作副本，被 DP 包裹的 model 在计算损失的时候汇总到主卡，并 backward 的时候会回传梯度求平均，然后分发到副卡更新权重，最终的model也是取主卡上的

# DDP

# FSDP
