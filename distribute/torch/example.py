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
    rank = dist.get_rank()          # 当前进程编号
    world_size = dist.get_world_size()  # 总进程数
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
