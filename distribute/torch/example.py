import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# # 自己用 mp 来创建进程，直接 python 启动即可
# def worker(rank, world_size):
#     dist.init_process_group(
#         backend="nccl",
#         init_method="tcp://localhost:29501",
#         rank=rank,
#         world_size=world_size
#     )
#     # 一个进程一张卡
#     torch.cuda.set_device(rank)

#     x = torch.tensor([rank], device="cuda", dtype=torch.float32)
#     dist.all_reduce(x, op=dist.ReduceOp.SUM)
#     dist.barrier()  # 进程同步
#     if rank == 0:
#         print(f"world_size = {world_size}")
#     print(f"rank {rank}: after all_reduce, x = {x.item()}")

#     dist.destroy_process_group()
    
# if __name__ == "__main__":
#     world_size = 4
#     mp.set_start_method("spawn", force=True)
#     mp.spawn(
#         worker, args=(world_size,), # 等价于 args=(local_rank, *args)，第一个参数强制传入local_rank
#         nprocs=4, 
#         join=True
#     )

# 用 torchrun 这个 launcher 来自动启动
def main():
    dist.init_process_group(
        backend="nccl",
        init_method="env://"
    )
    rank = dist.get_rank()  # 这个是全局进程编号，相对于分布式作业来说
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])  # 这个是本机 GPU 编号，在当前机器内唯一
    torch.cuda.set_device(local_rank)
    x = torch.tensor([rank], dtype=torch.float32, device="cuda")
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    dist.barrier()
    if rank == 0:
        print(f"world_size = {world_size}")
    print(f"rank {rank}: after all reduce, x = {x.item()}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()