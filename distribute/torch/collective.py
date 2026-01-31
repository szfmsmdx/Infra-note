import torch
import torch.distributed as dist
import os

def main():
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    def print_msg(msg):
        if rank == 0:
            print(f"\n[Master] {msg}")

    # boardcast
    print_msg("Broadcast")
    data = torch.zeros(10, device=device)
    src = 2
    if rank == src:
        data[:] = rank
    dist.broadcast(data, src=src)
    print(f"Rank {rank} broadcast 后的数据: {data.tolist()}")
    dist.barrier()

    # all reduce
    print_msg("All Reduce")
    data = torch.tensor([float(rank)], device=device)
    dist.all_reduce(data, dist.ReduceOp.SUM)
    print(f"Rank {rank} all_reduce 后的和: {data.item()}")
    dist.barrier()

    # all gather
    print_msg("All Gather")
    input_tensor = torch.tensor([float(rank)], device=device)
    gather_list = [torch.zeros(1, device=device) for _ in range(world_size)]
    dist.all_gather(gather_list, input_tensor)
    print(f"Rank {rank} all_gather 后的结果: {[t.item() for t in gather_list]}")
    dist.barrier()

    # reduce scatter
    print_msg("Reduce Scatter")
    input_list = [
        torch.tensor([rank * 100 + i], device=device, dtype=torch.float32)
        for i in range(world_size)
    ]
    print(f"Rank {rank} reduce_scatter 前的片段: {input_list}")
    output_tensor = torch.zeros(1, device=device)
    dist.reduce_scatter(output_tensor, input_list, dist.ReduceOp.SUM)
    print(f"Rank {rank} reduce_scatter 后的片段: {output_tensor.item()}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()