import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def main():
    dist.init_process_group(
        backend="nccl",
        init_method="env://"
    )
    rank = dist.get_rank()
    world_size=  dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    tensor = torch.zeros(1, device="cuda")
    
    if rank == 0:
        tensor += 100
        print(f"send tensor device is {tensor.device}")
        print(tensor.item())
        dist.send(tensor, dst=1)

    if rank == 1:
        print(f"origin tensor is {tensor.item()} and device is {tensor.device}")
        dist.recv(tensor, src=0)
        print(f"current tensor is {tensor.item()}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()