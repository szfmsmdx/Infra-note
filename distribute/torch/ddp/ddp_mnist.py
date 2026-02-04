import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
import os

# 定义一个简单 CNN 模型
class TinyCNN(nn.Module):
    def __init__(self):
        super(TinyCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.fc1 = nn.Linear(32 * 26 * 26, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = x.view(-1, 32 * 26 * 26)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

def train():
    # 1. 初始化分布式环境
    dist.init_process_group(backend='nccl')  # NCCL for CUDA
    rank = dist.get_rank()  # 当前进程的 rank (0 ~ world_size-1)
    world_size = dist.get_world_size()  # 总进程数（GPU 数）
    device = torch.device(f"cuda:{rank}")  # 每个进程绑定一个 GPU

    if rank == 0:
        print(f"启动 DDP，world_size={world_size}")

    # 2. 准备数据
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST('/data3/szf/Infra-note/distribute/torch/data', train=True, download=True, transform=transform)
    
    # 使用 DistributedSampler 分担数据
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler, num_workers=2)  # batch_size 是 per-GPU

    # 3. 模型、优化器、损失
    model = TinyCNN().to(device)
    ddp_model = DDP(model, device_ids=[rank])  # 包装为 DDP
    optimizer = optim.Adam(ddp_model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # 4. 训练循环
    ddp_model.train()
    for epoch in range(2):  # 简单跑 2 个 epoch
        sampler.set_epoch(epoch)  # 确保 shuffle 一致
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = ddp_model(data)
            loss = criterion(output, target)
            loss.backward()  # 这里触发梯度同步
            optimizer.step()
            if rank == 0 and batch_idx % 10 == 0:  # 只在 rank 0 打印
                print(f"Epoch {epoch}, Batch {batch_idx}: Loss = {loss.item():.4f}")

    # 5. 清理
    dist.destroy_process_group()

if __name__ == "__main__":
    train()