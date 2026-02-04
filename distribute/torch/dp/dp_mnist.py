import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. 定义一个非常轻量的 CNN (适合 4GB 显存)
class TinyCNN(nn.Module):
    def __init__(self):
        super(TinyCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(32 * 5 * 5, 10)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def train():
    # 2. 设置单进程可见的卡（比如你有 4 张，我们用 0,1,2,3）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 准备数据 (MNIST)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST('/data3/szf/Infra-note/distribute/torch/data', train=True, download=True, transform=transform)
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

if __name__ == "__main__":
    train()