from torch.optim.lr_scheduler import StepLR

EPOCH = 20
import torch
import torchvision.datasets
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("../src/CIFAR10", train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=1)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        # 如何計算 padding, stride 可參考官方文件中的公式
        self.model1 = nn.Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self,x):
        x = self.model1(x)
        return x

loss = nn.CrossEntropyLoss()
tudui = Tudui()

# 設置optimizer
optim = torch.optim.SGD(tudui.parameters(), lr=0.01)
scheduler = StepLR(optim, step_size=5, gamma=0.01)
for epoch in range(EPOCH):
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        outputs = tudui(imgs)
        result_loss = loss(outputs, targets)

        ## 將梯度重置為0
        optim.zero_grad()

        ## 反向傳播，求出梯度
        result_loss.backward()

        ## 調用優化器對每個參數進行更新
        ## 現在使用的優化器叫做 scheduler ，改優化器更新命令
        ## optim.step() -> scheduler.step()
        scheduler.step()

        running_loss = running_loss + result_loss
    print(running_loss)