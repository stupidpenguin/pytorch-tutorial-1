import torch
import torchvision
from torch import nn

vgg16 = torchvision.models.vgg16(pretrained=False)

# 第一種模型保存方式，保存模型結構與模型參數
torch.save(vgg16, "vgg16_method1.pth")

## 方式一容易犯錯的地方
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        return x
tudui = Tudui()
torch.save(tudui, "tudui_method1.pth")


# 第二種模型保存方式，保存模型參數，為官方所推薦
##　把 vgg16 的狀態以 dictionary 形式保存
torch.save(vgg16.state_dict(), "vgg16_method2.pth")