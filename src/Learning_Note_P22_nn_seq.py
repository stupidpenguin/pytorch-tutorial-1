# 先寫網路模型
import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear

# 寫完網絡的時候建議檢查網絡中的參數是否正確，因為網絡中key錯參數並不會特別報錯
from torch.utils.tensorboard import SummaryWriter


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        # 如何計算 padding, stride 可參考官方文件中的公式

        ''' 不用 nn.Sequential 的作法，比較囉唆
        self.conv1 = Conv2d(3, 32, 5, padding=2)
        ## Max-pooling kernel size 為 2
        self.maxpool1 = MaxPool2d(2)
        self.conv2 = Conv2d(32, 32, 5, padding=2)
        self.maxpool2 = MaxPool2d(2)
        self.conv3 = Conv2d(32, 64, 5, padding=2)
        self.maxpool3 = MaxPool2d(2)
        self.flatten = Flatten()
        self.linear1 = Linear(1024, 64)
        self.linear2 = Linear(64, 10)
        '''

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
    ''' 不用 Sequential 的 forward，真的很囉唆
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x
    '''

    def forward(self,x):
        x = self.model1(x)
        return x

# 驗證網絡正確性
tudui = Tudui()
print(tudui)

input = torch.ones((64, 3, 32, 32))
output = tudui(input)
print(output.shape)

# 使用 tensorboard 做 visualization
writer = SummaryWriter("../logs_seq")
writer.add_graph(tudui, input)
writer.close()