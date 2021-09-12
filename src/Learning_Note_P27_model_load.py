import torch
import torchvision
from torch import nn

# 用第一種方式保存模型後，怎麼加載模型
model1 = torch.load("vgg16_method1.pth")
#print(model1)

## 方式一容易犯錯的地方，會報錯，回報不知道Tudui這個class，須要在這裡重新宣告
model = torch.load('tudui_method1.pth')
print(model)
'''
## 總結方式一容易犯錯的地方：就是比較麻煩，要解決不知道class的問題，
## 須要在前面加上class，或是 import class 所在的 code file，
## 比如說：from Learning_Note_P27_model_save import *

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        return x
'''

'''
# 用第二種方式保存模型後，怎麼加載模型，為官方所推薦
model2 = torch.load("vgg16_method2.pth")
print(model2)
'''

vgg16 = torchvision.models.vgg16(pretrained = False)
vgg16.load_state_dict((torch.load("vgg16_method2.pth")))
#print(vgg16)