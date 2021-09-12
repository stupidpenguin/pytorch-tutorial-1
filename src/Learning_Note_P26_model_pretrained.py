import torchvision

# train_data = torchvision.datasets.ImageNet("./data_image_net", split="train", download=True,
#                                            transform=torchvision.transforms.ToTensor())
from torch import nn

vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)
print("ok")
# 觀看 vgg16 的模型結構
print(vgg16_true)

# 拿CIFAR10 上的資料 在 vgg16 的模型上跑
train_data = torchvision.datasets.CIFAR10('../data',
                                          train=True,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)
# 在現有模型中加上其他 layer 的方法
vgg16_true.add_module('add_linear', nn.Linear(1000, 10))

## 如果想要加在 vgg16 的 classifier 裡面
## vgg16_true.classifer.add_module('add_linear', nn.Linear(1000, 10))

print(vgg16_true)

# 示範修改模型中的某一層 layer， 修改 vgg16_false 中的第6號 layer
print(vgg16_false)
vgg16_false.classifier[6] = nn.Linear(4096,10)
print(vgg16_false)