import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_set = torchvision.datasets.CIFAR10(root="./CIFAR10", train = True, transform=dataset_transform, download = True)
test_set = torchvision.datasets.CIFAR10(root="./CIFAR10", train = False, transform=dataset_transform, download = True)

# 轉換成 tensor 形式後，查看 tensor 圖片
# print(test_set[0])

'''
# 查看 test_set 中有哪些類別
print(test_set.classes)
img, target = test_set[0]
print(img)
print(target)
img.show() #還transform成Tensor的時候可以此命令查看圖片
'''

writer = SummaryWriter("P10")
for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_set", img, i)

writer.close()