# https://www.youtube.com/watch?v=5vUdt5Bj_6w&list=PLgAyVnrNJ96CqYdjZ8v9YjQvCBcK5PZ-V&index=12
import os

from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image

writer = SummaryWriter("logs")

root_dir = "C:/Users/woody/PycharmProjects/pytorch-tutorial-1"
image_dir = "data/dataset_ants&bees/ants/0013035.jpg"
img_path = os.path.join(root_dir, image_dir)
img = Image.open(img_path)
print(img)

# 介紹如何使用 ToTensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("Transform Non-Tensor Image To Tensor", img_tensor)

# 介紹如何使用 Normalize
print(img_tensor[0][0][0])
## 隨意套用 Normalize 設定
'''
    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
    channels, this transform will normalize each channel of the input
    ``torch.*Tensor`` i.e.,
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``
'''
## trans_norm = transforms.Normalize([.5, .5, .5], [.5, .5, .5])
trans_norm = transforms.Normalize([1, 2, 3], [4, 5, 6])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Tensor after Normalized", img_norm, 1)

# 介紹如何使用 Resize
print(img.size)

## PIL 格式 image 經過 resize 後變成 512x512 (見設定) 之 PIL 格式 image
trans_resize = transforms.Resize((512,512))
img_resize = trans_resize(img)

## PIL 格式 image 經過 trans_totensor 之後，變成 tensor 格式 image
img_resize = trans_totensor(img_resize)
writer.add_image("Image after Resize & ToTensor", img_resize, 0)
print(img_resize)


# 介紹 Compose - resize - 2
trans_resize_2 = transforms.Resize(512)
## Compose() 中的參數須要是一個列表，在Python中列表的表示方法是[數據1, 數據2, ...]
## 所以我們會得到：Compose([transforms參數1, transforms參數2, ...])
trans_compose =  transforms.Compose([trans_resize_2, trans_totensor])
## 注意：如果是 trans_compose = transforms.Compose([trans_tensor, trans_resize]) 就會因為順序不對報錯
img_resize_2 = trans_compose(img)
writer.add_image("Resize", img_resize_2, 1)

# 介紹 RandomCrop 隨機裁減的使用
## 設定隨機裁減的尺寸
trans_random = transforms.RandomCrop((256,128))
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop", img_crop, i)


writer.close()