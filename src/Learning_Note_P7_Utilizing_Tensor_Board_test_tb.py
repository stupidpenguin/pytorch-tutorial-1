from PIL import Image
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os

writer = SummaryWriter("logs")
root_dir = "C:/Users/woody/PycharmProjects/pytorch-tutorial-1"
image_dir = "data/Practicing_Dataset/train/bees_image/16838648_415acd9e3f.jpg"
image_path = os.path.join(root_dir, image_dir)
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)
print(type(img_array))
print(img_array.shape)

writer.add_image("test", img_array, 3, dataformats='HWC')
# y = x

for i in range(100):
    writer.add_scalar("y=2x", 2*i, i)

writer.close()

# 從 terminal 呼叫出 tensorboard 的方法 2021/09/11
# $python -m tensorboard.main --logdir=C:\Users\woody\PycharmProjects\pytorch-tutorial-1 --host=127.0.0.1