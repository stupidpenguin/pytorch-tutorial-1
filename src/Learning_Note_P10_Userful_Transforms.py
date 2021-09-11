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
writer.add_image("ToTensr", img_tensor)
writer.close()