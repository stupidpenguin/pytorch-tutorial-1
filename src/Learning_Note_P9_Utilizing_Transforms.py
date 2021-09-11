import os
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# python的用法 -> tensor 數據類型
# 通過 transform.ToTensor去看兩個問題

# 2. 為什麼我們須要Tensor數據類型？
# -> 因為Tensor是特別設計用來方便深度學習訓練的數據類型，
# -> 比如使用Tensor可以讓我們更容易計算反向傳播，還有其他操作等等
root_dir = "C:/Users/woody/PycharmProjects/pytorch-tutorial-1"
image_dir = "data/dataset_ants&bees/ants/0013035.jpg"
img_path = os.path.join(root_dir, image_dir)
img = Image.open(img_path)

writer = SummaryWriter("logs")

# 1. transforms 該如何使用 (python)
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

# print(tensor_img)
print(tensor_img.shape)
print(tensor_img.type)

writer.add_image("Tensor_img", tensor_img)

writer.close()