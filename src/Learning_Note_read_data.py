import os
from PIL import Image
from torch.utils.data import Dataset

class MyData(Dataset):

    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)

    # 如何將資料夾內的圖片檔名，以字串資料型態的陣列儲存？
    '''
    首先 $import os
    並指定資料夾（相對/絕對）路徑 $dir_path = " ... "
    接著將資料夾內的路徑以陣列形式儲存 $img_path_list = os.listdir(dir_path)
    '''
    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path) #輸入此行後，便可成功獲取圖片的相關資訊
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)

root_dir = "data/dataset_ants&bees"
ants_label_dir = "ants"
bees_label_dir = "bees"
ants_dataset = MyData(root_dir, ants_label_dir)
bees_dataset = MyData(root_dir, bees_label_dir)

# 整理數據集的技巧
train_dataset = ants_dataset + bees_dataset