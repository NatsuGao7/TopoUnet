import os
import torch
from scipy.io import loadmat
from PIL import Image
import numpy as np
from torch.utils.data import Dataset


class ERDataset(Dataset):
    def __init__(self, root: str, train: bool, val: bool = False, transforms=None):
        super(ERDataset, self).__init__()
        if train:
            self.flag = "train"
        elif val:
            self.flag = "val"
        else:
            self.flag = "test" 
        
        # Data root path
        data_root = os.path.join(root, "ER", self.flag)
        assert os.path.exists(data_root), f"path '{data_root}' does not exist."
        
        self.transforms = transforms
        # Collect image and mask paths
        img_names = [i for i in os.listdir(os.path.join(data_root, "images")) if i.endswith(".tif")]
        self.img_list = [os.path.join(data_root, "images", i) for i in img_names]
        self.mask = [os.path.join(data_root, "masks", i) for i in img_names]

        self.PI = [os.path.join(data_root,"PI",i.split(".")[0]+".png") for i in img_names]
        
        # Check if the files exist
        for i in self.img_list:
            if not os.path.exists(i):
                raise FileNotFoundError(f"file {i} does not exist.")
        
        # Check mask files
        for i in self.mask:
            if not os.path.exists(i):
                raise FileNotFoundError(f"file {i} does not exist.")

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert('RGB')
        mask = Image.open(self.mask[idx]).convert('L')
        mask = np.array(mask) / 255
        mask = Image.fromarray(mask)

        PI_data = loadmat(self.PI[idx])['image']
        PI_data[PI_data < 0] = 0
        PI_data[PI_data > 1] = 1
        PI_data = torch.tensor(PI_data, dtype=torch.float32)
        

        
        
        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask, PI_data

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def collate_fn(batch):
        images, targets, pi = list(zip(*batch))  # 解包为三项，分别是图像、目标和持久性图像
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        batched_pi = cat_list(pi, fill_value=0)  # 对持久性图像进行类似处理
        return batched_imgs, batched_targets, batched_pi


def cat_list(images, fill_value=0):
    # Get the max size for padding
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    
    # Pad images to the max size
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs
