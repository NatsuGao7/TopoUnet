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

        self.PI = [os.path.join(data_root, "PI", i.split(".")[0] + ".png") for i in img_names]
        
        # Check if image files exist
        for i in self.img_list:
            if not os.path.exists(i):
                raise FileNotFoundError(f"file {i} does not exist.")
        
        # Check if mask files exist
        for i in self.mask:
            if not os.path.exists(i):
                raise FileNotFoundError(f"file {i} does not exist.")

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert('RGB')
        mask = Image.open(self.mask[idx]).convert('L')
        mask = np.array(mask) / 255
        mask = Image.fromarray(mask)

        # Load PI data from .mat file and normalize its values
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
        # Unpack the batch into images, targets, and persistence images
        images, targets, pi = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        batched_pi = cat_list(pi, fill_value=0)  # Process persistence images similarly
        return batched_imgs, batched_targets, batched_pi


def cat_list(images, fill_value=0):
    # Get the maximum size for padding
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    
    # Pad images to the maximum size
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs
