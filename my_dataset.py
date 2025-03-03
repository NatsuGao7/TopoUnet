import os
import torch
from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
from scipy.io import loadmat


class DriveDataset(Dataset):
    def __init__(self, root: str, train: bool, transforms=None):
        super(DriveDataset, self).__init__()
        self.flag = "training" if train else "test"
        data_root = os.path.join(root, "DRIVE", self.flag)
        assert os.path.exists(data_root), f"path '{data_root}' does not exist."
        self.transforms = transforms
        img_names = [i for i in os.listdir(os.path.join(data_root, "images")) if i.endswith(".tif")]
        self.img_list = [os.path.join(data_root, "images", i) for i in img_names]
        self.manual = [os.path.join(data_root, "1st_manual", i.split("_")[0] + "_manual1.gif")
                       for i in img_names]

        self.pi = [os.path.join(data_root, 'PI', i.split(".")[0] + '_PI.mat') for i in img_names]

        # Check if manual files exist
        for i in self.manual:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exist.")

        self.roi_mask = [os.path.join(data_root, "mask", i.split("_")[0] + f"_{self.flag}_mask.gif")
                         for i in img_names]
        # Check if ROI mask files exist
        for i in self.roi_mask:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exist.")

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert('RGB')
        manual = Image.open(self.manual[idx]).convert('L')
        manual = np.array(manual) / 255
        roi_mask = Image.open(self.roi_mask[idx]).convert('L')
        roi_mask = 255 - np.array(roi_mask)
        mask = np.clip(manual + roi_mask, a_min=0, a_max=255)
        # Convert back to PIL format because the transforms expect PIL input
        mask = Image.fromarray(mask)
        PI_data = loadmat(self.pi[idx])['image']
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
