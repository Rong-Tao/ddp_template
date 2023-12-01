import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

class Dataset_Class(Dataset):
    def __init__(self, img_dir, gt_dir, transform=None):
        """
        Args:
            img_dir (string): Directory with all the imgs (.npy files).
            gt_dir (string): Directory with all the gts (.npy files).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith('.npy')]
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.img_files[idx])
        img = np.load(img_name)
        img = torch.from_numpy(img).float()

        # Corresponding gt file
        gt_name = os.path.join(self.gt_dir, self.img_files[idx])
        gt = np.load(gt_name)
        gt = torch.from_numpy(gt).long() # or `.float()` depending on your gt nature

        if self.transform:
            img = self.transform(img)

        return img, gt
