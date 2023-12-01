import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler


def get_loaders(world_size, rank, batch_size, split_ratio=0.8):
    full_dataset = Dataset_Class()
    train_size = int(split_ratio * len(full_dataset))
    validation_size = len(full_dataset) - train_size
    train_dataset, validation_dataset = random_split(full_dataset, [train_size, validation_size])

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)

    validation_sampler = DistributedSampler(validation_dataset, num_replicas=world_size, rank=rank)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, sampler=validation_sampler)

    return train_loader, validation_loader

class Dataset_Class(Dataset):
    def __init__(self, img_dir, gt_dir, transform=None):
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
