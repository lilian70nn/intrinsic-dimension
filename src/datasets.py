"""
Dataset utilities used across image and tabular experiments.

This module defines lightweight Dataset wrappers for:
- Image datasets specified by explicit file paths
- Tabular datasets with paired numerical and categorical features
"""


from torch.utils.data import Dataset
from PIL import Image

class PathListDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        dummy_label = 0
        return img, dummy_label
    

class TabularPairDataset(Dataset):
    def __init__(self, x_num, x_cat, y):
        self.x_num = x_num
        self.x_cat = x_cat
        self.y = y

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return (self.x_num[idx], self.x_cat[idx]), self.y[idx]