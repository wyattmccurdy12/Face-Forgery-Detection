'''This is the script where I have an updated dataloader'''
from PIL import Image
import os
import torch
import csv
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, utils
from torchvision.transforms import v2
import torch.nn as nn
import torch.nn.functional as F
import argparse
from diff_utils import load_csv

class FaceDataset(Dataset):
    '''Provides dataloader with a zero-indexed dataset of image-label pairs'''
    def __init__(self, root_dir, sub_dir, transform):
        self.img_info_rows = load_csv(root_dir, sub_dir)
        self.transform = transform

    def __len__(self):
        return len(self.img_info_rows)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()

        img_path = self.img_info_rows[idx][1]
        img = Image.open(img_path)
        img = self.transform(img)

        is_fake = int(self.img_info_rows[idx][2])

        sample = {"img_array": img, "is_fake": is_fake}
        return sample