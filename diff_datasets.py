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
    def __init__(self, root_dir, sub_dir, transform, use_fft=False):
        self.img_info_rows = load_csv(root_dir, sub_dir)
        self.transform = transform
        self.use_fft = use_fft
        self.normalize_rgb = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    def __len__(self):
        return len(self.img_info_rows)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()

        img_path = self.img_info_rows[idx][1]
        img = Image.open(img_path)
        img_tensor = self.transform(img)


        if self.use_fft:
            # 1. Create a grayscale version for FFT
            grayscale_tensor = v2.Grayscale()(img_tensor)

            # 2. Compute the 2D FFT, shift the zero-frequency component to the center
            fft_shifted = torch.fft.fftshift(torch.fft.fft2(grayscale_tensor))

            # 3. Calculate the log-magnitude spectrum for better dynamic range
            fft_magnitude = torch.log(torch.abs(fft_shifted) + 1e-8)
            
            # 4. Normalize the FFT magnitude to [0.0, 1.0]
            min_val = torch.min(fft_magnitude)
            max_val = torch.max(fft_magnitude)
            fft_normalized = (fft_magnitude - min_val) / (max_val - min_val + 1e-8)

            # 5. Normalize the original RGB channels
            rgb_normalized = self.normalize_rgb(img_tensor)
            
            # 6. Concatenate the normalized RGB with the normalized FFT magnitude
            final_tensor = torch.cat((rgb_normalized, fft_normalized), dim=0)
        else:
            # If not using FFT, just normalize the RGB tensor as usual
            final_tensor = self.normalize_rgb(img_tensor)

        is_fake = int(self.img_info_rows[idx][2])

        sample = {"img_array": final_tensor, "is_fake": is_fake}
        return sample