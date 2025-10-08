'''
Test a network that has been saved as a state dict. 
Should always test on all datasets.
Will create testing info tables to ease Dataset creation (if they do not already exist).
'''

# Imports
from PIL import Image
import os
import torch
import csv
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, utils
import torchvision.models as models
from torchvision.transforms import v2

import torch.nn as nn
import torch.nn.functional as F
import argparse
from datetime import datetime
import timm

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from diff_datasets import FaceDataset

## Import resnet34 with cbam head
# from resnet18cbam import ResNetWithCbamHead

# Import the cbam with intermediate cbam blocks
from Res_CBAM_FFT import ResNetCBAM, BasicBlockCBAM
def resnet34_cbam(num_classes, in_channels):
    # The layer configuration for ResNet-34 is [3, 4, 6, 3]
    resnet34_layer_config = [3, 4, 6, 3] 

    # 2. Instantiate the model as ResNet-34
    resnet34_model = ResNetCBAM(
        block=BasicBlockCBAM, 
        layers=resnet34_layer_config, 
        num_classes=num_classes,
        in_channels=in_channels
    )
    
    return resnet34_model

import sys
sys.path.insert(1, "F3Net")
from models import F3Net
from F3Net.models_modified import F3Net as F3NetMod

sys.path.insert(1, "DIRE")
from DIRE.utils.utils import get_network, str2bool, to_cuda
import torchvision.transforms.functional as TF

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-n', '--network_name')
parser.add_argument('-b', '--batch_size')
parser.add_argument('-t', '--network_type')
parser.add_argument('-fft', '--fft_mode', default='N')

args = parser.parse_args()

BATCH_SIZE = int(args.batch_size)
MODEL_NAME = args.network_type
NUM_CLASSES = 1
FFT_MODE = True if args.fft_mode == 'Y' else False
IN_CHANNELS = 4 if FFT_MODE else 3

# Define transform
transform = v2.Compose([
    v2.PILToTensor(),
    v2.Resize((256, 256)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# os.chdir("Desktop/diff/") # go to dataset directory

# Create datasets for all
# fe_test_dataset = RealFakeFaceDataset(root_dir="test", sub_dir="fe", transform=transform)
# fs_test_dataset = RealFakeFaceDataset(root_dir="test", sub_dir="fs", transform=transform)
# i2i_test_dataset = RealFakeFaceDataset(root_dir="test", sub_dir="i2i", transform=transform)
# t2i_test_dataset = RealFakeFaceDataset(root_dir="test", sub_dir="t2i", transform=transform)
fe_test_dataset = FaceDataset(root_dir="test", sub_dir="fe", transform=transform, use_fft=FFT_MODE)
fs_test_dataset = FaceDataset(root_dir="test", sub_dir="fs", transform=transform, use_fft=FFT_MODE)
i2i_test_dataset = FaceDataset(root_dir="test", sub_dir="i2i", transform=transform, use_fft=FFT_MODE)
t2i_test_dataset = FaceDataset(root_dir="test", sub_dir="t2i", transform=transform, use_fft=FFT_MODE)

fe_test_dataloader = DataLoader(
    fe_test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=4, 
    pin_memory=True)
fs_test_dataloader = DataLoader(
    fs_test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=4, 
    pin_memory=True)
i2i_test_dataloader = DataLoader(
    i2i_test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=4, 
    pin_memory=True)
t2i_test_dataloader = DataLoader(
    t2i_test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=4, 
    pin_memory=True)

dataloaders_dict = {
    "fe": fe_test_dataloader,
    "fs": fs_test_dataloader, 
    "i2i": i2i_test_dataloader, 
    "t2i": t2i_test_dataloader
}

model = None
# Load the network
if args.network_type == "resnet50":
    model = models.resnet50()
    model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
    # checkpoint = torch.load(f"model_state_dicts/{args.network_name}")
    model.load_state_dict(torch.load(f"model_state_dicts/{args.network_name}", weights_only=True))
elif args.network_type == "xception":
    model = timm.create_model('xception', num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(f"model_state_dicts/{args.network_name}", weights_only=True))
elif args.network_type == "effnet":
    model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(f"model_state_dicts/{args.network_name}", weights_only=True))
elif args.network_type == "f3net": 
    model = F3Net(img_width=256, img_height=256)
    model.load_state_dict(torch.load(f"model_state_dicts/{args.network_name}", weights_only=True))
elif args.network_type == "f3mod":
    model = F3NetMod(img_width=224, img_height=224)
    model.load_state_dict(torch.load(f"model_state_dicts/{args.network_name}", weights_only=True))
elif args.network_type == "dire":
    model = get_network("resnet50")
    model.load_state_dict(torch.load(f"model_state_dicts/{args.network_name}", weights_only=True))
    # dire_state_dict = torch.load(f"model_state_dicts/{args.network_name}")
    # if "model" in dire_state_dict:
    #     dire_state_dict = dire_state_dict["model"]
    # model.load_state_dict(dire_state_dict) 
elif args.network_type == "resnet_cbam":
    model = ResNetWithCbamHead(pretrained=True, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(f"model_state_dicts/{args.network_name}", weights_only=True))
elif args.network_type == "resnet_cbam_i":
    model = resnet34_cbam(NUM_CLASSES, IN_CHANNELS) # with intermediate cbam blocks
    model.load_state_dict(torch.load(f"model_state_dicts/{args.network_name}", weights_only=True))
elif args.network_type == "resnet34":
    model = models.resnet34()
    model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(f"model_state_dicts/{args.network_name}", weights_only=True))

model = model.to("cuda")
    


# model = net_dict[MODEL_NAME]

preds_targs_dict = {
    "fe": {},
    "fs": {},
    "i2i": {},
    "t2i": {}
}

# iterate over the different faking method types
for fake_type in preds_targs_dict.keys():
    predicted = np.array([])
    target = np.array([])
    print(f"processing data for {fake_type} dataloader")
    
    with torch.no_grad():
        for data in dataloaders_dict[fake_type]:
            
            images, labels = data["img_array"].to("cuda"), data["is_fake"].to("cuda")
            # calculate outputs by running images through the network

            if MODEL_NAME == "f3net" or MODEL_NAME == "f3mod": 
                outputs = torch.flatten(model(images)[1])    
            else:
                outputs = torch.flatten(model(images))
            
            # assuming the network predicts negative for false and positive for true
            binary_pred = torch.where(outputs > 0, 1, 0)
            binary_pred = np.array(binary_pred.cpu())
    
            predicted = np.append(predicted, binary_pred)
    
            labels = np.array(labels.cpu())
            target = np.append(target, labels)

            # Try just printing out the classification report in mid-stream
            

    preds_targs_dict[fake_type]["predicted"] = predicted
    preds_targs_dict[fake_type]["target"] = target
    print(classification_report(target, predicted))

with open(f"test_results/{args.network_name}_test_results.txt", "w") as f:
    f.write(f"MODEL NAME: {args.network_name}\n")
    f.write("\n")

    for fake_type in preds_targs_dict.keys():
        f.write(f"{fake_type}: \n")

        target = preds_targs_dict[fake_type]["target"]
        predicted = preds_targs_dict[fake_type]["predicted"]
        

        accuracy = accuracy_score(target, predicted)
        precision = precision_score(target, predicted)
        recall = recall_score(target, predicted)
        f1score = f1_score(target, predicted)

        print(confusion_matrix(target, predicted))

        f.write(f"Accuracy: {accuracy} \n")
        f.write(f"Precision: {precision} \n")
        f.write(f"Recall: {recall} \n")
        f.write(f"F1: {f1score} \n")