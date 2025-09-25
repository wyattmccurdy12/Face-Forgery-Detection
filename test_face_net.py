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
from fakeface_data_utils import RealFakeFaceDataset, create_or_load_csv
from diff_datasets import FaceDataset

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-n', '--network_name')
parser.add_argument('-b', '--batch_size')
parser.add_argument('-t', '--network_type')

args = parser.parse_args()

BATCH_SIZE = int(args.batch_size)
MODEL_NAME = args.network_type
NUM_CLASSES = 1

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
fe_test_dataset = FaceDataset(root_dir="test", sub_dir="fe", transform=transform)
fs_test_dataset = FaceDataset(root_dir="test", sub_dir="fs", transform=transform)
i2i_test_dataset = FaceDataset(root_dir="test", sub_dir="i2i", transform=transform)
t2i_test_dataset = FaceDataset(root_dir="test", sub_dir="t2i", transform=transform)

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
    checkpoint = torch.load(f"model_state_dicts/{args.network_name}")
    model.load_state_dict(torch.load(f"model_state_dicts/{args.network_name}", weights_only=True))
elif args.network_type == "xception":
    model = timm.create_model('xception', num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(f"model_state_dicts/{args.network_name}", weights_only=True))
elif args.network_type == "effnet":
    model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(f"model_state_dicts/{args.network_name}", weights_only=True))

model = model.to("cuda")
    
    
# resnet = models.resnet50()
# resnet.fc = torch.nn.Linear(resnet.fc.in_features, NUM_CLASSES)

# checkpoint = torch.load(f"model_state_dicts/{args.network_name}")
# state_dict = checkpoint['state_dict']

# resnet.load_state_dict(state_dict, weights_only=True)
# resnet = torch.load(f"model_state_dicts/{args.network_name}")

# ## NETWORK
# net_dict = {
#     'simple_net': net.to("cuda"),
#     'resnet50': resnet
# }

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