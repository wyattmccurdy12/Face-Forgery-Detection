'''
Train baseline networks for a binary image classification task.
This version uses the full dataset for training without a validation split.
'''
# ==============================================================================
# Imports
# ==============================================================================
import argparse
import torch
import torch.nn as nn
import torchvision.models as tmodels
import timm
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm
import random
import numpy as np

from diff_datasets import FaceDataset

import sys
sys.path.insert(1, "F3Net")
from F3Net.models import F3Net

sys.path.insert(1, "DIRE")
from DIRE.utils.utils import get_network, str2bool, to_cuda
import torchvision.transforms.functional as TF

## Import resnet34 with cbam head
# from resnet18cbam import ResNetWithCbamHead

# Import the cbam with intermediate cbam blocks
from Res_CBAM_FFT import ResNetCBAM, BasicBlockCBAM

# ==============================================================================
# Setup & Configuration
# ==============================================================================

# Determine the processing device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Parse command-line arguments for experiment configuration
parser = argparse.ArgumentParser(description="Train a binary image classifier.")
parser.add_argument('-t', '--type_modification', required=True, help="Sub-directory for the training data.")
parser.add_argument('-b', '--batch_size', type=int, default=32, help="Batch size for training.")
parser.add_argument('-m', '--model_name', required=True, help="Model architecture to use.")
parser.add_argument('-s', '--subset_size', type=int, default=None, help="Number of samples to use from the training set (None for full set).")
parser.add_argument('-f', '--freeze_early', default='N')
parser.add_argument('-fft', '--fft_mode', default='N')
args = parser.parse_args()

# Set random seeds for reproducibility
seed = 12
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Variable declarations
BATCH_SIZE = args.batch_size
EPOCHS = 50 # Set to run for 50 epochs
NUM_CLASSES = 1 # Single output for binary classification with BCEWithLogitsLoss
MODEL_NAME = args.model_name
FREEZE_EARLY = True if args.freeze_early == 'Y' else False
FFT_MODE = True if args.fft_mode == 'Y' else False
NUM_CHANNELS = 4 if FFT_MODE else 3

# ==============================================================================
# Data Loading & Transforms
# ==============================================================================

# Define image transformations for data augmentation and normalization
# Normalization values are standard for ImageNet-pretrained models
transform = v2.Compose([
    v2.PILToTensor(),
    v2.RandomResizedCrop(size=(256, 256), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True), # Scale pixel values to [0.0, 1.0]
    # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create the training dataset using the full dataset
train_dataset = FaceDataset(
    root_dir="train",
    sub_dir=args.type_modification,
    transform=transform,
    use_fft=FFT_MODE
)

if args.subset_size is not None:
    from torch.utils.data import Subset
    indices = list(range(len(train_dataset)))
    random.shuffle(indices)
    train_dataset = Subset(train_dataset, indices[:args.subset_size])
    print(f"Using subset of size {args.subset_size} for training.")
else:
    print(f"Using full dataset for training. Train size: {len(train_dataset)}")

# Create the DataLoader for training
train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True, # Shuffle training data each epoch
    num_workers=4,
    pin_memory=True
)

# ==============================================================================
# Model, Loss Function & Optimizer
# ==============================================================================

# Initialize models and modify the final layer for our binary task
# Use the modern 'weights' API for pretrained models
resnet = tmodels.resnet34(weights=tmodels.ResNet34_Weights.DEFAULT)
resnet.fc = nn.Linear(resnet.fc.in_features, NUM_CLASSES)

xception = timm.create_model('xception', pretrained=True, num_classes=NUM_CLASSES)

effnet = timm.create_model('efficientnet_b0', pretrained=True, num_classes=NUM_CLASSES)

f3net = F3Net(img_width=256, img_height=256)

dire = get_network("resnet50", isTrain=True, continue_train=True)
dire_state_dict = torch.load("DIRE/checkpoints/lsun_adm.pth")
if "model" in dire_state_dict:
    dire_state_dict = dire_state_dict["model"]
dire.load_state_dict(dire_state_dict)


########################### Initialize pretrained weights for resnet cbam intermetidate ##############

# Use the architecture for ResNet-34 and 1000 classes to match ImageNet
resnet_cbam_i = ResNetCBAM(BasicBlockCBAM, layers=[3, 4, 6, 3], num_classes=1000, in_channels=NUM_CHANNELS)

# --- Load the Official Pretrained Model ---
pretrained_model = tmodels.resnet34(weights=tmodels.ResNet34_Weights.DEFAULT)

# Get the state dictionaries
pretrained_state_dict = pretrained_model.state_dict()
custom_state_dict = resnet_cbam_i.state_dict()

# Create a new dictionary to hold the weights that will be loaded
new_state_dict = {}

for name, param in pretrained_state_dict.items():
    # Check if the layer exists in your custom model and the shapes match
    if name in custom_state_dict and param.shape == custom_state_dict[name].shape:
        new_state_dict[name] = param

# Load the new state dict into your custom model
# strict=False allows your new CBAM layers to remain uninitialized
resnet_cbam_i.load_state_dict(new_state_dict, strict=False)

for m in resnet_cbam_i.modules():
    if isinstance(m, BasicBlockCBAM):
        # Set the weight of the last BatchNorm layer in the block to 0
        nn.init.constant_(m.bn2.weight, 0)

num_features = resnet_cbam_i.fc.in_features
resnet_cbam_i.fc = nn.Linear(num_features, NUM_CLASSES)

if FREEZE_EARLY:
    print("Early layer freezing in effect")
    # --- Freeze the early layers ---
    for name, param in resnet_cbam_i.named_parameters():
        # Freeze the stem and the first two blocks
        if name.startswith('conv1') or name.startswith('bn1') or \
           name.startswith('layer1') or name.startswith('layer2'):
            param.requires_grad = False
    
    # --- The final classifier should always be trainable for a new task ---
    # (This is already handled by the logic above, but it's good practice to be explicit)
    for param in resnet_cbam_i.fc.parameters():
        param.requires_grad = True

########################### ^ Initialize pretrained weights for resnet cbam intermetidate ^ ##############

# Dictionary to select the model based on command-line argument
net_dict = {
    'resnet34': resnet,
    'xception': xception,
    'effnet': effnet,
    'f3net': f3net, 
    'dire': dire,
    'resnet_cbam_i': resnet_cbam_i
}

model = net_dict[MODEL_NAME].to(device)

# Loss function and optimizer
criterion = nn.BCEWithLogitsLoss() # Combines Sigmoid and BCELoss for numerical stability
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


# ==============================================================================
# Training Loop
# ==============================================================================

print(f"\n--- Starting training for {MODEL_NAME} ---")

# Open log file
with open(f"train_logs/{MODEL_NAME}_{args.type_modification}_log.txt", "w") as f:

    MIN_LOSS = float('inf')
    for epoch in range(EPOCHS):
        print(f"\nEPOCH {epoch + 1}/{EPOCHS}")
        f.write(f"EPOCH {epoch + 1}/{EPOCHS}\n")

        # --- Training Phase ---
        model.train() # Set model to training mode
        running_tloss = 0.0
        for data in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            inputs = data['img_array'].to(device)
            labels = data['is_fake'].float().to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            if MODEL_NAME == "f3net":
                outputs = model(inputs)[1].squeeze()
            else:
                outputs = model(inputs).squeeze() 

            # Calculate loss and backpropagate
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_tloss += loss.item()

        avg_tloss = running_tloss / len(train_dataloader)

        print(f"LOSS | Train: {avg_tloss:.4f}")
        f.write(f"LOSS | Train: {avg_tloss:.4f}\n")
        if avg_tloss < MIN_LOSS:
            MIN_LOSS = avg_tloss
            if FREEZE_EARLY and FFT_MODE:
                chkpt_path = f'model_state_dicts/{MODEL_NAME}_froz_fft_{args.type_modification}_best.pth'
            elif (FREEZE_EARLY or FFT_MODE):
                if FREEZE_EARLY:
                    chkpt_path = f'model_state_dicts/{MODEL_NAME}_froz_{args.type_modification}_best.pth'
                elif FFT_MODE:
                    chkpt_path = f'model_state_dicts/{MODEL_NAME}_fft_{args.type_modification}_best.pth'
            else:
                chkpt_path = f'model_state_dicts/{MODEL_NAME}_{args.type_modification}_best.pth'
                
            torch.save(model.state_dict(), chkpt_path)
            print(f"Saved checkpoint to {chkpt_path}")




print("\n--- Training finished ---")