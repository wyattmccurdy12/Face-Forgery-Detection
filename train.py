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
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm
import random
import numpy as np

from diff_datasets import FaceDataset
from model_init import get_model

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
# parser.add_argument('--fusion_method', default='concat', choices=['concat', 'add', 'cross_attention'], help="Fusion method for hybrid models")
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
if MODEL_NAME == "f3mod": # for the swin transformer
    
    transform = v2.Compose([
        v2.PILToTensor(),
        v2.RandomResizedCrop(size=(224, 224), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True), # Scale pixel values to [0.0, 1.0]
        # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
else: 
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

# Initialize the model using the model_init module
model = get_model(MODEL_NAME, NUM_CLASSES, NUM_CHANNELS, FREEZE_EARLY).to(device)

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
            if MODEL_NAME == "f3net" or MODEL_NAME == "f3mod":
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
            
            # Build checkpoint filename
            filename_parts = [MODEL_NAME]
            
            # Add fusion method for hybrid models
            if 'hybrid' in MODEL_NAME and args.fusion_method != 'concat':
                filename_parts.append(args.fusion_method)
            
            # Add modifiers
            if FREEZE_EARLY:
                filename_parts.append('froz')
            if FFT_MODE:
                filename_parts.append('fft')
                
            filename_parts.extend([args.type_modification, 'best'])
            chkpt_path = f"model_state_dicts/{'_'.join(filename_parts)}.pth"
                
            torch.save(model.state_dict(), chkpt_path)
            print(f"Saved checkpoint to {chkpt_path}")




print("\n--- Training finished ---")