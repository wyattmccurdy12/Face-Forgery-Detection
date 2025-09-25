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

from diff_datasets import FaceDataset

import sys
sys.path.insert(1, "F3Net")
from models import F3Net

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
parser.add_argument('-m', '--model_name', choices=['resnet50', 'xception', 'effnet', 'f3net'], required=True, help="Model architecture to use.")
args = parser.parse_args()

# Variable declarations
BATCH_SIZE = args.batch_size
EPOCHS = 50 # Set to run for 50 epochs
NUM_CLASSES = 1 # Single output for binary classification with BCEWithLogitsLoss
MODEL_NAME = args.model_name

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
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create the training dataset using the full dataset
train_dataset = FaceDataset(
    root_dir="train",
    sub_dir=args.type_modification,
    transform=transform
)

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
resnet = tmodels.resnet50(weights=tmodels.ResNet50_Weights.IMAGENET1K_V2)
resnet.fc = nn.Linear(resnet.fc.in_features, NUM_CLASSES)

xception = timm.create_model('xception', pretrained=True, num_classes=NUM_CLASSES)

effnet = timm.create_model('efficientnet_b0', pretrained=True, num_classes=NUM_CLASSES)

f3net = F3Net(img_width=256, img_height=256)

# Dictionary to select the model based on command-line argument
net_dict = {
    'resnet50': resnet,
    'xception': xception,
    'effnet': effnet,
    'f3net': f3net
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

    for epoch in range(EPOCHS):
        print(f"\nEPOCH {epoch + 1}/{EPOCHS}")
        f.write(f"EPOCH {epoch + 1}/{EPOCHS}\n")

        # --- Training Phase ---
        model.train() # Set model to training mode
        running_tloss = 0.0
        for i, data in enumerate(train_dataloader):
            inputs = data['img_array'].to(device)
            labels = data['is_fake'].float().to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs).squeeze() # Use squeeze instead of flatten for clarity

            # Calculate loss and backpropagate
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_tloss += loss.item()

        avg_tloss = running_tloss / len(train_dataloader)

        print(f"LOSS | Train: {avg_tloss:.4f}")
        f.write(f"LOSS | Train: {avg_tloss:.4f}\n")

        # --- Model Saving ---
        # Save a checkpoint periodically
        if (epoch + 1) % 10 == 0:
            chkpt_path = f'model_state_dicts/{MODEL_NAME}_{args.type_modification}_E{epoch + 1}.pth'
            torch.save(model.state_dict(), chkpt_path)
            print(f"Saved checkpoint to {chkpt_path}")

# Save the final model state after training is complete
final_model_path = f'model_state_dicts/final_{MODEL_NAME}_{args.type_modification}_E{EPOCHS}.pth'
torch.save(model.state_dict(), final_model_path)
print(f"\nSaved final model to {final_model_path}")

print("\n--- Training finished ---")