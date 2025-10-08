# # Brain Tumor MRI Classification
# 
# #  Xception-KNN Hybrid Model
# 
# ### Author: Ethan Gilles
# 
# ---
# 
# ## Importing libraries

import random
import torchvision
import torch
import torch.nn as nn
import timm
import torchvision.models as models
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import numpy as np
import argparse
from tqdm import tqdm

from torchvision.transforms import v2
from diff_datasets import FaceDataset


# ## Cuda drivers
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Parse command line arguments
parser = argparse.ArgumentParser(description='Xception-KNN Hybrid Model')
parser.add_argument('-t', '--type_modification', type=str, required=True, help='Type of modification for sub_dir')
parser.add_argument('-s', '--subset', type=int, default=None, help='Subset size for training data (optional)')
parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size for DataLoader (default: 32)')
parser.add_argument('-x', '--xception_model', type=str, required=True, help='Path to pretrained Xception model weights')
args = parser.parse_args()

BATCH_SIZE = args.batch_size
TYPE_MODIFICATION = args.type_modification
XCEPTION_MODEL_PATH = args.xception_model


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
    sub_dir=TYPE_MODIFICATION,
    transform=transform
)

# If subset is specified, select a random subset of the training data
subset_flag = False
if args.subset is not None:
    subset_flag = True
    indices = random.sample(range(len(train_dataset)), args.subset)
    train_dataset = Subset(train_dataset, indices)


test_transform = v2.Compose([
    v2.PILToTensor(),
    v2.Resize((256, 256)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

TEST_DATASET_TYPE_MODIFICATIONS = ['fe', 'fs', 'i2i', 't2i']

test_dataset_fe = FaceDataset(
    root_dir="test",
    sub_dir='fe',
    transform=test_transform
)
test_dataset_fs = FaceDataset(
    root_dir="test",
    sub_dir='fs',
    transform=test_transform
)
test_dataset_i2i = FaceDataset(
    root_dir="test",
    sub_dir='i2i',
    transform=test_transform
)
test_dataset_t2i = FaceDataset(
    root_dir="test",
    sub_dir='t2i',
    transform=test_transform
)




# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Create test dataloaders for each dataset
test_loaders = {
    'fe': DataLoader(test_dataset_fe, batch_size=BATCH_SIZE, shuffle=False),
    'fs': DataLoader(test_dataset_fs, batch_size=BATCH_SIZE, shuffle=False),
    'i2i': DataLoader(test_dataset_i2i, batch_size=BATCH_SIZE, shuffle=False),
    't2i': DataLoader(test_dataset_t2i, batch_size=BATCH_SIZE, shuffle=False)
}

# # Create the Xception-KNN Hybrid
def create_feature_extractor():
    base_model = timm.create_model('xception', pretrained=True, num_classes=0)  # No final layer
    model = nn.Sequential(
        base_model,
        nn.Flatten(),
        nn.Dropout(0.3),
        nn.Linear(base_model.num_features, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        # Remove the final classification layer (softmax)
    )
    return model.to(device)

feature_extractor = create_feature_extractor()
feature_extractor.load_state_dict(torch.load(f"model_state_dicts/{XCEPTION_MODEL_PATH}"), strict=False)
feature_extractor.eval()


# ## K-Fold Cross Validation Training
# 
# 5 folds with a batch size of 32


# Hyperparameters
batch_size = 32
k_folds = 5
num_neighbors = 5
minkowski_p = 2

# Define the K-Fold Cross Validator
kfold = KFold(n_splits=k_folds, shuffle=True)

# Lists to store metrics for each fold
train_accuracies = []
val_accuracies = []
class_reports = []

for fold, (train_ids, val_ids) in enumerate(kfold.split(train_dataset)):
    print(f'FOLD {fold + 1}')
    print('--------------------------------')

    # Sample elements randomly from a given list of ids, no replacement.
    train_subsampler = Subset(train_dataset, train_ids)
    val_subsampler = Subset(train_dataset, val_ids)

    # Define data loaders for training and validation data in this fold
    train_loader = DataLoader(train_subsampler, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subsampler, batch_size=BATCH_SIZE, shuffle=False)

    print('Extracting training features...')
    train_features = []
    train_labels = []
    with torch.no_grad():
        for batch in tqdm(train_loader, desc=f'Fold {fold+1} Train Feature Extraction'):
            inputs = batch['img_array'].to(device)
            labels = batch['is_fake'].numpy()
            features = feature_extractor(inputs)
            train_features.append(features.cpu().numpy())
            train_labels.append(labels)
    train_features = np.concatenate(train_features, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)

    print('Extracting validation features...')
    val_features = []
    val_labels = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f'Fold {fold+1} Val Feature Extraction'):
            inputs = batch['img_array'].to(device)
            labels = batch['is_fake'].numpy()
            features = feature_extractor(inputs)
            val_features.append(features.cpu().numpy())
            val_labels.append(labels)
    val_features = np.concatenate(val_features, axis=0)
    val_labels = np.concatenate(val_labels, axis=0)

    print('Training KNN classifier...')
    knn = KNeighborsClassifier(n_neighbors=num_neighbors, p=minkowski_p)
    knn.fit(train_features, train_labels)

    # Compute training accuracy
    train_preds = knn.predict(train_features)
    train_accuracy = accuracy_score(train_labels, train_preds)
    train_accuracies.append(train_accuracy)
    
    # Compute validation accuracy
    val_preds = knn.predict(val_features)
    val_accuracy = accuracy_score(val_labels, val_preds)
    val_accuracies.append(val_accuracy)

    # Store classification report for this fold
    class_reports.append(classification_report(val_labels, val_preds, output_dict=True))

    print(f'Training Accuracy: {train_accuracy * 100:.2f}%')
    print(f'Validation Accuracy: {val_accuracy * 100:.2f}%')

# Compute macro and weighted averages
macro_accuracy_avg = np.mean(train_accuracies + val_accuracies)
# Use correct weights: number of samples per fold
fold_sizes = [len(train_ids) for train_ids, _ in kfold.split(train_dataset)] + [len(val_ids) for _, val_ids in kfold.split(train_dataset)]
weighted_accuracy_avg = np.average(train_accuracies + val_accuracies, weights=fold_sizes)

# Print final metrics
print('\nFinal Metrics:')
print(f'Training Accuracy (Avg): {np.mean(train_accuracies) * 100:.2f}%')
print(f'Validation Accuracy (Avg): {np.mean(val_accuracies) * 100:.2f}%')
print(f'Macro Accuracy Avg: {macro_accuracy_avg * 100:.2f}%')
print(f'Weighted Accuracy Avg: {weighted_accuracy_avg * 100:.2f}%')

print("\nClassification Report for Last Fold:")
print(classification_report(val_labels, val_preds))

print('Finished Cross-Validation')

# ## Retrain KNN on full training set for testing
print('Extracting features from full training set for final KNN...')
full_train_features = []
full_train_labels = []
feature_extractor.eval()
with torch.no_grad():
    for batch in tqdm(DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False), desc='Full Train Feature Extraction'):
        inputs = batch['img_array'].to(device)
        labels = batch['is_fake'].numpy()
        features = feature_extractor(inputs)
        full_train_features.append(features.cpu().numpy())
        full_train_labels.append(labels)
full_train_features = np.concatenate(full_train_features, axis=0)
full_train_labels = np.concatenate(full_train_labels, axis=0)

print('Training final KNN classifier on full training set...')
knn = KNeighborsClassifier(n_neighbors=num_neighbors, p=minkowski_p)
knn.fit(full_train_features, full_train_labels)

# Evaluate on each test dataset separately
test_accuracies = []
dataset_names = ['fe', 'fs', 'i2i', 't2i']

print('Evaluating on each test dataset...')
for dataset_name in dataset_names:
    print(f'\nEvaluating on {dataset_name} dataset...')
    
    # Extract features for current test dataset
    test_features = []
    test_labels = []
    feature_extractor.eval()
    with torch.no_grad():
        for batch in tqdm(test_loaders[dataset_name], desc=f'{dataset_name} Feature Extraction'):
            inputs = batch['img_array'].to(device)
            labels = batch['is_fake'].numpy()
            features = feature_extractor(inputs)
            test_features.append(features.cpu().numpy())
            test_labels.append(labels)

    test_features = np.concatenate(test_features, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)

    # Make predictions
    test_preds = knn.predict(test_features)
    
    # Calculate accuracy
    test_accuracy = accuracy_score(test_labels, test_preds)
    test_accuracies.append(test_accuracy)
    
    print(f'{dataset_name} Test Accuracy: {test_accuracy:.4f}')

# Print summary of all test accuracies
print('\n' + '='*50)
print('SUMMARY OF TEST ACCURACIES:')
print('='*50)
for i, dataset_name in enumerate(dataset_names):
    print(f'{dataset_name}: {test_accuracies[i]:.4f}')
print(f'Average Test Accuracy: {np.mean(test_accuracies):.4f}')

# Save results to file
import os
from datetime import datetime

# Create results directory if it doesn't exist
# results_dir = "/home/wyatt/Desktop/code/results"
# os.makedirs(results_dir, exist_ok=True)

# Generate filename based on model and training parameters
subset_str = f"_subset{args.subset}" if args.subset is not None else "_fulldata"
filename = f"XceptionKNN_{TYPE_MODIFICATION}{subset_str}_k{num_neighbors}_folds{k_folds}_batch{BATCH_SIZE}_{XCEPTION_MODEL_PATH.replace('.pth', '').replace('/', '_')}.txt"
filepath = os.path.join("./test_results", filename)

# Write results to file
with open(filepath, 'w') as f:
    f.write("="*60 + "\n")
    f.write("XCEPTION-KNN HYBRID MODEL RESULTS\n")
    f.write("="*60 + "\n")
    f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Model: Xception-KNN Hybrid\n")
    f.write(f"Xception Model Path: {XCEPTION_MODEL_PATH}\n")
    f.write(f"Training Dataset: {TYPE_MODIFICATION}\n")
    f.write(f"Subset Size: {args.subset if args.subset is not None else 'Full dataset'}\n")
    f.write(f"Batch Size: {BATCH_SIZE}\n")
    f.write(f"K-Fold Splits: {k_folds}\n")
    f.write(f"KNN Neighbors: {num_neighbors}\n")
    f.write(f"Minkowski p-value: {minkowski_p}\n")
    f.write("\n")
    
    f.write("CROSS-VALIDATION RESULTS:\n")
    f.write("-" * 30 + "\n")
    f.write(f"Training Accuracy (Avg): {np.mean(train_accuracies) * 100:.2f}%\n")
    f.write(f"Validation Accuracy (Avg): {np.mean(val_accuracies) * 100:.2f}%\n")
    f.write(f"Macro Accuracy Avg: {macro_accuracy_avg * 100:.2f}%\n")
    f.write(f"Weighted Accuracy Avg: {weighted_accuracy_avg * 100:.2f}%\n")
    f.write("\n")
    
    f.write("TEST DATASET ACCURACIES:\n")
    f.write("-" * 30 + "\n")
    for i, dataset_name in enumerate(dataset_names):
        f.write(f"{dataset_name}: {test_accuracies[i]:.4f}\n")
    f.write(f"Average Test Accuracy: {np.mean(test_accuracies):.4f}\n")
    f.write("\n")
    
    f.write("INDIVIDUAL FOLD RESULTS:\n")
    f.write("-" * 30 + "\n")
    for fold in range(k_folds):
        f.write(f"Fold {fold + 1}:\n")
        f.write(f"  Training Accuracy: {train_accuracies[fold] * 100:.2f}%\n")
        f.write(f"  Validation Accuracy: {val_accuracies[fold] * 100:.2f}%\n")
    f.write("\n")

print(f"\nResults saved to: {filepath}")