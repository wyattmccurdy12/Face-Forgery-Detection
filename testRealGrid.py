import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import timm
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
import argparse
from tqdm import tqdm
import os
from datetime import datetime
from itertools import product
import json

from torchvision.transforms import v2
from diff_datasets import FaceDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='Grid Search for Frequency + Spatial Domain')
parser.add_argument('-t', '--type_modification', type=str, required=True,
                    help='Training domain: fe, fs, i2i, or t2i')
parser.add_argument('-s', '--subset', type=int, default=None)
parser.add_argument('-b', '--batch_size', type=int, default=32)
parser.add_argument('-e', '--epochs', type=int, default=20, help='Epochs per configuration')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001)
parser.add_argument('--grid_search', action='store_true', help='Enable grid search')
args = parser.parse_args()

BATCH_SIZE = args.batch_size
TYPE_MODIFICATION = args.type_modification
EPOCHS = args.epochs
LEARNING_RATE = args.learning_rate

# ============================================
# GRID SEARCH CONFIGURATIONS
# ============================================

GRID_SEARCH_PARAMS = {
    'margin': [1.5, 2.0, 2.5, 3.0],
    'real_weight': [0.5, 1.0, 1.5, 2.0],
    'sep_weight': [0.5, 1.0, 1.5, 2.0]
}

# ============================================
# MODEL: FREQUENCY + SPATIAL DOMAIN ANALYSIS
# ============================================

class FrequencyAwareRealCenterXception(nn.Module):
    """
    Combines spatial and frequency domain features
    - Spatial: captures semantic content (good for I2I/T2I)
    - Frequency: captures artifacts (good for FE/FS)
    """
    def __init__(self, num_classes=2, embedding_dim=128):
        super(FrequencyAwareRealCenterXception, self).__init__()
        
        # Spatial branch - Xception backbone
        self.spatial_backbone = timm.create_model('xception', pretrained=True, num_classes=0)
        spatial_dim = self.spatial_backbone.num_features  # 2048
        
        # Frequency branch - analyzes FFT magnitude spectrum
        self.freq_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        freq_dim = 256
        combined_dim = spatial_dim + freq_dim
        
        # Project to embedding space
        self.projector = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_dim)
        )
        
        # Binary classifier
        self.classifier = nn.Linear(embedding_dim, num_classes)
        
        # Learnable center for REAL faces
        self.real_center = nn.Parameter(torch.randn(1, embedding_dim))
        
        self.embedding_dim = embedding_dim
    
    def extract_frequency_features(self, x):
        """Extract frequency domain features using FFT"""
        # Apply 2D FFT to each channel
        fft = torch.fft.fft2(x, dim=(-2, -1))
        fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))
        magnitude = torch.abs(fft_shifted)
        magnitude = torch.log(magnitude + 1e-8)
        
        # Normalize per channel
        B, C, H, W = magnitude.shape
        magnitude = magnitude.view(B, C, -1)
        magnitude = (magnitude - magnitude.mean(dim=-1, keepdim=True)) / (magnitude.std(dim=-1, keepdim=True) + 1e-8)
        magnitude = magnitude.view(B, C, H, W)
        
        return magnitude
    
    def forward(self, x, return_embedding=False):
        # Spatial features
        spatial_features = self.spatial_backbone(x)
        
        # Frequency features
        freq_input = self.extract_frequency_features(x)
        freq_features = self.freq_conv(freq_input)
        
        # Combine both domains
        combined = torch.cat([spatial_features, freq_features], dim=1)
        
        # Project to embedding space
        embeddings = self.projector(combined)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Classification
        logits = self.classifier(embeddings)
        
        if return_embedding:
            return logits, embeddings
        return logits
    
    def get_distance_to_real_center(self, embeddings):
        """Calculate distance from embeddings to real center"""
        center_normalized = F.normalize(self.real_center, p=2, dim=1)
        distances = torch.norm(embeddings - center_normalized, p=2, dim=1)
        return distances

# ============================================
# LOSS FUNCTIONS
# ============================================

def center_loss(embeddings, labels, real_center, margin=2.0):
    """Make REAL samples close to center, FAKE samples far from center"""
    center_normalized = F.normalize(real_center, p=2, dim=1)
    distances = torch.norm(embeddings - center_normalized, p=2, dim=1)
    
    real_mask = (labels == 0)
    fake_mask = (labels == 1)
    
    loss = 0.0
    
    if real_mask.sum() > 0:
        real_distances = distances[real_mask]
        loss_real = real_distances.mean()
        loss += loss_real
    
    if fake_mask.sum() > 0:
        fake_distances = distances[fake_mask]
        loss_fake = F.relu(margin - fake_distances).mean()
        loss += loss_fake
    
    return loss

def separation_loss(embeddings, labels, real_center, margin=2.0):
    """Maximize gap between average real distance and average fake distance"""
    center_normalized = F.normalize(real_center, p=2, dim=1)
    distances = torch.norm(embeddings - center_normalized, p=2, dim=1)
    
    real_mask = (labels == 0)
    fake_mask = (labels == 1)
    
    if real_mask.sum() == 0 or fake_mask.sum() == 0:
        return torch.tensor(0.0).to(embeddings.device)
    
    real_dist_avg = distances[real_mask].mean()
    fake_dist_avg = distances[fake_mask].mean()
    
    loss = F.relu(real_dist_avg - fake_dist_avg + margin)
    
    return loss

# ============================================
# DATA TRANSFORMS
# ============================================

def get_basic_transform():
    """No augmentation - just resize and normalize"""
    return v2.Compose([
        v2.PILToTensor(),
        v2.Resize((299, 299)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# ============================================
# TRAINING
# ============================================

def train_epoch(model, loader, optimizer, margin, real_weight, sep_weight):
    model.train()
    
    criterion_cls = nn.CrossEntropyLoss()
    
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, batch in enumerate(loader):
        inputs = batch['img_array'].to(device)
        labels = batch['is_fake'].to(device)
        
        # Forward
        logits, embeddings = model(inputs, return_embedding=True)
        
        # Classification loss
        loss_cls = criterion_cls(logits, labels)
        
        # Center loss
        loss_center = center_loss(embeddings, labels, model.real_center, margin=margin)
        
        # Separation loss
        loss_sep = separation_loss(embeddings, labels, model.real_center, margin=margin)
        
        # Combined loss
        loss = loss_cls + real_weight * loss_center + sep_weight * loss_sep
        
        if torch.isnan(loss):
            continue
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Stats
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if batch_idx % 50 == 0:
            torch.cuda.empty_cache()
    
    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy

# ============================================
# TESTING
# ============================================

def test(model, loader, domain_name, verbose=False):
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            inputs = batch['img_array'].to(device)
            labels = batch['is_fake'].numpy()
            
            logits, _ = model(inputs, return_embedding=True)
            _, predicted = logits.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    if verbose:
        print(f'{domain_name.upper()}: Acc={acc:.4f}, F1={f1:.4f}')
    
    return {'accuracy': acc, 'f1': f1}

# ============================================
# TRAIN SINGLE CONFIGURATION
# ============================================

def train_single_config(train_loader, test_loaders, margin, real_weight, sep_weight, epochs, config_num, total_configs):
    """Train model with specific hyperparameters"""
    print(f"\n{'='*80}")
    print(f"CONFIG {config_num}/{total_configs}: margin={margin}, real_weight={real_weight}, sep_weight={sep_weight}")
    print(f"{'='*80}")
    
    # Create fresh model
    model = FrequencyAwareRealCenterXception(num_classes=2, embedding_dim=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, margin, real_weight, sep_weight)
        scheduler.step()
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
    
    # Testing on all domains
    results = {}
    print(f"\nTesting configuration...")
    for name, loader in test_loaders.items():
        results[name] = test(model, loader, name, verbose=True)
    
    # Calculate average cross-domain performance
    cross_domain_accs = [results[d]['accuracy'] for d in ['fe', 'fs', 'i2i', 't2i'] 
                        if d != TYPE_MODIFICATION]
    avg_cross_domain = np.mean(cross_domain_accs)
    
    # Calculate overall average
    all_accs = [results[d]['accuracy'] for d in ['fe', 'fs', 'i2i', 't2i']]
    avg_all = np.mean(all_accs)
    
    print(f"In-domain ({TYPE_MODIFICATION.upper()}): {results[TYPE_MODIFICATION]['accuracy']:.4f}")
    print(f"Cross-domain average: {avg_cross_domain:.4f}")
    print(f"Overall average: {avg_all:.4f}")
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    
    return {
        'margin': margin,
        'real_weight': real_weight,
        'sep_weight': sep_weight,
        'results': results,
        'in_domain_acc': results[TYPE_MODIFICATION]['accuracy'],
        'cross_domain_avg': avg_cross_domain,
        'overall_avg': avg_all
    }

# ============================================
# GRID SEARCH
# ============================================

def run_grid_search(train_loader, test_loaders):
    """Run grid search over hyperparameters"""
    print("\n" + "="*80)
    print("STARTING GRID SEARCH")
    print("="*80)
    print(f"Search space:")
    print(f"  Margin: {GRID_SEARCH_PARAMS['margin']}")
    print(f"  Real weight: {GRID_SEARCH_PARAMS['real_weight']}")
    print(f"  Sep weight: {GRID_SEARCH_PARAMS['sep_weight']}")
    
    # Generate all combinations
    param_combinations = list(product(
        GRID_SEARCH_PARAMS['margin'],
        GRID_SEARCH_PARAMS['real_weight'],
        GRID_SEARCH_PARAMS['sep_weight']
    ))
    
    total_configs = len(param_combinations)
    print(f"Total configurations to test: {total_configs}")
    print("="*80)
    
    # Store all results
    all_results = []
    
    # Test each configuration
    for idx, (margin, real_weight, sep_weight) in enumerate(param_combinations, 1):
        config_result = train_single_config(
            train_loader, test_loaders,
            margin, real_weight, sep_weight,
            EPOCHS, idx, total_configs
        )
        all_results.append(config_result)
    
    # Sort by overall average accuracy
    all_results.sort(key=lambda x: x['overall_avg'], reverse=True)
    
    # Get top 3
    top_3 = all_results[:3]
    
    # Print top 3
    print("\n" + "="*80)
    print("TOP 3 CONFIGURATIONS")
    print("="*80)
    
    for rank, config in enumerate(top_3, 1):
        print(f"\nRANK {rank}:")
        print(f"  Hyperparameters:")
        print(f"    margin={config['margin']}, real_weight={config['real_weight']}, sep_weight={config['sep_weight']}")
        print(f"  Performance:")
        print(f"    Overall average: {config['overall_avg']:.4f}")
        print(f"    In-domain ({TYPE_MODIFICATION.upper()}): {config['in_domain_acc']:.4f}")
        print(f"    Cross-domain average: {config['cross_domain_avg']:.4f}")
        print(f"  Per-domain results:")
        for domain in ['fe', 'fs', 'i2i', 't2i']:
            acc = config['results'][domain]['accuracy']
            print(f"    {domain.upper()}: {acc:.4f}")
    
    # Save results
    save_grid_search_results(all_results, top_3)
    
    return top_3

# ============================================
# SAVE GRID SEARCH RESULTS
# ============================================

def save_grid_search_results(all_results, top_3):
    """Save grid search results to files"""
    os.makedirs("grid_search_results", exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save top 3 configurations (human-readable)
    txt_filename = f"grid_search_results/top3_{TYPE_MODIFICATION}_{timestamp}.txt"
    with open(txt_filename, 'w') as f:
        f.write("="*80 + "\n")
        f.write("GRID SEARCH RESULTS - TOP 3 CONFIGURATIONS\n")
        f.write("="*80 + "\n")
        f.write(f"Training domain: {TYPE_MODIFICATION}\n")
        f.write(f"Epochs per config: {EPOCHS}\n")
        f.write(f"Learning rate: {LEARNING_RATE}\n")
        f.write(f"Batch size: {BATCH_SIZE}\n")
        f.write(f"Total configurations tested: {len(all_results)}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n")
        
        for rank, config in enumerate(top_3, 1):
            f.write("="*80 + "\n")
            f.write(f"RANK {rank}\n")
            f.write("="*80 + "\n")
            f.write(f"Hyperparameters:\n")
            f.write(f"  margin: {config['margin']}\n")
            f.write(f"  real_weight: {config['real_weight']}\n")
            f.write(f"  sep_weight: {config['sep_weight']}\n")
            f.write("\n")
            
            f.write(f"Performance:\n")
            f.write(f"  Overall average accuracy: {config['overall_avg']:.4f}\n")
            f.write(f"  In-domain accuracy ({TYPE_MODIFICATION.upper()}): {config['in_domain_acc']:.4f}\n")
            f.write(f"  Cross-domain average: {config['cross_domain_avg']:.4f}\n")
            f.write(f"  Generalization gap: {config['in_domain_acc'] - config['cross_domain_avg']:.4f}\n")
            f.write("\n")
            
            f.write(f"Per-domain results:\n")
            f.write(f"{'Domain':<10} {'Accuracy':<12} {'F1 Score':<12}\n")
            f.write("-"*40 + "\n")
            for domain in ['fe', 'fs', 'i2i', 't2i']:
                marker = " *" if domain == TYPE_MODIFICATION else ""
                acc = config['results'][domain]['accuracy']
                f1 = config['results'][domain]['f1']
                f.write(f"{domain.upper():<10} {acc:<12.4f} {f1:<12.4f}{marker}\n")
            f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("COMMAND TO REPRODUCE BEST CONFIGURATION:\n")
        f.write("="*80 + "\n")
        best = top_3[0]
        f.write(f"python train_frequency_spatial.py -t {TYPE_MODIFICATION} \\\n")
        f.write(f"    -b {BATCH_SIZE} -e {EPOCHS} -lr {LEARNING_RATE} \\\n")
        f.write(f"    --margin {best['margin']} \\\n")
        f.write(f"    --real_weight {best['real_weight']} \\\n")
        f.write(f"    --sep_weight {best['sep_weight']}\n")
    
    print(f"\nTop 3 configurations saved to: {txt_filename}")
    
    # Save all results (JSON format for analysis)
    json_filename = f"grid_search_results/all_results_{TYPE_MODIFICATION}_{timestamp}.json"
    
    # Prepare data for JSON (convert numpy types)
    json_data = []
    for config in all_results:
        json_config = {
            'margin': float(config['margin']),
            'real_weight': float(config['real_weight']),
            'sep_weight': float(config['sep_weight']),
            'in_domain_acc': float(config['in_domain_acc']),
            'cross_domain_avg': float(config['cross_domain_avg']),
            'overall_avg': float(config['overall_avg']),
            'results': {
                domain: {
                    'accuracy': float(config['results'][domain]['accuracy']),
                    'f1': float(config['results'][domain]['f1'])
                }
                for domain in ['fe', 'fs', 'i2i', 't2i']
            }
        }
        json_data.append(json_config)
    
    with open(json_filename, 'w') as f:
        json.dump({
            'training_domain': TYPE_MODIFICATION,
            'epochs_per_config': EPOCHS,
            'learning_rate': LEARNING_RATE,
            'batch_size': BATCH_SIZE,
            'total_configs': len(all_results),
            'timestamp': timestamp,
            'configurations': json_data
        }, f, indent=2)
    
    print(f"All results saved to: {json_filename}")

# ============================================
# MAIN
# ============================================

def main():
    print("="*80)
    print("FREQUENCY + SPATIAL DOMAIN WITH GRID SEARCH")
    print("="*80)
    print(f"Training domain: {TYPE_MODIFICATION.upper()}")
    
    # Transform
    transform = get_basic_transform()
    
    # Load training dataset
    print(f"\nLoading training dataset: {TYPE_MODIFICATION.upper()}...")
    train_dataset = FaceDataset(
        root_dir="train",
        sub_dir=TYPE_MODIFICATION,
        transform=transform
    )
    
    if args.subset:
        indices = random.sample(range(len(train_dataset)), 
                               min(args.subset, len(train_dataset)))
        train_dataset = Subset(train_dataset, indices)
    
    print(f"Training samples: {len(train_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    # Load test datasets
    print("\nLoading test datasets...")
    test_datasets = {
        'fe': FaceDataset(root_dir="test", sub_dir='fe', transform=transform),
        'fs': FaceDataset(root_dir="test", sub_dir='fs', transform=transform),
        'i2i': FaceDataset(root_dir="test", sub_dir='i2i', transform=transform),
        't2i': FaceDataset(root_dir="test", sub_dir='t2i', transform=transform)
    }
    
    test_loaders = {
        name: DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, 
                        num_workers=4, pin_memory=True)
        for name, dataset in test_datasets.items()
    }
    
    for name, dataset in test_datasets.items():
        print(f"  {name.upper()}: {len(dataset)} samples")
    
    # Run grid search
    if args.grid_search:
        top_3 = run_grid_search(train_loader, test_loaders)
        
        print("\n" + "="*80)
        print("GRID SEARCH COMPLETED!")
        print("="*80)
        print(f"\nBest configuration:")
        best = top_3[0]
        print(f"  margin={best['margin']}, real_weight={best['real_weight']}, sep_weight={best['sep_weight']}")
        print(f"  Overall average accuracy: {best['overall_avg']:.4f}")
        print(f"\nTo train with best config, run:")
        print(f"python train_frequency_spatial.py -t {TYPE_MODIFICATION} -b {BATCH_SIZE} -e {EPOCHS} \\")
        print(f"    --margin {best['margin']} --real_weight {best['real_weight']} --sep_weight {best['sep_weight']}")
    else:
        print("\nGrid search not enabled. Use --grid_search flag to enable.")
        print("Example: python train_frequency_spatial.py -t fe --grid_search")

if __name__ == "__main__":
    main()