'''
Model initialization utilities for binary image classification.
Contains all model definitions and initialization logic.
'''

import torch
import torch.nn as nn
import torchvision.models as tmodels
import timm
import sys

# Add paths for custom models
sys.path.insert(1, "F3Net")
from F3Net.models import F3Net
from F3Net.models_modified import F3Net as F3NetMod

sys.path.insert(1, "DIRE")
from DIRE.utils.utils import get_network

# Import custom CBAM models
# from resnet18cbam import ResNetWithCbamHead
from Res_CBAM_FFT import ResNetCBAM, BasicBlockCBAM
# from Res_CBAM_BEiT_Hybrid import beit_resnet_cbam_hybrid, beit_resnet_cbam_hybrid_small


def initialize_resnet_cbam_intermediate(num_classes, num_channels, freeze_early=False):
    """Initialize ResNet with CBAM intermediate blocks."""
    # Use the architecture for ResNet-34 and 1000 classes to match ImageNet
    resnet_cbam_i = ResNetCBAM(BasicBlockCBAM, layers=[3, 4, 6, 3], num_classes=1000, in_channels=num_channels)

    # Load the Official Pretrained Model
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
    resnet_cbam_i.fc = nn.Linear(num_features, num_classes)

    if freeze_early:
        print("Early layer freezing in effect")
        # Freeze the early layers
        for name, param in resnet_cbam_i.named_parameters():
            # Freeze the stem and the first two blocks
            if name.startswith('conv1') or name.startswith('bn1') or \
               name.startswith('layer1') or name.startswith('layer2'):
                param.requires_grad = False
        
        # The final classifier should always be trainable for a new task
        for param in resnet_cbam_i.fc.parameters():
            param.requires_grad = True

    return resnet_cbam_i


def initialize_dire_model():
    """Initialize DIRE model with pretrained weights."""
    dire = get_network("resnet50", isTrain=True, continue_train=True)
    dire_state_dict = torch.load("DIRE/checkpoints/lsun_adm.pth")
    if "model" in dire_state_dict:
        dire_state_dict = dire_state_dict["model"]
    dire.load_state_dict(dire_state_dict)
    return dire


def get_model(model_name, num_classes=1, num_channels=3, freeze_early=False, fusion_method='concat'):
    """
    Get initialized model based on model name.
    
    Args:
        model_name: Name of the model architecture
        num_classes: Number of output classes
        num_channels: Number of input channels
        freeze_early: Whether to freeze early layers
        fusion_method: Fusion method for hybrid models ('concat', 'add', 'cross_attention')
    
    Returns:
        Initialized model
    """
    
    if model_name == 'resnet34':
        model = tmodels.resnet34(weights=tmodels.ResNet34_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == 'xception':
        model = timm.create_model('xception', pretrained=True, num_classes=num_classes)
        
    elif model_name == 'effnet':
        model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=num_classes)
        
    elif model_name == 'f3net':
        model = F3Net(img_width=256, img_height=256)

    elif model_name == 'f3mod':
        model = F3NetMod(img_width=224, img_height=224)
        
    elif model_name == 'dire':
        model = initialize_dire_model()
        
    elif model_name == 'resnet_cbam_i':
        model = initialize_resnet_cbam_intermediate(num_classes, num_channels, freeze_early)
        
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return model