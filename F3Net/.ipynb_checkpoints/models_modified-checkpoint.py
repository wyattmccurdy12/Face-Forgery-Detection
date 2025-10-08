import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import types
import timm  # <<< Added timm for Swin Transformer

# =================================================================================
# FAD, LFS, and Filter modules remain unchanged.
# They are included here for completeness.
# =================================================================================

# Filter Module
class Filter(nn.Module):
    def __init__(self, size, band_start, band_end, use_learnable=True, norm=False):
        super(Filter, self).__init__()
        self.use_learnable = use_learnable

        self.base = nn.Parameter(torch.tensor(generate_filter(band_start, band_end, size)), requires_grad=False)
        if self.use_learnable:
            self.learnable = nn.Parameter(torch.randn(size, size), requires_grad=True)
            self.learnable.data.normal_(0., 0.1)

        self.norm = norm
        if norm:
            self.ft_num = nn.Parameter(torch.sum(torch.tensor(generate_filter(band_start, band_end, size))), requires_grad=False)


    def forward(self, x):
        if self.use_learnable:
            filt = self.base + norm_sigma(self.learnable)
        else:
            filt = self.base

        if self.norm:
            y = x * filt / self.ft_num
        else:
            y = x * filt
        return y


# FAD Module
class FAD_Head(nn.Module):
    def __init__(self, size):
        super(FAD_Head, self).__init__()
        self._DCT_all = nn.Parameter(torch.tensor(DCT_mat(size)).float(), requires_grad=False)
        self._DCT_all_T = nn.Parameter(torch.transpose(torch.tensor(DCT_mat(size)).float(), 0, 1), requires_grad=False)
        low_filter = Filter(size, 0, size // 2.82)
        middle_filter = Filter(size, size // 2.82, size // 2)
        high_filter = Filter(size, size // 2, size * 2)
        all_filter = Filter(size, 0, size * 2)
        self.filters = nn.ModuleList([low_filter, middle_filter, high_filter, all_filter])

    def forward(self, x):
        x_freq = self._DCT_all @ x @ self._DCT_all_T
        y_list = []
        for i in range(4):
            x_pass = self.filters[i](x_freq)
            y = self._DCT_all_T @ x_pass @ self._DCT_all
            y_list.append(y)
        out = torch.cat(y_list, dim=1)
        return out

# LFS Module
class LFS_Head(nn.Module):
    def __init__(self, size, window_size, M):
        super(LFS_Head, self).__init__()
        self.window_size = window_size
        self._M = M
        self._DCT_patch = nn.Parameter(torch.tensor(DCT_mat(window_size)).float(), requires_grad=False)
        self._DCT_patch_T = nn.Parameter(torch.transpose(torch.tensor(DCT_mat(window_size)).float(), 0, 1), requires_grad=False)
        self.unfold = nn.Unfold(kernel_size=(window_size, window_size), stride=2, padding=4)
        self.filters = nn.ModuleList([Filter(window_size, window_size * 2. / M * i, window_size * 2. / M * (i+1), norm=True) for i in range(M)])
    
    def forward(self, x):
        x_gray = 0.299*x[:,0,:,:] + 0.587*x[:,1,:,:] + 0.114*x[:,2,:,:]
        x = x_gray.unsqueeze(1)
        x = (x + 1.) * 122.5
        N, C, W, H = x.size()
        S = self.window_size
        size_after = int((W - S + 8)/2) + 1
        assert size_after == 149
        x_unfold = self.unfold(x)
        L = x_unfold.size()[2]
        x_unfold = x_unfold.transpose(1, 2).reshape(N, L, C, S, S)
        x_dct = self._DCT_patch @ x_unfold @ self._DCT_patch_T
        y_list = []
        for i in range(self._M):
            y = torch.abs(x_dct)
            y = torch.log10(y + 1e-15)
            y = self.filters[i](y)
            y = torch.sum(y, dim=[2,3,4])
            y = y.reshape(N, size_after, size_after).unsqueeze(dim=1)
            y_list.append(y)
        out = torch.cat(y_list, dim=1)
        return out


# =================================================================================
# F3Net is modified to use Swin Transformer
# =================================================================================
class F3Net(nn.Module):
    def __init__(self, num_classes=1, img_width=299, img_height=299, LFS_window_size=10, LFS_M=6, mode='FAD', device=None):
        super(F3Net, self).__init__()
        assert img_width == img_height
        img_size = img_width
        self.num_classes = num_classes
        self.mode = mode
        self.window_size = LFS_window_size
        self._LFS_M = LFS_M

        # Define model name and feature dimension
        # You can change this to other swin models like 'swin_tiny_patch4_window7_224'
        self.backbone_name = 'swin_base_patch4_window7_224' 
        self.backbone_feature_dim = 1024 # Swin-Base feature dim is 1024

        # Init branches with Swin Transformer backbones
        if mode == 'FAD' or mode == 'Both':
            print("Initializing FAD branch with Swin Transformer")
            self.FAD_head = FAD_Head(img_size)
            self.FAD_backbone = self._init_swin_backbone(in_chans=12) # FAD produces 12 channels

        if mode == 'LFS' or mode == 'Both':
            print("Initializing LFS branch with Swin Transformer")
            self.LFS_head = LFS_Head(img_size, LFS_window_size, LFS_M)
            # LFS produces M channels (default 6)
            self.LFS_backbone = self._init_swin_backbone(in_chans=self._LFS_M)

        if mode == 'Original':
            # For a fair comparison, the original mode should also use Swin
            print("Initializing Original branch with Swin Transformer")
            self.backbone = self._init_swin_backbone(in_chans=3) # Original uses 3 RGB channels

        # Classifier
        # The input features to the fc layer must match the Swin backbone's output
        fc_in_features = self.backbone_feature_dim
        if self.mode == 'Both':
             fc_in_features = self.backbone_feature_dim * 2

        self.fc = nn.Linear(fc_in_features, num_classes)
        self.dp = nn.Dropout(p=0.2)
    
    def _init_swin_backbone(self, in_chans: int):
            """
            Initializes a Swin Transformer model with a modified input layer.
            """
            # 1. Create a pre-trained Swin Transformer
            backbone = timm.create_model(self.backbone_name, pretrained=True)
            
            # 2. Get the original patch embedding layer's weights
            original_patch_embed = backbone.patch_embed.proj
            original_weights = original_patch_embed.weight.clone()
    
            # 3. Create a new patch embedding layer with the desired number of input channels
            new_patch_embed = nn.Conv2d(in_chans, original_patch_embed.out_channels, 
                                        kernel_size=original_patch_embed.kernel_size,
                                        stride=original_patch_embed.stride,
                                        padding=original_patch_embed.padding)
            
            # 4. Adapt the weights from 3 channels to the new number of channels
            print(f"Adapting Swin patch_embed from 3 to {in_chans} channels.")
            with torch.no_grad():
                repetition_factor = in_chans // 3
                new_weights = original_weights.repeat(1, repetition_factor, 1, 1) / repetition_factor
                new_patch_embed.weight.copy_(new_weights)
                if new_patch_embed.bias is not None:
                    new_patch_embed.bias.copy_(original_patch_embed.bias)
    
            # 5. Replace the original patch embedding layer with the new one
            backbone.patch_embed.proj = new_patch_embed
            
            # 6. Remove the final classification head, as we have our own
            #    <<<<<<<<< THIS IS THE CORRECTED LINE >>>>>>>>>
            backbone.head = nn.Identity()
    
            return backbone

    def forward(self, x):
        if self.mode == 'FAD':
            fea_FAD = self.FAD_head(x)
            # Swin backbone directly outputs the feature vector
            y = self.FAD_backbone(fea_FAD)

        elif self.mode == 'LFS':
            fea_LFS = self.LFS_head(x)
            y = self.LFS_backbone(fea_LFS)

        elif self.mode == 'Original':
            y = self.backbone(x)

        elif self.mode == 'Both':
            fea_FAD = self.FAD_head(x)
            vec_FAD = self.FAD_backbone(fea_FAD)
            
            fea_LFS = self.LFS_head(x)
            vec_LFS = self.LFS_backbone(fea_LFS)
            
            y = torch.cat((vec_FAD, vec_LFS), dim=1)

        # Classifier head
        features = self.dp(y)
        output = self.fc(features)
        
        # Return both the final feature vector `y` and the logits `output`
        return y, output


# =================================================================================
# Utility functions remain unchanged
# =================================================================================

def DCT_mat(size):
    m = [[ (np.sqrt(1./size) if i == 0 else np.sqrt(2./size)) * np.cos((j + 0.5) * np.pi * i / size) for j in range(size)] for i in range(size)]
    return m

def generate_filter(start, end, size):
    return [[0. if i + j > end or i + j < start else 1. for j in range(size)] for i in range(size)]

def norm_sigma(x):
    return 2. * torch.sigmoid(x) - 1.