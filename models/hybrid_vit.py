import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from .vit import MultiHeadAttention, MLP, TransformerBlock
from utils.transfer_learning import load_pretrained_hybrid_vit_weights, freeze_layers, get_trainable_params

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class HybridVisionTransformer(nn.Module):
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 in_channels=1, 
                 num_classes=1,
                 embed_dim=768, 
                 depth=12, 
                 num_heads=12, 
                 mlp_ratio=4., 
                 qkv_bias=True,
                 drop_rate=0.1, 
                 attn_drop_rate=0.1):
        super().__init__()
        
        # CNN Feature Extractor
        self.cnn = nn.Sequential(
            CNNBlock(in_channels, 64),
            nn.MaxPool2d(2),
            CNNBlock(64, 128),
            nn.MaxPool2d(2),
            CNNBlock(128, 256),
            nn.MaxPool2d(2),
            CNNBlock(256, 512),
            nn.MaxPool2d(2)
        )
        
        # Calculate CNN output size
        cnn_output_size = img_size // 16  # After 4 maxpool layers
        cnn_output_channels = 512
        
        # Patch Embedding for Transformer
        self.patch_embed = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                     p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * cnn_output_channels, embed_dim)
        )
        
        num_patches = (cnn_output_size // patch_size) ** 2
        
        # Position Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer Blocks
        self.blocks = nn.Sequential(*[
            TransformerBlock(
                dim=embed_dim, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, 
                drop=drop_rate, 
                attn_drop=attn_drop_rate
            )
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Feature Fusion
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim + cnn_output_channels, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
            nn.Sigmoid()
        )
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        # Extract CNN features
        cnn_features = self.cnn(x)
        B = x.shape[0]
        
        # Global average pooling of CNN features
        cnn_pooled = F.adaptive_avg_pool2d(cnn_features, (1, 1)).view(B, -1)
        
        # Patch Embedding
        x = self.patch_embed(cnn_features)
        
        # Add CLS token
        cls_token = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)
        x = torch.cat([cls_token, x], dim=1)
        
        # Add Position Embedding
        x = x + self.pos_embed
        
        # Transformer Blocks
        x = self.blocks(x)
        x = self.norm(x)
        
        # Use CLS token
        x = x[:, 0]
        
        # Feature Fusion
        x = torch.cat([x, cnn_pooled], dim=1)
        x = self.fusion(x)
        
        # Classification
        x = self.classifier(x)
        
        return x

def get_hybrid_vit_model(img_size=224, patch_size=16, in_channels=1, num_classes=1, 
                        pretrained=False, pretrained_type='imagenet',
                        freeze_cnn=True, freeze_transformer=True, freeze_classifier=False):
    """
    Factory function to create a Hybrid Vision Transformer model for fracture classification
    
    Args:
        img_size (int): Input image size
        patch_size (int): Size of image patches
        in_channels (int): Number of input channels
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
        pretrained_type (str): Type of pretrained weights ('imagenet' or 'medical')
        freeze_cnn (bool): Whether to freeze CNN layers
        freeze_transformer (bool): Whether to freeze transformer layers
        freeze_classifier (bool): Whether to freeze classifier layers
    
    Returns:
        HybridVisionTransformer: Configured Hybrid ViT model
    """
    model = HybridVisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_classes=num_classes
    )
    
    if pretrained:
        model = load_pretrained_hybrid_vit_weights(model, pretrained_type)
        model = freeze_layers(model, 
                            freeze_cnn=freeze_cnn,
                            freeze_transformer=freeze_transformer,
                            freeze_classifier=freeze_classifier)
        
        # Print model parameter statistics
        param_stats = get_trainable_params(model)
        print(f"Model parameter statistics:")
        print(f"Total parameters: {param_stats['total_params']:,}")
        print(f"Trainable parameters: {param_stats['trainable_params']:,}")
        print(f"Frozen parameters: {param_stats['frozen_params']:,}")
    
    return model 