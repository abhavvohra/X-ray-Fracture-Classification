import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from utils.transfer_learning import load_pretrained_vit_weights, freeze_layers, get_trainable_params

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                     attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
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
        
        # Patch Embedding
        self.patch_embed = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                     p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, embed_dim)
        )
        
        num_patches = (img_size // patch_size) ** 2
        
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
        # Patch Embedding
        x = self.patch_embed(x)
        B = x.shape[0]
        
        # Add CLS token
        cls_token = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)
        x = torch.cat([cls_token, x], dim=1)
        
        # Add Position Embedding
        x = x + self.pos_embed
        
        # Transformer Blocks
        x = self.blocks(x)
        x = self.norm(x)
        
        # Classification
        x = x[:, 0]  # Use CLS token
        x = self.classifier(x)
        
        return x

def get_vit_model(img_size=224, patch_size=16, in_channels=1, num_classes=1, 
                 pretrained=False, pretrained_type='imagenet',
                 freeze_transformer=True, freeze_classifier=False):
    """
    Factory function to create a Vision Transformer model for fracture classification
    
    Args:
        img_size (int): Input image size
        patch_size (int): Size of image patches
        in_channels (int): Number of input channels
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
        pretrained_type (str): Type of pretrained weights ('imagenet' or 'medical')
        freeze_transformer (bool): Whether to freeze transformer layers
        freeze_classifier (bool): Whether to freeze classifier layers
    
    Returns:
        VisionTransformer: Configured ViT model
    """
    model = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_classes=num_classes
    )
    
    if pretrained:
        model = load_pretrained_vit_weights(model, pretrained_type)
        model = freeze_layers(model, 
                            freeze_cnn=False,  # No CNN in ViT
                            freeze_transformer=freeze_transformer,
                            freeze_classifier=freeze_classifier)
        
        # Print model parameter statistics
        param_stats = get_trainable_params(model)
        print(f"Model parameter statistics:")
        print(f"Total parameters: {param_stats['total_params']:,}")
        print(f"Trainable parameters: {param_stats['trainable_params']:,}")
        print(f"Frozen parameters: {param_stats['frozen_params']:,}")
    
    return model 