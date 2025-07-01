import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
import timm
from typing import Optional, Dict, Any

def load_pretrained_vit_weights(model: nn.Module, pretrained_type: str = 'imagenet') -> nn.Module:
    """
    Load pretrained weights for Vision Transformer
    
    Args:
        model: ViT model to load weights into
        pretrained_type: Type of pretrained weights ('imagenet' or 'medical')
    
    Returns:
        Model with loaded weights
    """
    if pretrained_type == 'imagenet':
        # Load ImageNet pretrained weights
        pretrained = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        
        # Transfer weights
        model.patch_embed[1].weight.data = pretrained.patch_embed[1].weight.data
        model.pos_embed.data = pretrained.pos_embed.data
        model.cls_token.data = pretrained.cls_token.data
        
        # Transfer transformer blocks
        for i, (src_block, dst_block) in enumerate(zip(pretrained.blocks, model.blocks)):
            dst_block.norm1.weight.data = src_block.norm1.weight.data
            dst_block.norm1.bias.data = src_block.norm1.bias.data
            dst_block.attn.qkv.weight.data = src_block.attn.qkv.weight.data
            dst_block.attn.qkv.bias.data = src_block.attn.qkv.bias.data
            dst_block.attn.proj.weight.data = src_block.attn.proj.weight.data
            dst_block.attn.proj.bias.data = src_block.attn.proj.bias.data
            dst_block.norm2.weight.data = src_block.norm2.weight.data
            dst_block.norm2.bias.data = src_block.norm2.bias.data
            dst_block.mlp.fc1.weight.data = src_block.mlp.fc1.weight.data
            dst_block.mlp.fc1.bias.data = src_block.mlp.fc1.bias.data
            dst_block.mlp.fc2.weight.data = src_block.mlp.fc2.weight.data
            dst_block.mlp.fc2.bias.data = src_block.mlp.fc2.bias.data
        
        model.norm.weight.data = pretrained.norm.weight.data
        model.norm.bias.data = pretrained.norm.bias.data
        
    elif pretrained_type == 'medical':
        # Load medical pretrained weights if available
        # This would be your custom pretrained weights on medical images
        try:
            state_dict = torch.load('pretrained_weights/medical_vit.pth')
            model.load_state_dict(state_dict)
        except FileNotFoundError:
            print("Medical pretrained weights not found. Using ImageNet weights instead.")
            return load_pretrained_vit_weights(model, 'imagenet')
    
    return model

def load_pretrained_hybrid_vit_weights(model: nn.Module, pretrained_type: str = 'imagenet') -> nn.Module:
    """
    Load pretrained weights for Hybrid Vision Transformer
    
    Args:
        model: Hybrid ViT model to load weights into
        pretrained_type: Type of pretrained weights ('imagenet' or 'medical')
    
    Returns:
        Model with loaded weights
    """
    if pretrained_type == 'imagenet':
        # Load pretrained CNN weights
        cnn_pretrained = timm.create_model('resnet50', pretrained=True)
        
        # Transfer CNN weights
        model.cnn[0].conv[0].weight.data = cnn_pretrained.conv1.weight.data[:, :1]  # Only first channel
        model.cnn[2].conv[0].weight.data = cnn_pretrained.layer1[0].conv1.weight.data
        model.cnn[4].conv[0].weight.data = cnn_pretrained.layer2[0].conv1.weight.data
        model.cnn[6].conv[0].weight.data = cnn_pretrained.layer3[0].conv1.weight.data
        
        # Load ViT weights
        vit_pretrained = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        
        # Transfer ViT weights
        model.patch_embed[1].weight.data = vit_pretrained.patch_embed[1].weight.data
        model.pos_embed.data = vit_pretrained.pos_embed.data
        model.cls_token.data = vit_pretrained.cls_token.data
        
        # Transfer transformer blocks
        for i, (src_block, dst_block) in enumerate(zip(vit_pretrained.blocks, model.blocks)):
            dst_block.norm1.weight.data = src_block.norm1.weight.data
            dst_block.norm1.bias.data = src_block.norm1.bias.data
            dst_block.attn.qkv.weight.data = src_block.attn.qkv.weight.data
            dst_block.attn.qkv.bias.data = src_block.attn.qkv.bias.data
            dst_block.attn.proj.weight.data = src_block.attn.proj.weight.data
            dst_block.attn.proj.bias.data = src_block.attn.proj.bias.data
            dst_block.norm2.weight.data = src_block.norm2.weight.data
            dst_block.norm2.bias.data = src_block.norm2.bias.data
            dst_block.mlp.fc1.weight.data = src_block.mlp.fc1.weight.data
            dst_block.mlp.fc1.bias.data = src_block.mlp.fc1.bias.data
            dst_block.mlp.fc2.weight.data = src_block.mlp.fc2.weight.data
            dst_block.mlp.fc2.bias.data = src_block.mlp.fc2.bias.data
        
        model.norm.weight.data = vit_pretrained.norm.weight.data
        model.norm.bias.data = vit_pretrained.norm.bias.data
        
    elif pretrained_type == 'medical':
        # Load medical pretrained weights if available
        try:
            state_dict = torch.load('pretrained_weights/medical_hybrid_vit.pth')
            model.load_state_dict(state_dict)
        except FileNotFoundError:
            print("Medical pretrained weights not found. Using ImageNet weights instead.")
            return load_pretrained_hybrid_vit_weights(model, 'imagenet')
    
    return model

def freeze_layers(model: nn.Module, 
                 freeze_cnn: bool = True, 
                 freeze_transformer: bool = True,
                 freeze_classifier: bool = False) -> nn.Module:
    """
    Freeze specific layers of the model for transfer learning
    
    Args:
        model: Model to freeze layers in
        freeze_cnn: Whether to freeze CNN layers
        freeze_transformer: Whether to freeze transformer layers
        freeze_classifier: Whether to freeze classifier layers
    
    Returns:
        Model with frozen layers
    """
    if isinstance(model, nn.Module):
        for name, param in model.named_parameters():
            if 'cnn' in name and freeze_cnn:
                param.requires_grad = False
            elif any(x in name for x in ['blocks', 'patch_embed', 'pos_embed', 'cls_token']) and freeze_transformer:
                param.requires_grad = False
            elif 'classifier' in name and freeze_classifier:
                param.requires_grad = False
    
    return model

def get_trainable_params(model: nn.Module) -> Dict[str, int]:
    """
    Get the number of trainable parameters in the model
    
    Args:
        model: Model to count parameters for
    
    Returns:
        Dictionary with total and trainable parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'frozen_params': total_params - trainable_params
    } 