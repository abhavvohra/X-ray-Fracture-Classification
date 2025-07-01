import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Tuple, List, Dict
import os
from PIL import Image
import torchvision.transforms as transforms

class CycleGANGenerator(nn.Module):
    def __init__(self, input_channels=1):
        super(CycleGANGenerator, self).__init__()
        
        # Initial convolution
        self.initial = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Downsampling
        self.down = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks
        self.residual = nn.Sequential(*[
            ResidualBlock(256) for _ in range(9)
        ])
        
        # Upsampling
        self.up = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Output convolution
        self.output = nn.Conv2d(64, input_channels, kernel_size=7, stride=1, padding=3)
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        x = self.initial(x)
        x = self.down(x)
        x = self.residual(x)
        x = self.up(x)
        x = self.output(x)
        return self.tanh(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(channels)
        )
    
    def forward(self, x):
        return x + self.block(x)

class StyleGANGenerator(nn.Module):
    def __init__(self, latent_dim=512, img_size=256):
        super(StyleGANGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        
        # Mapping network
        self.mapping = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, latent_dim),
            nn.LeakyReLU(0.2)
        )
        
        # Synthesis network
        self.synthesis = nn.ModuleList([
            SynthesisBlock(latent_dim, 512, 4),
            SynthesisBlock(latent_dim, 512, 8),
            SynthesisBlock(latent_dim, 512, 16),
            SynthesisBlock(latent_dim, 256, 32),
            SynthesisBlock(latent_dim, 128, 64),
            SynthesisBlock(latent_dim, 64, 128),
            SynthesisBlock(latent_dim, 32, 256)
        ])
        
        # Output layer
        self.output = nn.Conv2d(32, 1, kernel_size=1)
        self.tanh = nn.Tanh()
    
    def forward(self, z):
        w = self.mapping(z)
        x = None
        
        for block in self.synthesis:
            x = block(x, w)
        
        x = self.output(x)
        return self.tanh(x)

class SynthesisBlock(nn.Module):
    def __init__(self, latent_dim, in_channels, size):
        super(SynthesisBlock, self).__init__()
        self.size = size
        
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        
        self.style1 = StyleModulation(latent_dim, in_channels)
        self.style2 = StyleModulation(latent_dim, in_channels)
        self.style3 = StyleModulation(latent_dim, in_channels)
        
        self.upsample = nn.Upsample(size=(size, size), mode='bilinear', align_corners=True)
    
    def forward(self, x, w):
        if x is None:
            x = torch.randn(1, self.conv1.in_channels, self.size, self.size).to(w.device)
        
        x = self.upsample(x)
        x = self.style1(self.conv1(x), w)
        x = self.style2(self.conv2(x), w)
        x = self.style3(self.conv3(x), w)
        return x

class StyleModulation(nn.Module):
    def __init__(self, latent_dim, channels):
        super(StyleModulation, self).__init__()
        self.affine = nn.Linear(latent_dim, channels * 2)
    
    def forward(self, x, w):
        style = self.affine(w)
        gamma, beta = style.chunk(2, dim=1)
        gamma = gamma.view(-1, gamma.size(1), 1, 1)
        beta = beta.view(-1, beta.size(1), 1, 1)
        return gamma * x + beta

class SyntheticImageGenerator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.cyclegan = CycleGANGenerator().to(device)
        self.stylegan = StyleGANGenerator().to(device)
        
        # Load pretrained weights if available
        self._load_pretrained_weights()
    
    def _load_pretrained_weights(self):
        """Load pretrained weights for both generators"""
        cyclegan_path = 'pretrained_weights/cyclegan.pth'
        stylegan_path = 'pretrained_weights/stylegan.pth'
        
        if os.path.exists(cyclegan_path):
            self.cyclegan.load_state_dict(torch.load(cyclegan_path))
        
        if os.path.exists(stylegan_path):
            self.stylegan.load_state_dict(torch.load(stylegan_path))
    
    def generate_with_cyclegan(self, image: torch.Tensor) -> torch.Tensor:
        """
        Generate synthetic image using CycleGAN
        
        Args:
            image: Input image tensor (1, H, W)
        
        Returns:
            Generated image tensor
        """
        self.cyclegan.eval()
        with torch.no_grad():
            return self.cyclegan(image.unsqueeze(0).to(self.device))
    
    def generate_with_stylegan(self, num_images: int = 1) -> torch.Tensor:
        """
        Generate synthetic images using StyleGAN
        
        Args:
            num_images: Number of images to generate
        
        Returns:
            Generated image tensor
        """
        self.stylegan.eval()
        with torch.no_grad():
            z = torch.randn(num_images, 512).to(self.device)
            return self.stylegan(z)
    
    def generate_synthetic_dataset(self, 
                                 real_images: torch.Tensor,
                                 num_stylegan_images: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate synthetic dataset using both CycleGAN and StyleGAN
        
        Args:
            real_images: Real image tensor (N, 1, H, W)
            num_stylegan_images: Number of StyleGAN images to generate
        
        Returns:
            Tuple of (synthetic images, labels)
        """
        synthetic_images = []
        
        # Generate with CycleGAN
        for img in real_images:
            synthetic = self.generate_with_cyclegan(img)
            synthetic_images.append(synthetic)
        
        # Generate with StyleGAN
        stylegan_images = self.generate_with_stylegan(num_stylegan_images)
        synthetic_images.extend(stylegan_images)
        
        # Combine all synthetic images
        synthetic_images = torch.cat(synthetic_images, dim=0)
        
        # Create labels (assuming binary classification)
        labels = torch.ones(len(synthetic_images))
        
        return synthetic_images, labels
    
    def save_synthetic_images(self, 
                            images: torch.Tensor,
                            save_dir: str,
                            prefix: str = 'synthetic'):
        """
        Save generated images to disk
        
        Args:
            images: Image tensor (N, 1, H, W)
            save_dir: Directory to save images
            prefix: Prefix for image filenames
        """
        os.makedirs(save_dir, exist_ok=True)
        
        transform = transforms.ToPILImage()
        for i, img in enumerate(images):
            img = img.squeeze().cpu()
            img = transform(img)
            img.save(os.path.join(save_dir, f'{prefix}_{i:04d}.png')) 