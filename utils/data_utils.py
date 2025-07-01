import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
import random
import cv2
from scipy.ndimage import gaussian_filter

class MedicalImageAugmentation:
    """Custom augmentation class for medical images"""
    
    @staticmethod
    def add_gaussian_noise(image: np.ndarray, mean: float = 0, std: float = 0.1) -> np.ndarray:
        """Add Gaussian noise to the image"""
        noise = np.random.normal(mean, std, image.shape)
        noisy_image = image + noise
        return np.clip(noisy_image, 0, 1)
    
    @staticmethod
    def adjust_gamma(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        """Adjust gamma of the image"""
        return np.power(image, gamma)
    
    @staticmethod
    def elastic_transform(image: np.ndarray, alpha: float = 1, sigma: float = 0.5) -> np.ndarray:
        """Apply elastic transformation to the image"""
        shape = image.shape
        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        
        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))
        
        return np.reshape(map_coordinates(image, indices, order=1), shape)
    
    @staticmethod
    def random_erasing(image: np.ndarray, probability: float = 0.5) -> np.ndarray:
        """Randomly erase a rectangular region from the image"""
        if random.random() < probability:
            h, w = image.shape
            area = h * w
            
            # Randomly select the area to erase
            target_area = random.uniform(0.02, 0.2) * area
            aspect_ratio = random.uniform(0.3, 3)
            
            h_erase = int(round(np.sqrt(target_area * aspect_ratio)))
            w_erase = int(round(np.sqrt(target_area / aspect_ratio)))
            
            if h_erase < h and w_erase < w:
                i = random.randint(0, h - h_erase)
                j = random.randint(0, w - w_erase)
                image[i:i + h_erase, j:j + w_erase] = random.uniform(0, 1)
        
        return image

class MURADataset(Dataset):
    def __init__(self, 
                 data_dir: str,
                 csv_file: str,
                 transform: Optional[transforms.Compose] = None,
                 is_train: bool = True):
        """
        MURA Dataset loader
        
        Args:
            data_dir (str): Root directory of the MURA dataset
            csv_file (str): Path to the CSV file containing image paths and labels
            transform (Optional[transforms.Compose]): Optional transforms to apply
            is_train (bool): Whether this is training data (affects augmentation)
        """
        self.data_dir = data_dir
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.is_train = is_train
        self.medical_aug = MedicalImageAugmentation()
        
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = os.path.join(self.data_dir, self.df.iloc[idx]['Path'])
        label = self.df.iloc[idx]['Label']
        
        # Load and convert to grayscale
        image = Image.open(img_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
            
            # Apply medical-specific augmentations during training
            if self.is_train:
                # Convert to numpy for custom augmentations
                image_np = image.numpy()[0]  # Remove channel dimension
                
                # Apply augmentations with some probability
                if random.random() < 0.3:
                    image_np = self.medical_aug.add_gaussian_noise(image_np)
                if random.random() < 0.3:
                    gamma = random.uniform(0.8, 1.2)
                    image_np = self.medical_aug.adjust_gamma(image_np, gamma)
                if random.random() < 0.3:
                    image_np = self.medical_aug.elastic_transform(image_np)
                if random.random() < 0.3:
                    image_np = self.medical_aug.random_erasing(image_np)
                
                # Convert back to tensor
                image = torch.from_numpy(image_np).unsqueeze(0)
        
        return image, torch.tensor(label, dtype=torch.float32)

def get_transforms(is_train: bool = True) -> transforms.Compose:
    """
    Get transforms for data preprocessing and augmentation
    
    Args:
        is_train (bool): Whether these are training transforms
    
    Returns:
        transforms.Compose: Composition of transforms
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])

def create_data_loaders(
    data_dir: str,
    train_csv: str,
    val_csv: str,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders
    
    Args:
        data_dir (str): Root directory of the MURA dataset
        train_csv (str): Path to training CSV file
        val_csv (str): Path to validation CSV file
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of workers for data loading
    
    Returns:
        Tuple[DataLoader, DataLoader]: Training and validation data loaders
    """
    train_dataset = MURADataset(
        data_dir=data_dir,
        csv_file=train_csv,
        transform=get_transforms(is_train=True),
        is_train=True
    )
    
    val_dataset = MURADataset(
        data_dir=data_dir,
        csv_file=val_csv,
        transform=get_transforms(is_train=False),
        is_train=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader 