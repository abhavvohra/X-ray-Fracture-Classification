import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import argparse
from datetime import datetime
import json
import logging
from pathlib import Path

# Import our custom modules
from models.resnet import get_resnet_model
from models.unet import get_unet_model
from models.attention_unet import get_attention_unet_model
from models.densenet import get_densenet_model
from models.vit import get_vit_model
from models.hybrid_vit import get_hybrid_vit_model
from utils.data_utils import create_data_loaders
from utils.metrics import ModelEvaluator
from utils.synthetic_generation import SyntheticImageGenerator

def setup_logging(log_dir):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Bone Fracture Classification Training')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='data/MURA',
                      help='Path to MURA dataset')
    parser.add_argument('--train_csv', type=str, default='data/MURA/train.csv',
                      help='Path to training CSV file')
    parser.add_argument('--val_csv', type=str, default='data/MURA/valid.csv',
                      help='Path to validation CSV file')
    parser.add_argument('--img_size', type=int, default=224,
                      help='Input image size')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of workers for data loading')
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=50,
                      help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                      help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                      help='Weight decay')
    
    # Model parameters
    parser.add_argument('--in_channels', type=int, default=1,
                      help='Number of input channels')
    parser.add_argument('--num_classes', type=int, default=1,
                      help='Number of output classes')
    
    # Synthetic data parameters
    parser.add_argument('--num_synthetic_images', type=int, default=1000,
                      help='Number of synthetic images to generate')
    
    # Paths
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                      help='Directory to save model checkpoints')
    parser.add_argument('--results_dir', type=str, default='results',
                      help='Directory to save results')
    parser.add_argument('--log_dir', type=str, default='logs',
                      help='Directory to save logs')
    
    # Training options
    parser.add_argument('--use_synthetic', action='store_true',
                      help='Use synthetic data for training')
    parser.add_argument('--pretrained', action='store_true',
                      help='Use pretrained weights')
    parser.add_argument('--models', nargs='+', default=['resnet', 'unet', 'attention_unet', 'densenet', 'vit', 'hybrid_vit'],
                      help='Models to train')
    
    return parser.parse_args()

def train_model(model, train_loader, val_loader, args, logger):
    """Train a single model"""
    model = model.to(args.device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(args.num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_steps = 0
        
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.num_epochs}'):
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1).float())
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_steps += 1
        
        avg_train_loss = train_loss / train_steps
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_steps = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(args.device), labels.to(args.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1).float())
                val_loss += loss.item()
                val_steps += 1
        
        avg_val_loss = val_loss / val_steps
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 
                      os.path.join(args.checkpoint_dir, f'{model.__class__.__name__}_best.pth'))
        
        logger.info(f'Epoch {epoch+1}/{args.num_epochs}:')
        logger.info(f'Training Loss: {avg_train_loss:.4f}')
        logger.info(f'Validation Loss: {avg_val_loss:.4f}')
        logger.info('-' * 50)
    
    return model, train_losses, val_losses

def save_results(results, args):
    """Save training results"""
    # Save metrics
    metrics_file = os.path.join(args.results_dir, 'model_comparison.json')
    with open(metrics_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Save plots
    plt.figure(figsize=(15, 5))
    
    # Plot training losses
    plt.subplot(1, 2, 1)
    for name, history in results['training_histories'].items():
        plt.plot(history['train_losses'], label=name)
    plt.title('Training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot validation losses
    plt.subplot(1, 2, 2)
    for name, history in results['training_histories'].items():
        plt.plot(history['val_losses'], label=name)
    plt.title('Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, 'training_history.png'))
    plt.close()

def main():
    # Parse arguments
    args = parse_args()
    
    # Setup device
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(args.log_dir)
    logger.info(f"Starting training with arguments: {args}")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        data_dir=args.data_dir,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    
    # Generate synthetic data if requested
    if args.use_synthetic:
        logger.info("Generating synthetic data...")
        synthetic_generator = SyntheticImageGenerator(device=args.device)
        synthetic_images, synthetic_labels = synthetic_generator.generate_synthetic_dataset(
            real_images=next(iter(train_loader))[0],
            num_stylegan_images=args.num_synthetic_images
        )
        logger.info(f"Generated {len(synthetic_images)} synthetic images")
    
    # Initialize models
    model_factories = {
        'resnet': lambda: get_resnet_model(pretrained=args.pretrained),
        'unet': lambda: get_unet_model(),
        'attention_unet': lambda: get_attention_unet_model(),
        'densenet': lambda: get_densenet_model(pretrained=args.pretrained),
        'vit': lambda: get_vit_model(pretrained=args.pretrained),
        'hybrid_vit': lambda: get_hybrid_vit_model(pretrained=args.pretrained)
    }
    
    models = {name: factory() for name, factory in model_factories.items() 
             if name in args.models}
    
    # Train models
    trained_models = {}
    training_histories = {}
    
    for name, model in models.items():
        logger.info(f"\nTraining {name}...")
        trained_model, train_losses, val_losses = train_model(
            model, train_loader, val_loader, args, logger
        )
        trained_models[name] = trained_model
        training_histories[name] = {
            'train_losses': train_losses,
            'val_losses': val_losses
        }
    
    # Evaluate models
    logger.info("\nEvaluating models...")
    evaluator = ModelEvaluator()
    results = evaluator.compare_models(trained_models, val_loader, args.device)
    
    # Save results
    results_dict = {
        'metrics': results,
        'training_histories': training_histories
    }
    save_results(results_dict, args)
    
    logger.info(f"\nResults saved to {args.results_dir}")
    logger.info("Training completed!")

if __name__ == '__main__':
    main() 