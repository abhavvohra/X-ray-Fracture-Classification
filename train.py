import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from models.resnet import get_resnet_model
from utils.data_utils import create_data_loaders
from tqdm import tqdm
import os
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

def calculate_metrics(y_true, y_pred, y_prob):
    """
    Calculate various classification metrics
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities
    
    Returns:
        dict: Dictionary containing various metrics
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'auc_roc': roc_auc_score(y_true, y_prob)
    }

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    """
    Training function for the model
    
    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Number of epochs to train
        device: Device to train on ('cuda' or 'cpu')
    """
    best_val_loss = float('inf')
    best_metrics = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        train_probs = []
        
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            probs = outputs.detach().cpu().numpy()
            preds = (probs > 0.5).astype(int)
            train_preds.extend(preds)
            train_labels.extend(labels.cpu().numpy())
            train_probs.extend(probs)
        
        train_loss = train_loss / len(train_loader)
        train_metrics = calculate_metrics(
            np.array(train_labels),
            np.array(train_preds),
            np.array(train_probs)
        )
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        val_probs = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]'):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                probs = outputs.cpu().numpy()
                preds = (probs > 0.5).astype(int)
                val_preds.extend(preds)
                val_labels.extend(labels.cpu().numpy())
                val_probs.extend(probs)
        
        val_loss = val_loss / len(val_loader)
        val_metrics = calculate_metrics(
            np.array(val_labels),
            np.array(val_preds),
            np.array(val_probs)
        )
        
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}')
        print('Train Metrics:', {k: f'{v:.4f}' for k, v in train_metrics.items()})
        print(f'Val Loss: {val_loss:.4f}')
        print('Val Metrics:', {k: f'{v:.4f}' for k, v in val_metrics.items()})
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = val_metrics
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics
            }, 'best_model.pth')
            print('Saved best model checkpoint')

def main():
    parser = argparse.ArgumentParser(description='Train bone fracture classification model')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to MURA dataset directory')
    parser.add_argument('--train_csv', type=str, required=True, help='Path to training CSV file')
    parser.add_argument('--val_csv', type=str, required=True, help='Path to validation CSV file')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        data_dir=args.data_dir,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        batch_size=args.batch_size
    )
    
    # Initialize model
    model = get_resnet_model(num_classes=1, pretrained=True)
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Train model
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        device=device
    )

if __name__ == '__main__':
    main() 