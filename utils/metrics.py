import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    def __init__(self):
        self.metrics = {}
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
        """
        Calculate all evaluation metrics
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities
        
        Returns:
            Dictionary containing all metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'sensitivity': recall_score(y_true, y_pred),  # Same as recall for binary classification
            'specificity': recall_score(y_true, y_pred, pos_label=0),
            'f1': f1_score(y_true, y_pred),
            'auc_roc': roc_auc_score(y_true, y_prob)
        }
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm
        
        return metrics
    
    def evaluate_model(self, model: torch.nn.Module, 
                      dataloader: torch.utils.data.DataLoader,
                      device: torch.device) -> Dict[str, float]:
        """
        Evaluate model on a dataloader
        
        Args:
            model: PyTorch model
            dataloader: DataLoader for evaluation
            device: Device to run evaluation on
        
        Returns:
            Dictionary containing all metrics
        """
        model.eval()
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                probs = outputs.cpu().numpy()
                preds = (probs > 0.5).astype(int)
                
                all_preds.extend(preds)
                all_probs.extend(probs)
                all_labels.extend(labels.cpu().numpy())
        
        return self.calculate_metrics(
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs)
        )
    
    def plot_confusion_matrix(self, cm: np.ndarray, title: str = 'Confusion Matrix'):
        """
        Plot confusion matrix
        
        Args:
            cm: Confusion matrix
            title: Plot title
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
    
    def plot_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray, title: str = 'ROC Curve'):
        """
        Plot ROC curve
        
        Args:
            y_true: Ground truth labels
            y_prob: Predicted probabilities
            title: Plot title
        """
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend()
        plt.show()
    
    def compare_models(self, models: Dict[str, torch.nn.Module],
                      dataloader: torch.utils.data.DataLoader,
                      device: torch.device) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple models
        
        Args:
            models: Dictionary of model names and models
            dataloader: DataLoader for evaluation
            device: Device to run evaluation on
        
        Returns:
            Dictionary containing metrics for each model
        """
        results = {}
        for name, model in models.items():
            print(f"Evaluating {name}...")
            metrics = self.evaluate_model(model, dataloader, device)
            results[name] = metrics
            
            # Plot confusion matrix
            self.plot_confusion_matrix(metrics['confusion_matrix'], f'{name} Confusion Matrix')
        
        return results
    
    def print_comparison(self, results: Dict[str, Dict[str, float]]):
        """
        Print comparison of model results
        
        Args:
            results: Dictionary containing metrics for each model
        """
        metrics = ['accuracy', 'precision', 'recall', 'sensitivity', 'specificity', 'f1', 'auc_roc']
        
        print("\nModel Comparison:")
        print("-" * 80)
        print(f"{'Model':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'AUC-ROC':<10}")
        print("-" * 80)
        
        for name, metrics_dict in results.items():
            print(f"{name:<15}", end='')
            for metric in metrics:
                if metric != 'confusion_matrix':
                    print(f"{metrics_dict[metric]:.4f}".ljust(10), end='')
            print()
        
        print("-" * 80) 