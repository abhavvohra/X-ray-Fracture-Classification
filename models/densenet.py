import torch
import torch.nn as nn
import torchvision.models as models

class DenseNetFractureClassifier(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(DenseNetFractureClassifier, self).__init__()
        
        # Load pretrained DenseNet121
        self.densenet = models.densenet121(pretrained=pretrained)
        
        # Modify the first layer to accept single-channel input (X-ray images)
        self.densenet.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Replace the final classifier
        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.densenet(x)

def get_densenet_model(num_classes=1, pretrained=True):
    """
    Factory function to create a DenseNet model for fracture classification
    
    Args:
        num_classes (int): Number of output classes (default: 1 for binary classification)
        pretrained (bool): Whether to use pretrained weights (default: True)
    
    Returns:
        DenseNetFractureClassifier: Configured DenseNet model
    """
    return DenseNetFractureClassifier(num_classes=num_classes, pretrained=pretrained) 