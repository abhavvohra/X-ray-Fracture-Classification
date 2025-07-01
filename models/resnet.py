import torch
import torch.nn as nn
import torchvision.models as models

class ResNetFractureClassifier(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(ResNetFractureClassifier, self).__init__()
        
        # Load pretrained ResNet50
        self.resnet = models.resnet50(pretrained=pretrained)
        
        # Modify the first layer to accept single-channel input (X-ray images)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Replace the final fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
            nn.Sigmoid()  # Using sigmoid for binary classification
        )
    
    def forward(self, x):
        return self.resnet(x)

def get_resnet_model(num_classes=1, pretrained=True):
    """
    Factory function to create a ResNet model for fracture classification
    
    Args:
        num_classes (int): Number of output classes (default: 1 for binary classification)
        pretrained (bool): Whether to use pretrained weights (default: True)
    
    Returns:
        ResNetFractureClassifier: Configured ResNet model
    """
    return ResNetFractureClassifier(num_classes=num_classes, pretrained=pretrained) 