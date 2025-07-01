# Bone Fracture Classification System

This project implements a deep learning-based bone fracture classification system using various architectures including ResNet, U-Net, Attention U-Net, DenseNet, Vision Transformers, and Hybrid ViT.

## Project Structure

```
.
├── models/             # Model architectures
│   └── resnet.py      # ResNet implementation
├── data/              # Dataset directory
├── utils/             # Utility functions
├── train.py           # Training script
├── requirements.txt   # Project dependencies
└── README.md          # Project documentation
```

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Current Implementation

Currently implemented:
- ResNet50-based fracture classifier with transfer learning
- Basic training pipeline with validation
- Support for binary classification (fracture/no fracture)

## Usage

To train the model:
```bash
python train.py
```

## Next Steps

1. Implement remaining architectures:
   - U-Net
   - Attention U-Net
   - DenseNet
   - Vision Transformers
   - Hybrid ViT

2. Add data loading and preprocessing for MURA dataset
3. Implement data augmentation
4. Add evaluation metrics
5. Implement synthetic image generation with CycleGAN and StyleGAN 