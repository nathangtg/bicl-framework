# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, Any

# --- New TinyNet Model for Ultra-Fast Validation ---
class TinyNet(nn.Module):
    """
    A very small CNN designed for extremely fast sanity checks on image datasets
    like CIFAR-10. It is not intended for high performance but for rapid
    validation of framework logic.
    
    Architecture: [Conv -> ReLU -> Pool] x 2 -> [FC -> ReLU] -> [FC]
    """
    def __init__(self, num_classes=10):
        super(TinyNet, self).__init__()
        # Input: 3x32x32
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1) # -> 16x32x32
        self.pool1 = nn.MaxPool2d(2, 2)                         # -> 16x16x16
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)# -> 32x16x16
        self.pool2 = nn.MaxPool2d(2, 2)                         # -> 32x8x8
        
        # The flattened size is 32 * 8 * 8 = 2048
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8) # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- Enhanced and Robust `get_model` Function ---
def get_model(config: Dict[str, Any]) -> nn.Module:
    """
    Factory function to create and return a neural network model based on the
    experiment configuration. Handles ResNet-18, MLPs, and the new efficient
    TinyNet for quick validation runs.

    Args:
        config (Dict[str, Any]): The experiment configuration dictionary.

    Returns:
        A PyTorch model (nn.Module).
    """
    model_config = config['model']
    model_name = model_config.get('name', 'resnet18').lower() # Default to resnet18 if not specified

    print(f"  > Initializing model: {model_name}")

    if model_name == 'resnet18':
        # --- ResNet-18 for full-scale experiments ---
        pretrained = model_config.get('pretrained', True)
        print(f"    - Using pretrained weights: {pretrained}")
        
        # Use updated `weights` argument for modern torchvision
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        
        # Replace the final fully connected layer to match the number of classes
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, model_config['num_classes'])
        return model
        
    elif model_name == 'mlp':
        # --- MLP for simpler, non-image datasets ---
        try:
            layers = []
            input_size = model_config['input_size']
            hidden_sizes = model_config.get('hidden_sizes', [256, 128]) # Provide sensible defaults
            
            for hidden_size in hidden_sizes:
                layers.extend([
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_size),
                    nn.Dropout(model_config.get('dropout_rate', 0.3))
                ])
                input_size = hidden_size
                
            layers.append(nn.Linear(input_size, model_config['num_classes']))
            return nn.Sequential(*layers)
        except KeyError as e:
            raise KeyError(f"Missing required config for MLP: {e}. Please specify 'input_size'.")
            
    elif model_name == 'tinynet':
        # --- TinyNet for ultra-efficient validation runs ---
        return TinyNet(num_classes=model_config['num_classes'])
        
    else:
        raise ValueError(f"Model '{model_name}' is not recognized. "
                         "Available models: 'resnet18', 'mlp', 'tinynet'.")