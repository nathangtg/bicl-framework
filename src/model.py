import torch.nn as nn

def get_model(config):
    model_config = config['model']
    layers = []
    prev_size = model_config['input_size']
    for hidden_size in model_config['hidden_sizes']:
        layers.extend([
            nn.Linear(prev_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(model_config['dropout_rate'])
        ])
        prev_size = hidden_size
    layers.append(nn.Linear(prev_size, model_config['num_classes']))
    return nn.Sequential(*layers)# src/model.py
import torch
import torch.nn as nn
from torchvision import models
from typing import Dict, Any

def get_model(config: Dict[str, Any]) -> nn.Module:
    """
    Factory function to create and return a neural network model based on the
    experiment configuration.

    Args:
        config (Dict[str, Any]): The experiment configuration dictionary.

    Returns:
        A PyTorch model (nn.Module).
    """
    model_config = config['model']
    model_name = model_config['name']

    if model_name.lower() == 'resnet18':
        # Load a ResNet-18 model, optionally with pretrained weights
        model = models.resnet18(pretrained=model_config.get('pretrained', True))
        
        # Replace the final fully connected layer to match the number of classes
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, model_config['num_classes'])
        return model
        
    elif model_name.lower() == 'mlp':
        # Create a standard MLP for synthetic data experiments
        layers = []
        input_size = model_config['input_size']
        hidden_sizes = model_config['hidden_sizes']
        
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
        
    else:
        raise ValueError(f"Model '{model_name}' is not recognized.")

