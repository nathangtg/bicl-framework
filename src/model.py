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
    return nn.Sequential(*layers)