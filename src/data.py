import numpy as np
import torch
from torch.utils.data import TensorDataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def generate_diverse_tasks(config):
    data_config = config['data']
    num_tasks = data_config['num_tasks']
    n_samples = data_config['n_samples_per_task']
    input_size = config['model']['input_size']
    num_classes = config['model']['num_classes']
    task_difficulty = data_config['task_difficulty']
    base_seed = config['experiment']['seed']
    
    tasks = []
    for i in range(num_tasks):
        if task_difficulty == 'progressive':
            n_informative = max(5, min(20, 5 + i * 3))
            class_sep = max(0.5, 1.5 - i * 0.2)
        else:
            n_informative = 15
            class_sep = 1.0

        X, y = make_classification(
            n_samples=n_samples, n_features=input_size, n_classes=num_classes,
            n_informative=n_informative, n_redundant=5, n_clusters_per_class=1,
            class_sep=class_sep, random_state=base_seed + i
        )

        if i > 0:
            angle = i * 0.3; cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation_subset = np.random.choice(X.shape[1], X.shape[1]//2, replace=False)
            X_rot = X[:, rotation_subset].copy()
            X[:, rotation_subset] = X_rot * cos_a - X_rot * sin_a
            X = X * (1.0 + i * 0.2)
            X += np.random.normal(0, 0.1 + i * 0.05, X.shape)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=base_seed + i, stratify=y)
        
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
        tasks.append((train_dataset, test_dataset))
    return tasks