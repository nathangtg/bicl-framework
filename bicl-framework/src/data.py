# src/data.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, TensorDataset, Subset
from typing import List, Tuple
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import logging

# ==============================================================================
# Synthetic Task Generation
# (This section is unchanged as it's already fast)
# ==============================================================================

def generate_diverse_tasks(config: dict) -> List[Tuple[Dataset, Dataset]]:
    """
    Generates a sequence of synthetic tasks with progressively increasing
    difficulty, designed for rapid testing and debugging of continual learning models.
    """
    data_config = config['data']
    num_tasks = data_config['num_tasks']
    n_samples = data_config['n_samples_per_task']
    input_size = config['model']['input_size']
    num_classes = config['model']['num_classes']
    task_difficulty = data_config.get('task_difficulty', 'progressive')
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
            angle = i * 0.3
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation_subset = np.random.choice(X.shape[1], X.shape[1] // 2, replace=False)
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

# ==============================================================================
# Real-World Benchmark Datasets
# (Wrappers are unchanged)
# ==============================================================================

class TaskSplitter(Dataset):
    """
    A custom PyTorch Dataset wrapper to create a task-specific subset of a larger dataset.
    """
    def __init__(self, dataset: Dataset, task_labels: List[int]):
        self.dataset = dataset
        self.task_labels = set(task_labels)
        self.indices = [i for i, (_, label) in enumerate(dataset) if label in self.task_labels]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        original_idx = self.indices[idx]
        return self.dataset[original_idx]

class PermutedMNIST(Dataset):
    """
    A custom Dataset for the Permuted MNIST benchmark.
    """
    def __init__(self, dataset: Dataset, permutation: np.ndarray):
        self.dataset = dataset
        self.permutation = permutation

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img, label = self.dataset[idx]
        img_flat = img.view(-1)
        permuted_img_flat = img_flat[self.permutation]
        return permuted_img_flat.view(img.shape), label

# ==============================================================================
# Main Benchmark Loading Function (MODIFIED FOR EFFICIENCY)
# ==============================================================================

def get_benchmark_tasks(config: dict) -> List[Tuple[Dataset, Dataset]]:
    """
    Fetches and prepares a real-world benchmark dataset for a continual learning experiment.
    This function now supports Split CIFAR-10/100, Permuted MNIST, and an efficient
    'subset_fraction' option for rapid testing.
    """
    data_config = config['data']
    benchmark_name = data_config['benchmark']
    data_path = data_config['data_path']
    num_tasks = data_config['num_tasks']
    seed = config['experiment']['seed']
    
    rng = np.random.default_rng(seed)

    # --- Split CIFAR-10 / CIFAR-100 Logic ---
    if benchmark_name.lower() in ['cifar10', 'cifar100']:
        if benchmark_name.lower() == 'cifar10':
            logging.info("Preparing Split CIFAR-10 benchmark...")
            dataset_loader = datasets.CIFAR10
            num_classes = 10
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        else: # cifar100
            logging.info("Preparing Split CIFAR-100 benchmark...")
            dataset_loader = datasets.CIFAR100
            num_classes = 100
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])

        # Load the full datasets first
        full_train_set = dataset_loader(root=data_path, train=True, download=True, transform=transform)
        test_set = dataset_loader(root=data_path, train=False, download=True, transform=transform)

        # --------------------------------------------------------------------------
        # --- EFFICIENCY BOOST: SUBSET THE TRAINING DATA ---
        # This is the only change needed. It reads 'subset_fraction' from the config.
        subset_fraction = data_config.get('subset_fraction', 1.0)
        if subset_fraction < 1.0:
            num_samples = int(len(full_train_set) * subset_fraction)
            # Use a fixed seed for reproducibility of the subset itself
            subset_indices = np.random.choice(len(full_train_set), num_samples, replace=False)
            train_set = Subset(full_train_set, subset_indices)
            logging.info(f"Using a {subset_fraction*100:.0f}% subset of the training data ({len(train_set)} samples).")
        else:
            train_set = full_train_set # Use the full dataset if fraction is 1.0
        # --------------------------------------------------------------------------
        
        all_labels = list(range(num_classes))
        rng.shuffle(all_labels)
        class_splits = np.array_split(all_labels, num_tasks)
        
        tasks = []
        for task_labels in class_splits:
            train_task_dataset = TaskSplitter(train_set, task_labels)
            test_task_dataset = TaskSplitter(test_set, task_labels)
            tasks.append((train_task_dataset, test_task_dataset))
        return tasks

    # --- Permuted MNIST Logic (Also modified for efficiency) ---
    elif benchmark_name.lower() == 'pmnist':
        logging.info("Preparing Permuted MNIST benchmark...")
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        
        full_train_set = datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
        test_set = datasets.MNIST(root=data_path, train=False, download=True, transform=transform)

        # --- EFFICIENCY BOOST FOR PMNIST ---
        subset_fraction = data_config.get('subset_fraction', 1.0)
        if subset_fraction < 1.0:
            num_samples = int(len(full_train_set) * subset_fraction)
            subset_indices = np.random.choice(len(full_train_set), num_samples, replace=False)
            train_set = Subset(full_train_set, subset_indices)
            logging.info(f"Using a {subset_fraction*100:.0f}% subset of the training data ({len(train_set)} samples).")
        else:
            train_set = full_train_set
        # --- END EFFICIENCY BOOST ---
        
        tasks = []
        pixel_indices = np.arange(28 * 28)
        for _ in range(num_tasks):
            permuted_indices = rng.permutation(pixel_indices)
            train_task_dataset = PermutedMNIST(train_set, permuted_indices)
            test_task_dataset = PermutedMNIST(test_set, permuted_indices)
            tasks.append((train_task_dataset, test_task_dataset))
        return tasks

    else:
        raise NotImplementedError(f"Benchmark '{benchmark_name}' is not supported yet.")