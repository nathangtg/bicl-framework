import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import time
import numpy as np
from collections import defaultdict

from .model import get_model
from .frameworks import EnhancedUnifiedFramework, EnhancedEWC
from .data import generate_diverse_tasks
from .utils import reset_weights

class ExperimentRunner:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def _train_one_task(self, model, train_dataset, method, cl_framework, beta, task_id):
        train_config = self.config['training']
        optimizer = optim.Adam(model.parameters(), lr=train_config['learning_rate'], weight_decay=train_config['weight_decay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

        val_size = len(train_dataset) // 5
        train_size = len(train_dataset) - val_size
        train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
        train_loader = DataLoader(train_subset, batch_size=train_config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=train_config['batch_size'])

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(train_config['epochs']):
            model.train()
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                base_loss = nn.functional.cross_entropy(outputs, batch_y)
                
                if method == 'unified':
                    cl_framework.add_to_memory(batch_X, batch_y)
                    total_loss = cl_framework.unified_loss(base_loss, self.device, task_id)
                elif method == 'ewc':
                    total_loss = cl_framework.ewc_loss(base_loss)
                else: # vanilla
                    total_loss = base_loss
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config['gradient_clip_norm'])
                optimizer.step()

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    val_loss += nn.functional.cross_entropy(model(batch_X), batch_y).item()
            avg_val_loss = val_loss / len(val_loader)
            scheduler.step(avg_val_loss)

            if train_config['early_stopping']['enabled']:
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= train_config['early_stopping']['patience']:
                    print(f"  Early stopping at epoch {epoch + 1}")
                    break
        return

    def _evaluate_one_task(self, model, test_dataset):
        model.eval()
        correct, total = 0, 0
        loader = DataLoader(test_dataset, batch_size=self.config['training']['batch_size'])
        with torch.no_grad():
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        return correct / total

    def run_comprehensive_experiment(self):
        exp_config = self.config['experiment']
        methods = exp_config['methods_to_run']
        num_tasks = self.config['data']['num_tasks']
        
        all_results = {m: {f'task_{i}': [] for i in range(num_tasks)} for m in methods}
        forgetting_results = {m: [] for m in methods}
        complexity_results = {m: {'time': [], 'fisher_time': []} for m in methods}

        print(f"Running {exp_config['num_runs']} experiments with {num_tasks} tasks each...")
        for run in range(exp_config['num_runs']):
            print(f"\n🔄 RUN {run + 1}/{exp_config['num_runs']}")
            tasks = generate_diverse_tasks(self.config)
            
            for method in methods:
                print(f"  Testing {method.upper()}...")
                model = get_model(self.config).to(self.device)
                model.apply(reset_weights)
                
                cl_framework = None
                if method == 'ewc': cl_framework = EnhancedEWC(model, self.config)
                elif method == 'unified': cl_framework = EnhancedUnifiedFramework(model, self.config)
                
                method_time, fisher_time = 0, 0
                task_accuracies = []
                beta = self.config['frameworks']['unified']['beta_values'][run % len(self.config['frameworks']['unified']['beta_values'])]

                for task_idx, (train_ds, test_ds) in enumerate(tasks):
                    print(f"    Task {task_idx + 1}/{num_tasks} - ", end="")
                    if method == 'ewc' and task_idx > 0:
                        prev_train_ds, _ = tasks[task_idx-1]
                        fisher_start = time.time()
                        cl_framework.compute_fisher_information(DataLoader(prev_train_ds, batch_size=64), self.device)
                        fisher_time += time.time() - fisher_start
                        cl_framework.save_optimal_params()
                    
                    start_time = time.time()
                    self._train_one_task(model, train_ds, method, cl_framework, beta, task_idx)
                    method_time += time.time() - start_time
                    
                    current_acc = self._evaluate_one_task(model, test_ds)
                    task_accuracies.append(current_acc)
                    print(f"Acc: {current_acc:.3f}")
                    
                    if method == 'unified' and task_idx > 0:
                        cl_framework.set_reference_parameters(task_id=task_idx)

                final_accuracies = [self._evaluate_one_task(model, test_ds) for _, test_ds in tasks]
                for i, acc in enumerate(final_accuracies): all_results[method][f'task_{i}'].append(acc)
                
                forgetting = np.mean([max(0, task_accuracies[i] - final_accuracies[i]) for i in range(len(task_accuracies))])
                forgetting_results[method].append(forgetting)
                complexity_results[method]['time'].append(method_time)
                complexity_results[method]['fisher_time'].append(fisher_time)
                print(f"    Forget: {forgetting:.3f}, Time: {method_time:.1f}s")
                
        return all_results, forgetting_results, complexity_results