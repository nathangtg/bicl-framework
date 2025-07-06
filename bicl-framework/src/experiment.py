# src/experiment.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import time
import numpy as np
import logging
import itertools
import copy
from collections import defaultdict
from typing import Dict, Any, Tuple, List

# --- Correctly structured imports ---
from .model import get_model
from .frameworks import BICLFramework, EWC
from .data import get_benchmark_tasks, generate_diverse_tasks
from .utils import reset_weights

class ExperimentRunner:
    """
    Orchestrates the entire continual learning experiment pipeline.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the experiment runner with a configuration dictionary.
        """
        self.config = config
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
            
        logging.info(f"ExperimentRunner initialized. Using device: {self.device}")

    def _train_one_task(self, model: nn.Module, train_dataset: DataLoader, method: str, 
                        cl_framework: Any, task_id: int) -> None:
        """
        Handles the complete training and validation loop for a single task.
        
        **This method contains the corrected training loop for BICL.**
        """
        train_config = self.config['training']
        optimizer = optim.Adam(model.parameters(), lr=train_config['learning_rate'], 
                               weight_decay=train_config['weight_decay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

        val_size = max(1, len(train_dataset) // 5)
        train_size = len(train_dataset) - val_size
        
        if train_size <= 0: # Handle very small datasets
            train_subset, val_subset = train_dataset, train_dataset
        else:
            train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_subset, batch_size=train_config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=train_config['batch_size'])

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(train_config['epochs']):
            model.train()
            # ======================================================================
            # --- THE FINAL, CORRECTED TRAINING LOOP ---
            # ======================================================================
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                
                # 1. Standard forward pass to get base loss
                outputs = model(batch_X)
                base_loss = nn.functional.cross_entropy(outputs, batch_y)
                
                # 2. Apply CL framework to calculate the total, regularized loss
                if method == 'bicl':
                    # Use the new `calculate_loss` method
                    total_loss = cl_framework.calculate_loss(base_loss)
                elif method == 'ewc':
                    total_loss = cl_framework.ewc_loss(base_loss)
                else:  # 'vanilla'
                    total_loss = base_loss
                
                # 3. Standard backward pass on the total loss
                total_loss.backward()

                # 4. **CRITICAL NEW STEP FOR BICL**
                #    Update importance weights using the gradients from .backward()
                if method == 'bicl':
                    cl_framework.after_backward_update()
                
                # 5. Clip gradients and take an optimizer step
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config['gradient_clip_norm'])
                optimizer.step()
            # ======================================================================
            # --- END OF CORRECTED LOOP ---
            # ======================================================================

            # --- Validation loop (unchanged) ---
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    # Check if val_loader is not empty before calculating loss
                    if len(val_loader) > 0:
                        val_loss += nn.functional.cross_entropy(model(batch_X), batch_y).item()
            
            avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
            scheduler.step(avg_val_loss)

            if train_config['early_stopping']['enabled']:
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= train_config['early_stopping']['patience']:
                    logging.debug(f"    Early stopping at epoch {epoch + 1}")
                    break

    # The rest of your ExperimentRunner class is perfectly fine and requires no changes.
    # _evaluate_on_tasks, _calculate_metrics, _generate_trials, and 
    # run_comprehensive_experiment are all correct.

    def _evaluate_on_tasks(self, model: nn.Module, tasks: list) -> Dict[int, float]:
        """Evaluates the model's performance on a list of test datasets."""
        model.eval()
        accuracies = {}
        with torch.no_grad():
            for i, (_, test_ds) in enumerate(tasks):
                correct, total = 0, 0
                loader = DataLoader(test_ds, batch_size=self.config['training']['batch_size'])
                for batch_X, batch_y in loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_X)
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
                accuracies[i] = correct / total if total > 0 else 0
        return accuracies

    def _calculate_metrics(self, task_results: Dict[int, Dict[int, float]]) -> Tuple[float, float]:
        """Calculates average accuracy and Backward Transfer (BWT)."""
        num_tasks = len(task_results)
        if num_tasks == 0: return 0.0, 0.0
        
        final_accuracies = [task_results[num_tasks - 1].get(i, 0) for i in range(num_tasks)]
        avg_accuracy = np.mean(final_accuracies)
        
        bwt = 0.0
        if num_tasks > 1:
            for i in range(num_tasks - 1):
                acc_after_final = task_results[num_tasks - 1].get(i, 0)
                acc_at_learning = task_results[i].get(i, 0)
                bwt += (acc_after_final - acc_at_learning)
            bwt /= (num_tasks - 1)

        return avg_accuracy, bwt

    def _generate_trials(self) -> List[Dict[str, Any]]:
        """
        Generates a list of experiment trials based on hyperparameter lists
        in the configuration file. This enables grid search.
        """
        trials = []
        methods = self.config['experiment']['methods_to_run']
        logging.info("Generating hyperparameter trials...")
        
        for method in methods:
            logging.debug(f"  Processing method: '{method}'")
            framework_config = self.config['frameworks'].get(method, {})
            if not framework_config:
                logging.debug(f"    -> No framework config found. Creating a single trial.")
                trials.append({'id': method, 'method': method, 'params': {}})
                continue

            param_lists = {k: v for k, v in framework_config.items() if isinstance(v, (list, tuple))}
            fixed_params = {k: v for k, v in framework_config.items() if not isinstance(v, (list, tuple))}
            
            if not param_lists:
                trials.append({'id': method, 'method': method, 'params': fixed_params})
                continue

            param_names = list(param_lists.keys())
            param_combinations = list(itertools.product(*param_lists.values()))
            
            for combo in param_combinations:
                trial_params = fixed_params.copy()
                trial_id_parts = [method]
                for name, value in zip(param_names, combo):
                    trial_params[name] = value
                    short_name = ''.join([c for c in name if c.isupper()]) or name.replace('_', '')[:4]
                    trial_id_parts.append(f"{short_name}{value}")
                
                trials.append({
                    'id': "_".join(trial_id_parts),
                    'method': method,
                    'params': trial_params
                })
                
        logging.info(f"Generated {len(trials)} unique trials for the experiment.")
        for trial in trials:
            logging.info(f"  - Trial generated: {trial['id']}")
        return trials

    def run_comprehensive_experiment(self) -> Tuple[Dict, Dict, Dict]:
        """
        Runs the full, multi-run statistical experiment, iterating through all
        generated hyperparameter trials.
        """
        exp_config = self.config['experiment']
        num_tasks = self.config['data']['num_tasks']
        
        trials = self._generate_trials()
        
        all_accuracies = {t['id']: [] for t in trials}
        all_bwt = {t['id']: [] for t in trials}
        all_complexity = {t['id']: defaultdict(list) for t in trials}

        for run in range(exp_config['num_runs']):
            logging.info(f"--- Starting Statistical Run {run + 1}/{exp_config['num_runs']} ---")
            
            if 'benchmark' in self.config['data']:
                logging.info(f"Loading benchmark dataset: {self.config['data']['benchmark']}")
                tasks = get_benchmark_tasks(self.config)
            else:
                logging.info("Generating diverse synthetic tasks...")
                tasks = generate_diverse_tasks(self.config)
            
            for i, trial in enumerate(trials):
                trial_id = trial['id']
                method = trial['method']
                
                logging.info(f"  -- Starting Trial {i+1}/{len(trials)}: {trial_id} --")
                model = get_model(self.config).to(self.device)
                model.apply(reset_weights)
                
                trial_config = copy.deepcopy(self.config)
                trial_config['frameworks'][method] = trial['params']

                cl_framework = None
                if method == 'ewc':
                    cl_framework = EWC(model, trial_config, self.device)
                elif method == 'bicl':
                    cl_framework = BICLFramework(model, trial_config, self.device)
                
                method_time, fisher_time = 0, 0
                task_results_for_run = defaultdict(dict)

                for task_idx in range(num_tasks):
                    train_ds, _ = tasks[task_idx]
                    logging.info(f"    Training on Task {task_idx + 1}/{num_tasks}...")
                    
                    start_time = time.time()
                    self._train_one_task(model, train_ds, method, cl_framework, task_idx)
                    method_time += time.time() - start_time
                    
                    current_accuracies = self._evaluate_on_tasks(model, tasks[:task_idx+1])
                    task_results_for_run[task_idx] = current_accuracies
                    logging.debug(f"      Accuracies after task {task_idx+1}: {current_accuracies}")

                    if cl_framework and hasattr(cl_framework, 'on_task_finish'):
                        if method == 'ewc':
                            loader = DataLoader(train_ds, batch_size=self.config['training']['batch_size'])
                            fisher_start = time.time()
                            cl_framework.on_task_finish(loader)
                            fisher_time += time.time() - fisher_start
                        else:
                            cl_framework.on_task_finish()

                avg_acc, bwt = self._calculate_metrics(task_results_for_run)
                all_accuracies[trial_id].append(avg_acc)
                all_bwt[trial_id].append(bwt)
                all_complexity[trial_id]['time'].append(method_time)
                all_complexity[trial_id]['fisher_time'].append(fisher_time)
                
                logging.info(f"    Trial complete. Avg Acc: {avg_acc:.3f}, BWT: {bwt:.3f}, Time: {method_time:.1f}s")
                
        return all_accuracies, all_bwt, all_complexity