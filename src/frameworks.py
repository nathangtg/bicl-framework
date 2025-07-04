import torch
import torch.nn as nn
import numpy as np
import time

class EnhancedUnifiedFramework:
    def __init__(self, model, config):
        self.model = model
        self.config = config['frameworks']['unified']
        self.beta = self.config['beta_values'][0]
        self.divergence_type = self.config['divergence_type']
        self.memory_buffer_size = self.config['memory_buffer_size']
        self.replay_ratio = self.config['replay_ratio']
        self.theta_star, self.memory_buffer, self.task_importance = None, [], {}
        self.adaptive_beta, self.performance_history = self.beta, []

    def set_reference_parameters(self, task_id=0):
        self.theta_star = {name: param.clone().detach() for name, param in self.model.named_parameters()}
        self.task_importance[task_id] = self._compute_parameter_importance()

    def _compute_parameter_importance(self):
        return {name: p.grad.data.abs().clone() if p.grad is not None else torch.zeros_like(p) 
                for name, p in self.model.named_parameters()}

    def adaptive_beta_update(self, current_performance):
        if self.performance_history:
            perf_trend = current_performance - np.mean(self.performance_history[-3:])
            if perf_trend < -0.1: self.adaptive_beta *= 1.2
            elif perf_trend > 0.05: self.adaptive_beta *= 0.9
        self.adaptive_beta = np.clip(self.adaptive_beta, 0.01, 100.0)
        self.performance_history.append(current_performance)

    def _compute_divergence(self, task_id=0):
        if self.divergence_type == 'l2':
            divergence = 0.0
            importance = self.task_importance.get(task_id, {})
            for name, param in self.model.named_parameters():
                if name in self.theta_star:
                    diff = param - self.theta_star[name]
                    weighted_diff = importance.get(name, 1.0) * diff
                    divergence += torch.norm(weighted_diff)**2
            return divergence
        elif self.divergence_type == 'adaptive':
            l2_div, kl_div = 0.0, 0.0
            for name, param in self.model.named_parameters():
                if name in self.theta_star:
                    l2_div += torch.norm(param - self.theta_star[name])**2
                    if len(param.shape) > 1:
                        p_old = torch.softmax(self.theta_star[name].flatten(), dim=0)
                        p_new = torch.softmax(param.flatten(), dim=0)
                        kl_div += torch.sum(p_old * torch.log(p_old / (p_new + 1e-8)))
            return 0.7 * l2_div + 0.3 * kl_div
        return 0.0

    def _experience_replay_loss(self, current_loss, device):
        if not self.memory_buffer: return current_loss
        replay_samples = min(len(self.memory_buffer), int(len(self.memory_buffer) * self.replay_ratio))
        if replay_samples == 0: return current_loss
        
        replay_indices = np.random.choice(len(self.memory_buffer), replay_samples, replace=False)
        replay_loss = 0.0
        
        was_training = self.model.training
        self.model.eval()
        for idx in replay_indices:
            x_replay, y_replay = self.memory_buffer[idx]
            x_replay, y_replay = x_replay.to(device), y_replay.to(device)
            with torch.enable_grad():
                outputs = self.model(x_replay.unsqueeze(0))
                replay_loss += nn.functional.cross_entropy(outputs, y_replay.unsqueeze(0))
        if was_training: self.model.train()
        
        return current_loss + 0.5 * (replay_loss / len(replay_indices))

    def unified_loss(self, new_task_loss, device, task_id=0):
        if self.theta_star is None: return new_task_loss
        stability_term = self._compute_divergence(task_id)
        replay_loss = self._experience_replay_loss(new_task_loss, device)
        return replay_loss + self.adaptive_beta * stability_term

    def add_to_memory(self, x, y, max_samples=10):
        indices = np.random.choice(len(x), min(max_samples, len(x)), replace=False)
        for idx in indices:
            if len(self.memory_buffer) < self.memory_buffer_size:
                self.memory_buffer.append((x[idx].cpu(), y[idx].cpu()))
            else:
                replace_idx = np.random.randint(0, self.memory_buffer_size)
                self.memory_buffer[replace_idx] = (x[idx].cpu(), y[idx].cpu())

class EnhancedEWC:
    def __init__(self, model, config):
        self.model = model
        self.config = config['frameworks']['ewc']
        self.lambda_reg = self.config['lambda']
        self.fisher_info, self.optimal_params, self.task_count = {}, {}, 0

    def compute_fisher_information(self, dataloader, device):
        fisher = {name: torch.zeros_like(p) for name, p in self.model.named_parameters() if p.requires_grad}
        self.model.eval()
        sample_count = 0
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            self.model.zero_grad()
            output = self.model(data)
            log_likelihoods = nn.functional.log_softmax(output, dim=1)
            sampled_labels = torch.multinomial(torch.exp(log_likelihoods), 1).squeeze()
            log_likelihood = log_likelihoods.gather(1, sampled_labels.unsqueeze(-1)).squeeze()
            for ll in log_likelihood:
                self.model.zero_grad()
                ll.backward(retain_graph=True)
                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        fisher[name] += param.grad.data ** 2
            sample_count += data.shape[0]

        for name in fisher: fisher[name] /= sample_count
        if self.task_count > 0:
            for name in self.fisher_info: self.fisher_info[name] += fisher[name]
        else: self.fisher_info = fisher

    def save_optimal_params(self):
        self.optimal_params = {name: p.clone().detach() for name, p in self.model.named_parameters() if p.requires_grad}
        self.task_count += 1

    def ewc_loss(self, current_loss: torch.Tensor) -> torch.Tensor:
        if self.task_count == 0: return current_loss
        penalty = sum(torch.sum(self.fisher_info[n]*(p-self.optimal_params[n])**2)
                      for n, p in self.model.named_parameters() if n in self.optimal_params)
        return current_loss + (self.lambda_reg / 2) * penalty