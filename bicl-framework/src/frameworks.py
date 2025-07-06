# src/frameworks.py
import torch
import torch.nn as nn
from typing import Dict, Any
import logging

class BICLFramework:
    """
    The FINAL, STABLE, and CORRECT implementation of the BICL framework.

    This version is architected to work WITH the PyTorch autograd engine by
    separating the loss calculation from the importance weight update.
    It relies on the two most impactful and robust mechanisms: Synaptic
    Consolidation and Homeostatic Regulation.
    """
    def __init__(self, model: nn.Module, config: Dict[str, Any], device: torch.device):
        """
        Initializes the BICL framework.
        """
        self.model = model
        self.config = config['frameworks']['bicl']
        self.device = device
        logging.info("  > Initializing FINAL STABLE BICLFramework.")

        # Snapshot of parameters from the previous task (θ*)
        self.theta_star: Dict[str, torch.Tensor] = {
            n: p.clone().detach().to(self.device) for n, p in self.model.named_parameters()
        }

        # Per-parameter importance weights (Ω)
        self.importance: Dict[str, torch.Tensor] = {
            n: torch.zeros_like(p, device=self.device) for n, p in self.model.named_parameters()
        }

    def calculate_loss(self, new_task_loss: torch.Tensor) -> torch.Tensor:
        """
        Calculates the regularized BICL loss. This function does NOT compute
        gradients internally. It relies on importance weights from the *previous*
        training step, making it compatible with the standard PyTorch autograd flow.

        Args:
            new_task_loss (torch.Tensor): The loss on the current task (L_new).

        Returns:
            torch.Tensor: The final, regularized loss to be used for `loss.backward()`.
        """
        # --- 1. Synaptic Consolidation (Stability Term) ---
        stability_loss = 0.0
        beta_stability = self.config['beta_stability']
        for name, param in self.model.named_parameters():
            if name in self.theta_star:
                stability_loss += torch.sum(self.importance[name] * (param - self.theta_star[name])**2)
        
        # --- 2. Homeostatic Regulation Regularizer ---
        homeo_reg_loss = 0.0
        gamma_homeo = self.config['gamma_homeo']
        alpha = self.config['homeostasis_alpha']
        beta_h = self.config['homeostasis_beta_h']
        tau = self.config['homeostasis_tau']
        for param in self.model.parameters():
            penalty = alpha * torch.tanh(beta_h * param) * torch.sigmoid(torch.abs(param) - tau)
            homeo_reg_loss += torch.sum(penalty**2)

        # --- 3. Combine losses ---
        return new_task_loss + (beta_stability * stability_loss) + (gamma_homeo * homeo_reg_loss)

    def after_backward_update(self):
        """
        **CRITICAL STEP:** This method MUST be called *after* `loss.backward()` and
        *before* `optimizer.step()`.

        It uses the gradients that `loss.backward()` has just computed and stored
        in `param.grad` to update the importance weights for the *next* step.
        """
        decay = self.config['importance_decay']
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # Update importance: Ω_t ← γ*Ω_{t-1} + (1-γ)*(∂L_t/∂θ_t)^2
                # This uses the fresh gradient from the backward pass.
                self.importance[name] = (decay * self.importance[name] + 
                                         (1 - decay) * param.grad.data.abs()**2)

    def on_task_finish(self):
        """
        Updates the reference parameters (θ*) for the next task. This should be
        called after all epochs for a task are complete.
        """
        self.theta_star = {
            n: p.clone().detach().to(self.device) for n, p in self.model.named_parameters()
        }


# ==============================================================================
# Elastic Weight Consolidation (EWC) Framework (Unchanged, for baseline)
# ==============================================================================

class EWC:
    """
    A robust and clean implementation of Elastic Weight Consolidation (EWC).
    """
    def __init__(self, model: nn.Module, config: Dict[str, Any], device: torch.device):
        self.model = model
        self.config = config['frameworks']['ewc']
        self.device = device
        self.lambda_reg = self.config['lambda']
        self.fisher_info: Dict[str, torch.Tensor] = {}
        self.optimal_params: Dict[str, torch.Tensor] = {}
        self.task_count = 0

    def compute_fisher_information(self, dataloader: torch.utils.data.DataLoader):
        fisher = {n: torch.zeros_like(p, device=self.device) 
                  for n, p in self.model.named_parameters() if p.requires_grad}
        self.model.eval()
        sample_count = 0
        for data, target in dataloader:
            data, target = data.to(self.device), target.to(self.device)
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

        for name in fisher:
            fisher[name] /= sample_count
        
        if self.task_count > 0:
            for name in self.fisher_info:
                self.fisher_info[name] += fisher[name]
        else:
            self.fisher_info = fisher

    def ewc_loss(self, current_loss: torch.Tensor) -> torch.Tensor:
        if self.task_count == 0:
            return current_loss
        penalty = sum(torch.sum(self.fisher_info[n] * (p - self.optimal_params[n])**2)
                      for n, p in self.model.named_parameters() if n in self.optimal_params)
        return current_loss + (self.lambda_reg / 2) * penalty

    def on_task_finish(self, dataloader: torch.utils.data.DataLoader):
        logging.info("    Computing Fisher Information for EWC...")
        self.compute_fisher_information(dataloader)
        self.optimal_params = {n: p.clone().detach().to(self.device) 
                               for n, p in self.model.named_parameters() if p.requires_grad}
        self.task_count += 1