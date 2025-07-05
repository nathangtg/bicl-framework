# src/frameworks.py
import torch
import torch.nn as nn
from typing import Dict, Any

class BICLFramework:
    """
    A professional and validated implementation of the Bio-Inspired Continual 
    Learning (BICL) framework, as described in "Moving On: Toward a 
    Neurocomputational Theory of Adaptive Forgetting."

    This framework translates core principles of neural plasticity into
    differentiable regularization terms to mitigate catastrophic forgetting.
    """
    def __init__(self, model: nn.Module, config: Dict[str, Any], device: torch.device):
        """
        Initializes the BICL framework.

        Args:
            model (nn.Module): The neural network model to be trained.
            config (Dict[str, Any]): The configuration dictionary, expected to
                                     contain a 'bicl' key with framework-specific
                                     hyperparameters.
            device (torch.device): The device (CPU or CUDA) to which tensors will be moved.
        """
        self.model = model
        self.config = config['frameworks']['bicl']
        self.device = device

        # A snapshot of the model parameters from the previous task (θ*).
        # This is crucial for the stability-plasticity penalty.
        self.theta_star: Dict[str, torch.Tensor] = {
            n: p.clone().detach().to(self.device) for n, p in self.model.named_parameters()
        }

        # Per-parameter importance weights (Ω), representing synaptic consolidation.
        # Implements the principles of long-term potentiation.
        self.importance: Dict[str, torch.Tensor] = {
            n: torch.zeros_like(p, device=self.device) for n, p in self.model.named_parameters()
        }

    def update_importance(self, grads: tuple[torch.Tensor, ...]):
        """
        Updates the synaptic importance weights (Ω) using the gradients from the
        current task's loss. This models synaptic consolidation, where synapses
        that are critical for the current task are strengthened.

        This is an efficient implementation of Equation (4) from the paper,
        using an exponential moving average to track parameter sensitivity.

        Args:
            grads (tuple[torch.Tensor, ...]): A tuple of gradients of the new task
                                              loss with respect to model parameters.
        """
        decay = self.config['importance_decay']
        for (name, _), grad in zip(self.model.named_parameters(), grads):
            if grad is not None:
                # Update importance: Ω_i ← γ * Ω_i + (1-γ) * (∂L/∂θ_i)^2
                self.importance[name] = (decay * self.importance[name] + 
                                         (1 - decay) * grad.data.abs()**2)

    def bicl_loss(self, new_task_loss: torch.Tensor) -> torch.Tensor:
        """
        Computes the complete, unified BICL loss function as defined by
        Equation (6) in the paper.

        L_BICL = L_new + β * Σ Ω(θ - θ*)^2 + γ * R_homo(θ) + δ * R_decay(θ)

        Args:
            new_task_loss (torch.Tensor): The loss on the current task (L_new).

        Returns:
            torch.Tensor: The final, regularized loss to be used for backpropagation.
        """
        # --- 1. Calculate Gradients for Regularizers ---
        # We need the gradients of the new task loss to compute importance (Ω)
        # and usage for adaptive forgetting. We compute them here once.
        grads = torch.autograd.grad(
            new_task_loss, self.model.parameters(), retain_graph=True
        )

        # --- 2. Synaptic Consolidation (Stability Term) ---
        # This is the core stability term from EWC-like methods, weighted by our
        # biologically-plausible importance measure Ω.
        # Term: β * Σ Ω(θ - θ*)^2
        stability_loss = 0.0
        beta_stability = self.config['beta_stability']
        for name, param in self.model.named_parameters():
            if name in self.theta_star:
                stability_loss += torch.sum(self.importance[name] * (param - self.theta_star[name])**2)
        
        # --- 3. Homeostatic Regulation Regularizer ---
        # Models the biological process where neurons maintain activity within a
        # stable range. This prevents parameter drift into non-functional regimes.
        # Term: γ * R_homo(θ) where R_homo = ||θ - θ_homo||^2
        homeo_reg_loss = 0.0
        gamma_homeo = self.config['gamma_homeo']
        alpha = self.config['homeostasis_alpha']
        beta_h = self.config['homeostasis_beta_h']
        tau = self.config['homeostasis_tau']

        for param in self.model.parameters():
            # This is the regularization penalty derived from Eq. 3: ||α*tanh(βh*θ)*σ(|θ|-τ)||^2
            penalty = alpha * torch.tanh(beta_h * param) * torch.sigmoid(torch.abs(param) - tau)
            homeo_reg_loss += torch.sum(penalty**2)

        # --- 4. Adaptive Forgetting Regularizer ---
        # Models synaptic pruning, where unused connections decay over time.
        # This frees up neural resources for new learning.
        # Term: δ * R_decay(θ) where R_decay = ||θ - θ_decay||^2
        forget_reg_loss = 0.0
        delta_forget = self.config['delta_forget']
        lambda_forget = self.config['forget_lambda']

        for (name, param), grad in zip(self.model.named_parameters(), grads):
            if grad is not None:
                # Parameter usage is modeled as the magnitude of its gradient.
                usage = torch.tanh(grad.data.abs())
                
                # This is the regularization penalty derived from Eq. 5: ||θ*(1-exp(-λ*(1-usage)))||^2
                # Assuming Δt = 1 per training step.
                penalty = param * (1 - torch.exp(-lambda_forget * (1 - usage)))
                forget_reg_loss += torch.sum(penalty**2)
        
        # --- 5. Update Importance Weights for Next Iteration ---
        self.update_importance(grads)

        # --- 6. Combine All Losses into the Final BICL Objective Function ---
        final_loss = (new_task_loss +
                      beta_stability * stability_loss +
                      gamma_homeo * homeo_reg_loss +
                      delta_forget * forget_reg_loss)
        
        return final_loss

    def on_task_finish(self):
        """
        Should be called after training on a task is complete. This method
        updates the reference parameters (θ*) to the newly learned optimal state.
        """
        self.theta_star = {
            n: p.clone().detach().to(self.device) for n, p in self.model.named_parameters()
        }


# ==============================================================================
# Elastic Weight Consolidation (EWC) Framework
# ==============================================================================

class EWC:
    """
    A robust and clean implementation of Elastic Weight Consolidation (EWC),
    a key baseline for continual learning.
    """
    def __init__(self, model: nn.Module, config: Dict[str, Any], device: torch.device):
        """
        Initializes the EWC framework.
        """
        self.model = model
        self.config = config['frameworks']['ewc']
        self.device = device
        self.lambda_reg = self.config['lambda']
        
        self.fisher_info: Dict[str, torch.Tensor] = {}
        self.optimal_params: Dict[str, torch.Tensor] = {}
        self.task_count = 0

    def compute_fisher_information(self, dataloader: torch.utils.data.DataLoader):
        """
        Computes the Fisher Information Matrix for the current model parameters.
        This is done by averaging the squared gradients of the log-likelihoods.
        """
        fisher = {n: torch.zeros_like(p, device=self.device) 
                  for n, p in self.model.named_parameters() if p.requires_grad}
        
        self.model.eval()
        sample_count = 0
        for data, target in dataloader:
            data, target = data.to(self.device), target.to(self.device)
            self.model.zero_grad()
            
            output = self.model(data)
            log_likelihoods = nn.functional.log_softmax(output, dim=1)
            
            # Sample from the model's output distribution to get log-likelihoods
            sampled_labels = torch.multinomial(torch.exp(log_likelihoods), 1).squeeze()
            log_likelihood = log_likelihoods.gather(1, sampled_labels.unsqueeze(-1)).squeeze()
            
            for ll in log_likelihood:
                self.model.zero_grad()
                ll.backward(retain_graph=True)
                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        fisher[name] += param.grad.data ** 2
            
            sample_count += data.shape[0]

        # Normalize by the number of samples
        for name in fisher:
            fisher[name] /= sample_count
        
        # Accumulate Fisher information across tasks
        if self.task_count > 0:
            for name in self.fisher_info:
                self.fisher_info[name] += fisher[name]
        else:
            self.fisher_info = fisher

    def ewc_loss(self, current_loss: torch.Tensor) -> torch.Tensor:
        """
        Calculates the EWC loss by adding the quadratic penalty term.
        """
        if self.task_count == 0:
            return current_loss
            
        penalty = sum(torch.sum(self.fisher_info[n] * (p - self.optimal_params[n])**2)
                      for n, p in self.model.named_parameters() if n in self.optimal_params)
                      
        return current_loss + (self.lambda_reg / 2) * penalty

    def on_task_finish(self, dataloader: torch.utils.data.DataLoader):
        """
        To be called after a task is finished. Computes the Fisher info for the
        completed task and saves the optimal parameters.
        """
        print("    Computing Fisher Information for EWC...")
        self.compute_fisher_information(dataloader)
        self.optimal_params = {n: p.clone().detach().to(self.device) 
                               for n, p in self.model.named_parameters() if p.requires_grad}
        self.task_count += 1
