"""Elastic Weight Consolidation (EWC) implementation."""

from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel


class EWC:
    """
    Elastic Weight Consolidation for protecting important parameters.

    Implements both offline and online EWC variants.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        lambda_ewc: float = 100.0,
        online: bool = True,
        decay_factor: float = 0.9,
    ):
        """
        Initialize EWC.

        Args:
            model: Model to apply EWC to
            lambda_ewc: Weight for EWC regularization
            online: Whether to use online EWC (with decay)
            decay_factor: Decay factor for online EWC
        """
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.online = online
        self.decay_factor = decay_factor

        # Store Fisher information and optimal parameters
        self.fisher_info: Dict[str, torch.Tensor] = {}
        self.optimal_params: Dict[str, torch.Tensor] = {}

    def compute_fisher(
        self,
        dataloader,
        task_name: str,
        device: str = "cpu",
    ):
        """
        Compute Fisher information matrix for a task.

        Args:
            dataloader: DataLoader for the task
            task_name: Name of the task
            device: Device to use
        """
        self.model.eval()
        fisher = {}
        n_samples = 0

        # Initialize Fisher with zeros
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param.data)

        # Compute gradients squared
        for batch in dataloader:
            # Move batch to device
            if isinstance(batch, dict):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            else:
                batch = [item.to(device) if isinstance(item, torch.Tensor) else item for item in batch]

            self.model.zero_grad()

            # Forward pass
            if isinstance(batch, dict):
                outputs = self.model(**batch)
            else:
                outputs = self.model(*batch)

            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

            # Backward pass
            loss.backward()

            # Accumulate Fisher (diagonal approximation)
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.data ** 2

            n_samples += 1

        # Average Fisher
        for name in fisher:
            fisher[name] /= n_samples

        # Update Fisher (online or offline)
        if self.online and self.fisher_info:
            # Online EWC: exponential moving average
            for name in fisher:
                if name in self.fisher_info:
                    self.fisher_info[name] = (
                        self.decay_factor * self.fisher_info[name]
                        + (1 - self.decay_factor) * fisher[name]
                    )
                else:
                    self.fisher_info[name] = fisher[name]
        else:
            # Offline EWC: replace
            self.fisher_info = fisher

        # Store optimal parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.optimal_params[name] = param.data.clone()

    def compute_ewc_loss(self) -> torch.Tensor:
        """
        Compute EWC regularization loss.

        Returns:
            EWC loss tensor
        """
        if not self.fisher_info:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)

        ewc_loss = torch.tensor(0.0, device=next(self.model.parameters()).device)

        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.fisher_info:
                if name in self.optimal_params:
                    fisher = self.fisher_info[name]
                    optimal = self.optimal_params[name]
                    ewc_loss += torch.sum(fisher * (param - optimal) ** 2)

        return self.lambda_ewc * ewc_loss

    def save_checkpoint(self, path: str):
        """Save EWC state to checkpoint."""
        checkpoint = {
            "fisher_info": self.fisher_info,
            "optimal_params": self.optimal_params,
            "lambda_ewc": self.lambda_ewc,
            "online": self.online,
            "decay_factor": self.decay_factor,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Load EWC state from checkpoint."""
        checkpoint = torch.load(path)
        self.fisher_info = checkpoint["fisher_info"]
        self.optimal_params = checkpoint["optimal_params"]
        self.lambda_ewc = checkpoint.get("lambda_ewc", self.lambda_ewc)
        self.online = checkpoint.get("online", self.online)
        self.decay_factor = checkpoint.get("decay_factor", self.decay_factor)


