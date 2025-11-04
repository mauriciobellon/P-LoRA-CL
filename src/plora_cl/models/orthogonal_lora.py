"""Orthogonal LoRA implementation."""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from peft import PeftModel

from .lora_adapters import LoRAAdapterManager


class OrthogonalLoRA(LoRAAdapterManager):
    """
    LoRA adapter manager with orthogonal constraints.

    Ensures that adapters for different tasks occupy orthogonal subspaces.
    """

    def __init__(
        self,
        base_model,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
        lambda_ortho: float = 0.1,
    ):
        """
        Initialize orthogonal LoRA manager.

        Args:
            base_model: Base model
            r: Rank of LoRA matrices
            lora_alpha: LoRA alpha scaling
            lora_dropout: LoRA dropout rate
            target_modules: Target modules for LoRA
            lambda_ortho: Weight for orthogonal regularization
        """
        super().__init__(
            base_model=base_model,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
        )
        self.lambda_ortho = lambda_ortho

    def compute_orthogonal_loss(
        self,
        current_task: str,
        previous_tasks: List[str],
    ) -> torch.Tensor:
        """
        Compute orthogonal regularization loss.

        Penalizes projections of current task adapter onto previous task adapters.

        Args:
            current_task: Name of current task
            previous_tasks: List of previous task names

        Returns:
            Orthogonal loss tensor
        """
        if not previous_tasks:
            return torch.tensor(0.0, device=next(self.original_base_model.parameters()).device)

        # Get current adapter weights
        current_adapter = self.get_current_adapter()
        if current_adapter is None:
            return torch.tensor(0.0, device=next(self.original_base_model.parameters()).device)

        total_loss = torch.tensor(0.0, device=next(self.original_base_model.parameters()).device)

        # Get LoRA weights from current adapter
        current_lora_weights = self._extract_lora_weights(current_adapter)

        for prev_task in previous_tasks:
            # Get previous task adapter
            prev_adapter = self.get_adapter(prev_task)
            if prev_adapter is None:
                continue

            prev_lora_weights = self._extract_lora_weights(prev_adapter)

            # Compute orthogonal loss
            for module_name in current_lora_weights:
                if module_name not in prev_lora_weights:
                    continue

                current_A = current_lora_weights[module_name]["lora_A"]
                prev_A = prev_lora_weights[module_name]["lora_A"]

                # Compute projection: ||Proj_span(prev_A)(current_A)||^2
                # This is approximated by ||current_A @ prev_A.T||^2
                if current_A.shape == prev_A.shape:
                    projection = torch.matmul(current_A, prev_A.T)
                    total_loss += torch.norm(projection) ** 2

        return self.lambda_ortho * total_loss

    def _extract_lora_weights(self, adapter: PeftModel) -> Dict[str, Dict]:
        """
        Extract LoRA weight matrices from adapter.

        Args:
            adapter: PEFT model with LoRA

        Returns:
            Dictionary mapping module names to LoRA weights
        """
        weights = {}
        for name, module in adapter.named_modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                weights[name] = {
                    "lora_A": module.lora_A.weight.data,
                    "lora_B": module.lora_B.weight.data,
                }
        return weights

    def apply_orthogonal_projection(
        self,
        current_task: str,
        previous_tasks: List[str],
    ):
        """
        Apply Gram-Schmidt orthogonalization to ensure orthogonality.

        Args:
            current_task: Name of current task
            previous_tasks: List of previous task names
        """
        # This is a more aggressive approach that directly orthogonalizes
        # For now, we rely on regularization loss
        # Can be implemented if needed
        pass
