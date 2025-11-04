"""Loss functions for continual learning."""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CompositeLoss(nn.Module):
    """
    Composite loss function combining task loss, orthogonal regularization, and EWC.

    L_total = L_task + λ_ortho * L_ortho + λ_ewc * L_ewc
    """

    def __init__(
        self,
        lambda_ortho: float = 0.1,
        lambda_ewc: float = 100.0,
        use_orthogonal: bool = True,
        use_ewc: bool = True,
    ):
        """
        Initialize composite loss.

        Args:
            lambda_ortho: Weight for orthogonal regularization
            lambda_ewc: Weight for EWC regularization
            use_orthogonal: Whether to use orthogonal regularization
            use_ewc: Whether to use EWC regularization
        """
        super().__init__()
        self.lambda_ortho = lambda_ortho
        self.lambda_ewc = lambda_ewc
        self.use_orthogonal = use_orthogonal
        self.use_ewc = use_ewc

        # Task loss (cross-entropy)
        self.task_loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        orthogonal_loss: Optional[torch.Tensor] = None,
        ewc_loss: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute composite loss.

        Args:
            logits: Model predictions
            labels: True labels
            orthogonal_loss: Orthogonal regularization loss (optional)
            ewc_loss: EWC regularization loss (optional)

        Returns:
            Dictionary with individual loss components and total loss
        """
        # Task loss
        task_loss = self.task_loss_fn(logits, labels)

        # Orthogonal loss
        if self.use_orthogonal and orthogonal_loss is not None:
            ortho_loss = self.lambda_ortho * orthogonal_loss
        else:
            ortho_loss = torch.tensor(0.0, device=logits.device)

        # EWC loss
        if self.use_ewc and ewc_loss is not None:
            ewc_loss_val = self.lambda_ewc * ewc_loss
        else:
            ewc_loss_val = torch.tensor(0.0, device=logits.device)

        # Total loss
        total_loss = task_loss + ortho_loss + ewc_loss_val

        return {
            "total_loss": total_loss,
            "task_loss": task_loss,
            "orthogonal_loss": ortho_loss,
            "ewc_loss": ewc_loss_val,
        }


