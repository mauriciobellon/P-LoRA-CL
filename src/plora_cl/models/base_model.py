"""Base model wrapper for continual learning."""

from typing import Optional

import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)


class BaseCLModel(nn.Module):
    """
    Base model wrapper for continual learning.

    Supports task-aware inference with separate classification heads per task.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        device: str = "cpu",
        freeze_base: bool = True,
    ):
        """
        Initialize the base model.

        Args:
            model_name: Name of the base model
            device: Device to use
            freeze_base: Whether to freeze base model parameters
        """
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.freeze_base = freeze_base

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load base model WITHOUT classification head
        # We'll add task-specific heads separately
        self.base_model = AutoModel.from_pretrained(model_name)

        # Freeze base if requested
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False

        # Task-specific classification heads
        self.task_heads = nn.ModuleDict()

        # Store task configs
        self.task_configs = {}

    def add_task_head(self, task_name: str, num_classes: int):
        """
        Add a classification head for a specific task.

        Args:
            task_name: Name of the task
            num_classes: Number of classes for this task
        """
        # Get hidden size from base model
        hidden_size = self.base_model.config.hidden_size

        # Create classification head
        head = nn.Linear(hidden_size, num_classes)
        self.task_heads[task_name] = head

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        task_name: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            task_name: Name of the task (required for task-aware inference)

        Returns:
            Logits for the specified task
        """
        if task_name is None:
            raise ValueError("task_name must be provided for task-aware inference")

        if task_name not in self.task_heads:
            raise ValueError(f"Task head for {task_name} not found")

        # Get base model outputs
        outputs = self.base_model.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Use [CLS] token
        pooled_output = outputs.last_hidden_state[:, 0]

        # Pass through task-specific head
        logits = self.task_heads[task_name](pooled_output)

        return logits

    def get_tokenizer(self) -> PreTrainedTokenizer:
        """Get the tokenizer."""
        return self.tokenizer

    def to_device(self, device: str):
        """Move model to device."""
        self.device = device
        self.to(device)
        return self
