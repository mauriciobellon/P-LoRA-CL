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
        use_lateral_connections: bool = False,
    ):
        """
        Initialize the base model.

        Args:
            model_name: Name of the base model
            device: Device to use
            freeze_base: Whether to freeze base model parameters
            use_lateral_connections: Whether to use lateral connections from previous tasks
        """
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.freeze_base = freeze_base
        self.use_lateral_connections = use_lateral_connections

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

        # Lateral connections components
        if use_lateral_connections:
            hidden_size = self.base_model.config.hidden_size
            self.lateral_fusion = nn.Linear(hidden_size * 2, hidden_size)
            self.lateral_gate = nn.Linear(hidden_size, hidden_size)
        else:
            self.lateral_fusion = None
            self.lateral_gate = None

        # Store previous task adapters for lateral connections
        self.previous_task_adapters = {}

    def add_task_head(self, task_name: str, num_classes: int):
        """
        Add a classification head for a specific task.

        Args:
            task_name: Name of the task
            num_classes: Number of classes for this task
        """
        # Get hidden size from base model
        hidden_size = self.base_model.config.hidden_size

        # Create classification head and move to device
        head = nn.Linear(hidden_size, num_classes)
        head = head.to(self.device)
        self.task_heads[task_name] = head

    def register_previous_adapter(self, task_name: str, adapter):
        """
        Register an adapter from a previous task for lateral connections.

        Args:
            task_name: Name of the previous task
            adapter: Adapter model for the previous task
        """
        if self.use_lateral_connections:
            self.previous_task_adapters[task_name] = adapter

    def get_lateral_connections_output(self, current_hidden: torch.Tensor, task_name: str) -> torch.Tensor:
        """
        Get output from lateral connections with previous task adapters.

        Args:
            current_hidden: Current task hidden representation [CLS]
            task_name: Name of the current task

        Returns:
            Fused representation combining current and previous task information
        """
        if not self.use_lateral_connections or not self.previous_task_adapters:
            return current_hidden

        # For lateral connections, we need the original input to run previous adapters
        # This is a limitation of the current implementation - we need to modify
        # the forward pass to accept the input tensors for lateral processing

        # For now, implement a simpler approach: use stored prototypes or
        # average current representation with learned lateral connections
        if hasattr(self, '_lateral_memory') and self._lateral_memory:
            # Use stored lateral memory from previous tasks
            prev_combined = torch.stack(list(self._lateral_memory.values())).mean(dim=0)

            # Fuse current and previous representations
            combined = torch.cat([current_hidden, prev_combined], dim=-1)
            fused = self.lateral_fusion(combined)

            # Apply gating mechanism
            gate = torch.sigmoid(self.lateral_gate(current_hidden))
            output = gate * fused + (1 - gate) * current_hidden

            return output

        return current_hidden

    def update_lateral_memory(self, task_name: str, hidden_repr: torch.Tensor):
        """
        Update lateral memory with representation from a completed task.

        Args:
            task_name: Name of the task
            hidden_repr: Hidden representation to store
        """
        if self.use_lateral_connections:
            if not hasattr(self, '_lateral_memory'):
                self._lateral_memory = {}

            # Store a prototype representation (mean pooling over batch)
            prototype = hidden_repr.detach().mean(dim=0)
            self._lateral_memory[task_name] = prototype

    def forward_with_lateral_input(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        task_name: str,
        current_adapter=None,
        previous_adapters=None,
    ) -> torch.Tensor:
        """
        Forward pass with full lateral connections support.

        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            task_name: Name of the task
            current_adapter: Current task adapter
            previous_adapters: Dict of previous task adapters for lateral connections

        Returns:
            Logits for the specified task
        """
        if task_name not in self.task_heads:
            raise ValueError(f"Task head for {task_name} not found")

        # Get current task representation
        if current_adapter is not None:
            current_outputs = current_adapter(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        else:
            current_outputs = self.base_model.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        current_hidden = current_outputs.last_hidden_state[:, 0]

        # Apply lateral connections if enabled and adapters provided
        if self.use_lateral_connections and previous_adapters:
            previous_representations = []

            for prev_task_name, prev_adapter in previous_adapters.items():
                with torch.no_grad():
                    prev_outputs = prev_adapter(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    prev_hidden = prev_outputs.last_hidden_state[:, 0]
                    previous_representations.append(prev_hidden)

            if previous_representations:
                # Average representations from previous tasks
                prev_combined = torch.stack(previous_representations).mean(dim=0)

                # Fuse current and previous representations
                combined = torch.cat([current_hidden, prev_combined], dim=-1)
                fused = self.lateral_fusion(combined)

                # Apply gating mechanism
                gate = torch.sigmoid(self.lateral_gate(current_hidden))
                current_hidden = gate * fused + (1 - gate) * current_hidden

        # Pass through task-specific head
        logits = self.task_heads[task_name](current_hidden)
        return logits

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        task_name: Optional[str] = None,
        current_adapter = None,
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            task_name: Name of the task (required for task-aware inference)
            current_adapter: Current task adapter (LoRA PEFT model)

        Returns:
            Logits for the specified task
        """
        if task_name is None:
            raise ValueError("task_name must be provided for task-aware inference")

        if task_name not in self.task_heads:
            raise ValueError(f"Task head for {task_name} not found")

        # Get model outputs (use adapter if available)
        if current_adapter is not None:
            outputs = current_adapter(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        else:
            outputs = self.base_model.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # Use [CLS] token
        pooled_output = outputs.last_hidden_state[:, 0]

        # Apply lateral connections if enabled
        if self.use_lateral_connections:
            pooled_output = self.get_lateral_connections_output(pooled_output, task_name)

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
