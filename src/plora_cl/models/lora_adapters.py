"""LoRA adapter implementation - simplified version."""

from typing import Dict, List, Optional

import torch
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import AutoModel, PreTrainedModel


def get_target_modules_for_model(model: PreTrainedModel) -> List[str]:
    """
    Automatically detect target modules for LoRA based on model architecture.

    Args:
        model: The model to inspect

    Returns:
        List of target module names
    """
    # Check for BERT-style models (bert, distilbert)
    sample_modules = [name for name, _ in model.named_modules()][:50]
    if any("q_lin" in name for name in sample_modules):
        return ["q_lin", "k_lin", "v_lin", "out_lin"]
    elif any("query" in name.lower() for name in sample_modules):
        return ["query", "key", "value", "dense"]
    else:
        return ["q_proj", "k_proj", "v_proj", "o_proj"]


class LoRAAdapterManager:
    """
    Manager for LoRA adapters per task.

    Creates separate PEFT models for each task to avoid conflicts.
    """

    def __init__(
        self,
        base_model: PreTrainedModel,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
    ):
        """
        Initialize the LoRA adapter manager.

        Args:
            base_model: Base model to attach adapters to
            r: Rank of LoRA matrices
            lora_alpha: LoRA alpha scaling parameter
            lora_dropout: LoRA dropout rate
            target_modules: List of module names to target (default: auto-detect)
        """
        self.original_base_model = base_model
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

        # Auto-detect target modules if not provided
        if target_modules is None:
            self.target_modules = get_target_modules_for_model(base_model)
        else:
            self.target_modules = target_modules

        # Store adapter configs and PEFT models per task
        self.task_adapter_configs: Dict[str, LoraConfig] = {}
        self.task_peft_models: Dict[str, PeftModel] = {}
        self.current_task: Optional[str] = None

    def add_task_adapter(
        self,
        task_name: str,
        r: Optional[int] = None,
        lora_alpha: Optional[int] = None,
        lora_dropout: Optional[float] = None,
    ) -> PeftModel:
        """
        Add a LoRA adapter for a specific task.

        Args:
            task_name: Name of the task
            r: Rank (uses default if None)
            lora_alpha: Alpha (uses default if None)
            lora_dropout: Dropout (uses default if None)

        Returns:
            PEFT model with adapter
        """
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=r or self.r,
            lora_alpha=lora_alpha or self.lora_alpha,
            lora_dropout=lora_dropout or self.lora_dropout,
            target_modules=self.target_modules,
            bias="none",
        )

        # Store config
        self.task_adapter_configs[task_name] = lora_config

        # Create a fresh copy of base model for this adapter
        # Reload from config to ensure clean state
        model_name = self.original_base_model.config.name_or_path if hasattr(self.original_base_model.config, 'name_or_path') else None
        if model_name:
            # Reload base model to ensure clean state
            device = next(self.original_base_model.parameters()).device
            clean_base_model = AutoModel.from_pretrained(model_name)
            clean_base_model = clean_base_model.to(device)
            # Copy weights from original (in case it was modified)
            clean_base_model.load_state_dict(self.original_base_model.state_dict(), strict=False)
        else:
            # Fallback: use original (shouldn't happen normally)
            clean_base_model = self.original_base_model

        # Create adapter on clean base model
        peft_model = get_peft_model(clean_base_model, lora_config)

        # Store the PEFT model for this task
        self.task_peft_models[task_name] = peft_model

        # Activate this adapter
        self.activate_task(task_name)

        return peft_model

    def activate_task(self, task_name: str):
        """
        Activate adapter for a specific task.

        Args:
            task_name: Name of the task
        """
        if task_name not in self.task_adapter_configs:
            raise ValueError(f"Adapter for task {task_name} not found")

        # Simply activate the stored PEFT model
        if task_name in self.task_peft_models:
            self.current_task = task_name
        else:
            raise ValueError(f"PEFT model for task {task_name} not found")

    def freeze_previous_adapters(self, current_task: str):
        """
        Freeze adapters for all tasks except the current one.

        Args:
            current_task: Name of the current task
        """
        for task_name, peft_model in self.task_peft_models.items():
            for name, param in peft_model.named_parameters():
                if "lora" in name.lower():
                    param.requires_grad = (task_name == current_task)

    def get_adapter(self, task_name: str) -> Optional[PeftModel]:
        """Get adapter for a specific task."""
        return self.task_peft_models.get(task_name)

    def get_current_adapter(self) -> Optional[PeftModel]:
        """Get currently active adapter."""
        if self.current_task:
            return self.task_peft_models.get(self.current_task)
        return None

    def save_adapter_state(self, task_name: str):
        """Save adapter state for a task (already saved in task_peft_models)."""
        # State is already saved in task_peft_models, nothing to do
        pass
    
    def get_all_adapter_states(self) -> Dict[str, Dict]:
        """
        Get all adapter states for checkpointing.
        
        Returns:
            Dictionary mapping task names to adapter state dicts
        """
        adapter_states = {}
        for task_name, peft_model in self.task_peft_models.items():
            adapter_states[task_name] = peft_model.state_dict()
        return adapter_states
    
    def load_adapter_state(self, task_name: str, state_dict: Dict):
        """
        Load adapter state from checkpoint.
        
        Args:
            task_name: Name of the task
            state_dict: State dictionary to load
        """
        if task_name in self.task_peft_models:
            self.task_peft_models[task_name].load_state_dict(state_dict)
    
    @property
    def adapters(self) -> Dict[str, PeftModel]:
        """Get all adapters (alias for task_peft_models)."""
        return self.task_peft_models