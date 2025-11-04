"""Baseline implementations for comparison."""

from typing import Optional

from ..data.datasets import create_dataloader, load_task_dataset
from ..training.trainer import CLTrainer


class FineTuningSequentialTrainer(CLTrainer):
    """
    Baseline: Fine-tuning sequential without any protection.

    Trains the full model sequentially on each task.
    """

    def __init__(self, **kwargs):
        """Initialize fine-tuning trainer."""
        super().__init__(
            use_ewc=False,
            use_orthogonal=False,
            use_replay=False,
            use_lateral=False,
            **kwargs,
        )
        # Unfreeze base model
        for param in self.base_model.base_model.parameters():
            param.requires_grad = True


class LoRASequentialTrainer(CLTrainer):
    """
    Baseline: Single LoRA adapter reused sequentially.

    Uses one LoRA adapter that is updated for each task.
    """

    def __init__(self, **kwargs):
        """Initialize LoRA sequential trainer."""
        super().__init__(
            use_ewc=False,
            use_orthogonal=False,
            use_replay=False,
            use_lateral=False,
            **kwargs,
        )
        # Will reuse single adapter across tasks
        self.single_adapter_task = None

    def train_task(self, task_idx, train_loader, val_loader=None):
        """Override to reuse single adapter."""
        task_config = self.task_configs[task_idx]
        task_name = task_config.name

        # Use single adapter for all tasks
        if self.single_adapter_task is None:
            self.single_adapter_task = task_name
            super().train_task(task_idx, train_loader, val_loader)
        else:
            # Reuse adapter
            self.adapter_manager.activate_task(self.single_adapter_task)
            # Don't freeze - allow updates
            self.adapter_manager.freeze_previous_adapters("")  # Empty to unfreeze all
            super().train_task(task_idx, train_loader, val_loader)


class JointTrainingTrainer(CLTrainer):
    """
    Upper bound: Joint training on all tasks simultaneously.

    This is not realistic for continual learning but provides an upper bound.
    """

    def __init__(self, **kwargs):
        """Initialize joint training trainer."""
        super().__init__(
            use_ewc=False,
            use_orthogonal=False,
            use_replay=False,
            use_lateral=False,
            **kwargs,
        )

    def train_sequence(self):
        """Override to train on all tasks simultaneously."""
        # Load all datasets
        all_train_loaders = []
        all_task_names = []

        for task_idx in range(len(self.task_configs)):
            task_config = self.task_configs[task_idx]
            task_name = task_config.name

            # Load dataset
            train_dataset = load_task_dataset(task_config, split="train[:80%]")
            train_loader = create_dataloader(
                train_dataset,
                self.base_model.tokenizer,
                task_config,
                batch_size=self.batch_size,
                shuffle=True,
            )

            all_train_loaders.append(train_loader)
            all_task_names.append(task_name)

            # Add task head
            self.base_model.add_task_head(task_name, task_config.num_classes)

        # Create single adapter for joint training
        joint_task_name = "joint"
        peft_model = self.adapter_manager.add_task_adapter(joint_task_name)
        self.adapter_manager.activate_task(joint_task_name)

        # Train on mixed batches from all tasks
        # Implementation would cycle through all loaders
        # This is simplified - full implementation would need proper batching
        print("Joint training not fully implemented - use as reference only")
        return super().train_sequence()
