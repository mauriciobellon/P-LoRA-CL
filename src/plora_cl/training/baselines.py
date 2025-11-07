"""Baseline implementations for comparison."""

from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

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

    def _create_mixed_dataloader(self, task_configs) -> DataLoader:
        """
        Create a mixed dataloader that samples from all tasks.

        Args:
            task_configs: List of task configurations

        Returns:
            DataLoader that yields mixed batches from all tasks
        """
        # Load all datasets
        all_datasets = []
        all_task_names = []

        for task_config in task_configs:
            # Load full training dataset
            train_dataset = load_task_dataset(task_config, split="train")
            all_datasets.append(train_dataset)
            all_task_names.append(task_config.name)

            # Add task head
            self.base_model.add_task_head(task_config.name, task_config.num_classes)

        # Create mixed dataset class
        class MixedDataset(torch.utils.data.Dataset):
            def __init__(self, datasets, task_names):
                self.datasets = datasets
                self.task_names = task_names
                # Calculate cumulative lengths for sampling
                self.cum_lengths = [0]
                for dataset in datasets:
                    self.cum_lengths.append(self.cum_lengths[-1] + len(dataset))

            def __len__(self):
                return sum(len(d) for d in self.datasets)

            def __getitem__(self, idx):
                # Find which dataset this index belongs to
                for i, (start, end) in enumerate(zip(self.cum_lengths[:-1], self.cum_lengths[1:])):
                    if start <= idx < end:
                        dataset_idx = i
                        local_idx = idx - start
                        break

                # Get sample from appropriate dataset
                sample = self.datasets[dataset_idx][local_idx]

                # Add task name to sample
                sample["task_name"] = self.task_names[dataset_idx]
                return sample

        # Create mixed dataset
        mixed_dataset = MixedDataset(all_datasets, all_task_names)

        # Create dataloader with shuffling
        mixed_loader = torch.utils.data.DataLoader(
            mixed_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # Avoid issues with multiprocessing
            collate_fn=self._collate_mixed_batch,
        )

        return mixed_loader

    def _collate_mixed_batch(self, batch):
        """
        Collate function for mixed batches with different task names.

        Args:
            batch: List of samples from different tasks

        Returns:
            Collated batch with task-specific labels
        """
        if not batch:
            return {}

        # Get tokenizer for padding
        tokenizer = self.base_model.tokenizer

        # Separate by task
        task_batches = {}
        for sample in batch:
            task_name = sample["task_name"]
            if task_name not in task_batches:
                task_batches[task_name] = []
            task_batches[task_name].append(sample)

        # Process each task batch
        collated = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "task_names": [],
        }

        for task_name, task_samples in task_batches.items():
            # Extract inputs
            inputs = [s["text"] for s in task_samples]
            labels = [s["label"] for s in task_samples]

            # Tokenize
            tokenized = tokenizer(
                inputs,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt",
            )

            collated["input_ids"].append(tokenized["input_ids"])
            collated["attention_mask"].append(tokenized["attention_mask"])
            collated["labels"].append(torch.tensor(labels))
            collated["task_names"].extend([task_name] * len(task_samples))

        # Concatenate all task batches
        collated["input_ids"] = torch.cat(collated["input_ids"], dim=0)
        collated["attention_mask"] = torch.cat(collated["attention_mask"], dim=0)
        collated["labels"] = torch.cat(collated["labels"], dim=0)

        return collated

    def train_sequence(self):
        """Override to train on all tasks simultaneously."""
        print("Starting joint training on all tasks...")

        # Create mixed dataloader
        mixed_loader = self._create_mixed_dataloader(self.task_configs)

        # Create single adapter for joint training
        joint_task_name = "joint"
        peft_model = self.adapter_manager.add_task_adapter(joint_task_name)
        self.adapter_manager.activate_task(joint_task_name)

        # Set up training components
        trainable_params = list(peft_model.parameters()) + list(self.base_model.task_heads.parameters())
        optimizer = torch.optim.AdamW(trainable_params, lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=len(mixed_loader) * self.epochs,
        )

        # Training loop
        peft_model.train()
        self.global_step = 0

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, batch in enumerate(mixed_loader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                task_names = batch["task_names"]

                # Forward pass through base model
                base_outputs = peft_model.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )

                pooled_output = base_outputs.last_hidden_state[:, 0]

                # Get logits for each sample's task
                logits = []
                for i, task_name in enumerate(task_names):
                    task_logits = self.base_model.task_heads[task_name](pooled_output[i:i+1])
                    logits.append(task_logits)

                logits = torch.cat(logits, dim=0)

                # Compute loss
                loss_fn = torch.nn.CrossEntropyLoss()
                loss = loss_fn(logits, labels)

                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps

                # Backward pass
                loss.backward()

                # Update weights
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, self.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    self.global_step += 1

                epoch_loss += loss.item()
                num_batches += 1

                # Progress logging
                if batch_idx % 10 == 0:
                    print(f"Joint training - Epoch {epoch+1}/{self.epochs}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")

            avg_epoch_loss = epoch_loss / num_batches
            print(f"Joint training - Epoch {epoch+1}/{self.epochs} completed, Avg Loss: {avg_epoch_loss:.4f}")

        # Evaluate on all tasks
        print("Evaluating joint training results...")
        for task_idx in range(len(self.task_configs)):
            task_config = self.task_configs[task_idx]
            test_dataset = load_task_dataset(task_config, split="test")
            test_loader = create_dataloader(
                test_dataset,
                self.base_model.tokenizer,
                task_config,
                batch_size=self.batch_size,
                shuffle=False,
            )

            self.evaluate_task(task_idx, test_loader, current_task_idx=len(self.task_configs)-1)

        return self.metrics.get_summary()
