"""Main trainer for continual learning experiments."""

import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from ..data.datasets import TASK_SEQUENCE, TaskConfig, create_dataloader, load_task_dataset
from ..evaluation.metrics import CLMetrics
from ..evaluation.tracker import ExperimentConfig, ExperimentTracker
from ..models.base_model import BaseCLModel
from ..models.ewc import EWC
from ..models.orthogonal_lora import OrthogonalLoRA
from ..training.loss import CompositeLoss
from ..training.replay import PseudoReplayGenerator


class CLTrainer:
    """
    Trainer for continual learning experiments.

    Implements sequential training with O-LoRA, EWC, and generative replay.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        device: str = "auto",
        experiment_dir: str = "experiments",
        experiment_name: str = "default",
        seed: int = 42,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        epochs: int = 3,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lambda_ortho: float = 0.1,
        lambda_ewc: float = 100.0,
        replay_ratio: float = 0.2,
        use_ewc: bool = True,
        use_orthogonal: bool = True,
        use_replay: bool = True,
        use_lateral: bool = False,
        gradient_accumulation_steps: int = 1,
        warmup_ratio: float = 0.1,
        max_grad_norm: float = 1.0,
        checkpoint_every: int = 100,
        keep_last_n_checkpoints: int = 3,
    ):
        """
        Initialize CL trainer.

        Args:
            model_name: Name of base model
            device: Device to use ('auto', 'cpu', 'cuda')
            experiment_dir: Directory for experiments
            experiment_name: Name of this experiment
            seed: Random seed
            batch_size: Batch size
            learning_rate: Learning rate
            epochs: Number of epochs per task
            lora_r: LoRA rank
            lora_alpha: LoRA alpha
            lambda_ortho: Orthogonal regularization weight
            lambda_ewc: EWC regularization weight
            replay_ratio: Ratio of replay samples in batch
            use_ewc: Whether to use EWC
            use_orthogonal: Whether to use orthogonal LoRA
            use_replay: Whether to use generative replay
            use_lateral: Whether to use lateral connections
            gradient_accumulation_steps: Gradient accumulation steps
            warmup_ratio: Warmup ratio for LR scheduler
            max_grad_norm: Maximum gradient norm for clipping
            checkpoint_every: Save checkpoint every N steps (0 to disable)
            keep_last_n_checkpoints: Keep only last N checkpoints to save disk space
        """
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Set seed
        self._set_seed(seed)

        # Initialize base model
        self.base_model = BaseCLModel(
            model_name=model_name,
            device=str(self.device),
            freeze_base=True,
            use_lateral_connections=use_lateral,
        ).to_device(str(self.device))

        # Initialize LoRA adapter manager
        if use_orthogonal:
            self.adapter_manager = OrthogonalLoRA(
                base_model=self.base_model.base_model,
                r=lora_r,
                lora_alpha=lora_alpha,
                lambda_ortho=lambda_ortho,
            )
        else:
            from ..models.lora_adapters import LoRAAdapterManager
            self.adapter_manager = LoRAAdapterManager(
                base_model=self.base_model.base_model,
                r=lora_r,
                lora_alpha=lora_alpha,
            )

        # Initialize EWC
        self.ewc = EWC(
            model=self.base_model.base_model,
            lambda_ewc=lambda_ewc,
            online=True,
        ) if use_ewc else None

        # Initialize replay generator
        self.replay_generator = PseudoReplayGenerator(
            task_configs=TASK_SEQUENCE,
            tokenizer=self.base_model.tokenizer,
            device=str(self.device),
        ) if use_replay else None

        # Training config
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.replay_ratio = replay_ratio
        self.lambda_ortho = lambda_ortho
        self.lambda_ewc = lambda_ewc
        self.use_ewc = use_ewc
        self.use_orthogonal = use_orthogonal
        self.use_replay = use_replay
        self.use_lateral = use_lateral
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_ratio = warmup_ratio
        self.max_grad_norm = max_grad_norm

        # Initialize metrics and tracker
        self.metrics = CLMetrics(num_tasks=len(TASK_SEQUENCE))
        self.tracker = ExperimentTracker(experiment_dir, experiment_name)

        # Store task configs
        self.task_configs = TASK_SEQUENCE
        self.task_names = [task.name for task in TASK_SEQUENCE]

        # Training state
        self.current_task_idx = 0
        self.trained_tasks: List[str] = []
        self.seed = seed

        # Checkpointing
        self.checkpoint_every = checkpoint_every
        self.keep_last_n_checkpoints = keep_last_n_checkpoints
        self.checkpoint_dir = Path(experiment_dir) / experiment_name / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.global_step = 0
        self.start_epoch = 0
        self.start_batch = 0

    def _set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        import random
        random.seed(seed)

    def save_checkpoint(
        self,
        task_idx: int,
        epoch: int,
        batch_idx: int,
        optimizer,
        scheduler,
        loss: float,
    ):
        """
        Save a training checkpoint.

        Args:
            task_idx: Current task index
            epoch: Current epoch
            batch_idx: Current batch index
            optimizer: Optimizer state
            scheduler: Scheduler state
            loss: Current loss value
        """
        checkpoint_path = self.checkpoint_dir / f"checkpoint_task{task_idx}_epoch{epoch}_batch{batch_idx}.pt"

        checkpoint = {
            "global_step": self.global_step,
            "task_idx": task_idx,
            "epoch": epoch,
            "batch_idx": batch_idx,
            "trained_tasks": self.trained_tasks,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": loss,
            "seed": self.seed,
            "metrics_state": self.metrics.get_state(),
            "current_task_idx": self.current_task_idx,  # Add for better resume logic
        }

        # Save base model state
        checkpoint["base_model_state_dict"] = self.base_model.base_model.state_dict()

        # Save task heads
        checkpoint["task_heads"] = {
            name: head.state_dict()
            for name, head in self.base_model.task_heads.items()
        }

        # Save adapter states
        checkpoint["adapter_states"] = self.adapter_manager.get_all_adapter_states()
        checkpoint["adapter_config"] = {
            "r": self.adapter_manager.r,
            "lora_alpha": self.adapter_manager.lora_alpha,
        }

        # Save EWC state if enabled
        if self.ewc is not None:
            checkpoint["ewc_state"] = self.ewc.get_state()

        # Save replay generator state if enabled
        if self.replay_generator is not None:
            checkpoint["replay_state"] = self.replay_generator.get_state()

        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}", flush=True)

        # Clean up old checkpoints for this task only
        self._cleanup_old_checkpoints(task_idx)

        return checkpoint_path

    def _cleanup_old_checkpoints(self, current_task_idx: int):
        """
        Remove old checkpoints for the current task only, keeping last N per task.
        This preserves checkpoints from completed tasks while managing disk space.
        """
        if self.keep_last_n_checkpoints <= 0:
            return

        # Only clean up checkpoints for the current task
        task_checkpoints = sorted(
            self.checkpoint_dir.glob(f"checkpoint_task{current_task_idx}_*.pt"),
            key=lambda x: x.stat().st_mtime
        )

        if len(task_checkpoints) > self.keep_last_n_checkpoints:
            for old_checkpoint in task_checkpoints[:-self.keep_last_n_checkpoints]:
                old_checkpoint.unlink()
                print(f"Removed old checkpoint: {old_checkpoint.name}", flush=True)

    def _load_checkpoint_metadata(self, checkpoint_path: str):
        """
        Load only metadata from checkpoint (task index, trained tasks, etc).
        Used to determine where to resume without loading full model state.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        print(f"Loading checkpoint metadata from {checkpoint_path}...", flush=True)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Restore only training state metadata
        self.global_step = checkpoint["global_step"]
        self.current_task_idx = checkpoint["task_idx"]
        self.start_epoch = checkpoint["epoch"]
        self.start_batch = checkpoint["batch_idx"] + 1
        self.trained_tasks = checkpoint["trained_tasks"]

        print(f"Checkpoint metadata: task={self.current_task_idx}, epoch={self.start_epoch}, batch={self.start_batch}", flush=True)
        print(f"Trained tasks: {self.trained_tasks}", flush=True)

    def _load_checkpoint_models(self, checkpoint_path: str):
        """
        Load model states from checkpoint (base model, adapters, task heads, etc).
        This is called early in train_sequence to restore models before evaluation.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        print(f"Loading model states from checkpoint...", flush=True)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Restore base model
        self.base_model.base_model.load_state_dict(checkpoint["base_model_state_dict"])

        # Restore task heads
        for name, state_dict in checkpoint["task_heads"].items():
            if name not in self.base_model.task_heads:
                task_idx = self.task_names.index(name)
                task_config = self.task_configs[task_idx]
                self.base_model.add_task_head(name, task_config.num_classes)
            self.base_model.task_heads[name].load_state_dict(state_dict)

        # Restore adapters
        for task_name, adapter_state in checkpoint["adapter_states"].items():
            if task_name not in self.adapter_manager.adapters:
                self.adapter_manager.add_task_adapter(task_name)
            self.adapter_manager.load_adapter_state(task_name, adapter_state)

        # Restore EWC state
        if self.ewc is not None and "ewc_state" in checkpoint:
            self.ewc.load_state(checkpoint["ewc_state"])

        # Restore replay generator state
        if self.replay_generator is not None and "replay_state" in checkpoint:
            self.replay_generator.load_state(checkpoint["replay_state"])

        # Restore metrics
        self.metrics.load_state(checkpoint["metrics_state"])

        print(f"Model states loaded successfully!", flush=True)

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load optimizer and scheduler states from checkpoint.
        Model states should already be loaded via _load_checkpoint_models.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Tuple of (optimizer_state_dict, scheduler_state_dict)
        """
        print(f"Loading optimizer/scheduler states from {checkpoint_path}...", flush=True)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        print(f"Optimizer/scheduler states loaded!", flush=True)

        return checkpoint["optimizer_state_dict"], checkpoint["scheduler_state_dict"]

    def get_latest_checkpoint(self) -> Optional[Path]:
        """
        Find the latest checkpoint file.

        Returns:
            Path to latest checkpoint or None if no checkpoints exist
        """
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.pt"), key=lambda x: x.stat().st_mtime)
        return checkpoints[-1] if checkpoints else None

    def train_task(
        self,
        task_idx: int,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        resume_from_checkpoint: bool = False,
    ):
        """
        Train on a single task.

        Args:
            task_idx: Index of the task
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            resume_from_checkpoint: Whether to resume from checkpoint
        """
        task_config = self.task_configs[task_idx]
        task_name = task_config.name

        print(f"\n{'='*60}", flush=True)
        print(f"Training on Task {task_idx + 1}: {task_name}", flush=True)
        print(f"{'='*60}", flush=True)

        # Add task head if not resuming or if it doesn't exist
        if task_name not in self.base_model.task_heads:
            print(f"Adding task head for {task_name} with {task_config.num_classes} classes...", flush=True)
            self.base_model.add_task_head(task_name, task_config.num_classes)

        # Add LoRA adapter for this task if not resuming or if it doesn't exist
        if task_name not in self.adapter_manager.adapters:
            print(f"Adding LoRA adapter for {task_name}...", flush=True)
            peft_model = self.adapter_manager.add_task_adapter(task_name)
            print(f"LoRA adapter added successfully!", flush=True)
        else:
            peft_model = self.adapter_manager.adapters[task_name]

        self.adapter_manager.activate_task(task_name)

        # Freeze previous adapters
        print(f"Freezing previous adapters...", flush=True)
        self.adapter_manager.freeze_previous_adapters(task_name)

        # Get parameters to optimize (only LoRA adapters and task head)
        print(f"Collecting trainable parameters...", flush=True)
        trainable_params = []
        for name, param in peft_model.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)

        # Add task head parameters
        if task_name in self.base_model.task_heads:
            trainable_params.extend(list(self.base_model.task_heads[task_name].parameters()))

        num_trainable = sum(p.numel() for p in trainable_params)
        print(f"Total trainable parameters: {num_trainable:,}", flush=True)

        # Initialize optimizer
        print(f"Initializing optimizer (lr={self.learning_rate})...", flush=True)
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=0.01,
        )

        # Calculate total steps
        num_training_steps = len(train_loader) * self.epochs // self.gradient_accumulation_steps
        num_warmup_steps = int(num_training_steps * self.warmup_ratio)
        print(f"Training steps: {num_training_steps}, Warmup steps: {num_warmup_steps}", flush=True)

        # Initialize scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        # Load optimizer/scheduler states if resuming
        # Note: Model states already loaded in train_sequence
        if resume_from_checkpoint:
            latest_checkpoint = self.get_latest_checkpoint()
            optimizer_state, scheduler_state = self.load_checkpoint(str(latest_checkpoint))
            optimizer.load_state_dict(optimizer_state)
            scheduler.load_state_dict(scheduler_state)

            # Check if we need to skip this task entirely (already completed)
            # If we're on the last epoch and the start batch is at/past the end, task is done
            if self.start_epoch >= self.epochs - 1 and self.start_batch >= len(train_loader):
                print(f"Task {task_idx} ({task_name}) already completed, skipping...", flush=True)
                # Make sure it's in trained_tasks
                if task_name not in self.trained_tasks:
                    self.trained_tasks.append(task_name)
                return
            elif self.start_epoch >= self.epochs:
                print(f"Task {task_idx} ({task_name}) already completed (epochs), skipping...", flush=True)
                if task_name not in self.trained_tasks:
                    self.trained_tasks.append(task_name)
                return

        # Initialize loss function
        loss_fn = CompositeLoss(
            lambda_ortho=self.lambda_ortho if self.use_orthogonal else 0.0,
            lambda_ewc=self.lambda_ewc if self.use_ewc else 0.0,
            use_orthogonal=self.use_orthogonal,
            use_ewc=self.use_ewc,
        )

        # Training loop
        peft_model.train()
        start_time = time.time()

        # Determine starting point
        start_epoch = self.start_epoch if resume_from_checkpoint else 0
        print(f"\nStarting training for {self.epochs} epochs with {len(train_loader)} batches per epoch...", flush=True)
        if resume_from_checkpoint:
            print(f"Resuming from epoch {start_epoch + 1}, batch {self.start_batch}", flush=True)

        for epoch in range(start_epoch, self.epochs):
            print(f"\n--- Epoch {epoch + 1}/{self.epochs} ---", flush=True)
            epoch_loss = 0.0
            num_batches = 0

            # Determine starting batch
            start_batch = self.start_batch if (resume_from_checkpoint and epoch == start_epoch) else 0

            # If starting batch is at or beyond the end of the dataloader, skip this epoch
            if start_batch >= len(train_loader):
                print(f"  Epoch {epoch + 1} already completed, skipping...", flush=True)
                continue

            for batch_idx, batch in enumerate(train_loader):
                # Skip batches if resuming from checkpoint
                if batch_idx < start_batch:
                    continue

                if batch_idx % 10 == 0:
                    print(f"  Batch {batch_idx}/{len(train_loader)} (global step: {self.global_step})", end='\r', flush=True)

                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Prepare replay samples if enabled
                replay_samples = []
                if self.use_replay and self.trained_tasks:
                    num_replay = int(self.batch_size * self.replay_ratio)
                    for prev_task in self.trained_tasks:
                        prev_task_idx = self.task_names.index(prev_task)
                        prev_config = self.task_configs[prev_task_idx]
                        replay_batch = self.replay_generator.generate_batch_replay(
                            prev_task,
                            prev_config.num_classes,
                            self.batch_size,
                            self.replay_ratio / len(self.trained_tasks),
                        )
                        replay_samples.extend(replay_batch)

                # Forward pass - use lateral connections if enabled
                if self.use_lateral and self.trained_tasks:
                    # Get previous task adapters for lateral connections
                    previous_adapters = {}
                    for prev_task in self.trained_tasks:
                        prev_adapter = self.adapter_manager.get_adapter(prev_task)
                        if prev_adapter is not None:
                            previous_adapters[prev_task] = prev_adapter

                    # Use forward with lateral input
                    logits = self.base_model.forward_with_lateral_input(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        task_name=task_name,
                        current_adapter=peft_model,
                        previous_adapters=previous_adapters,
                    )
                else:
                    # Standard forward pass
                    base_outputs = peft_model.base_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                    )
                    pooled_output = base_outputs.last_hidden_state[:, 0]
                    logits = self.base_model.task_heads[task_name](pooled_output)

                # Compute orthogonal loss
                orthogonal_loss = None
                if self.use_orthogonal and self.trained_tasks:
                    orthogonal_loss = self.adapter_manager.compute_orthogonal_loss(
                        task_name,
                        self.trained_tasks,
                    )

                # Compute EWC loss
                ewc_loss = None
                if self.use_ewc:
                    ewc_loss = self.ewc.compute_ewc_loss()

                # Compute loss
                loss_dict = loss_fn(logits, labels, orthogonal_loss, ewc_loss)
                loss = loss_dict["total_loss"]

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

                    # Save checkpoint periodically
                    if self.checkpoint_every > 0 and self.global_step % self.checkpoint_every == 0:
                        self.save_checkpoint(
                            task_idx=task_idx,
                            epoch=epoch,
                            batch_idx=batch_idx,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            loss=loss_dict["total_loss"].item(),
                        )

                epoch_loss += loss_dict["total_loss"].item()
                num_batches += 1

            # Only log and save if we actually processed batches
            if num_batches > 0:
                avg_loss = epoch_loss / num_batches
                print(f"\n  Epoch {epoch + 1}/{self.epochs} completed! Avg Loss: {avg_loss:.4f}", flush=True)

                # Log metrics
                self.tracker.log_metrics(
                    step=epoch,
                    task_idx=task_idx,
                    task_name=task_name,
                    metrics={"loss": avg_loss},
                )

                # Save checkpoint at end of epoch
                if self.checkpoint_every > 0:
                    self.save_checkpoint(
                        task_idx=task_idx,
                        epoch=epoch,
                        batch_idx=len(train_loader) - 1,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        loss=avg_loss,
                    )

        training_time = time.time() - start_time

        # Reset start epoch and batch for next task
        self.start_epoch = 0
        self.start_batch = 0

        # Save adapter state
        self.adapter_manager.save_adapter_state(task_name)

        # Compute Fisher for EWC after training
        if self.use_ewc:
            print("Computing Fisher information matrix...", flush=True)
            task_head = self.base_model.task_heads[task_name]
            self.ewc.compute_fisher(train_loader, task_name, str(self.device), task_head)
            print("Fisher computation complete!", flush=True)

        # Mark task as trained
        self.trained_tasks.append(task_name)

        # Log computational cost
        if torch.cuda.is_available():
            vram_peak = torch.cuda.max_memory_allocated() / 1e9  # GB
            torch.cuda.reset_peak_memory_stats()
        else:
            vram_peak = None

        self.tracker.log_computational_cost(
            task_idx=task_idx,
            task_name=task_name,
            training_time=training_time,
            vram_peak=vram_peak,
        )

        print(f"\n✓ Training completed in {training_time:.2f} seconds", flush=True)

    def evaluate_task(
        self,
        task_idx: int,
        test_loader: DataLoader,
        current_task_idx: int,
    ) -> Dict[str, float]:
        """
        Evaluate on a task.

        Args:
            task_idx: Index of task to evaluate
            test_loader: Test data loader
            current_task_idx: Index of task just trained

        Returns:
            Dictionary of metrics
        """
        task_config = self.task_configs[task_idx]
        task_name = task_config.name

        # Activate appropriate adapter
        self.adapter_manager.activate_task(task_name)
        peft_model = self.adapter_manager.get_current_adapter()
        peft_model.eval()

        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass - use lateral connections if enabled
                if self.use_lateral and current_task_idx > task_idx:
                    # During evaluation of previous tasks, use lateral connections if enabled
                    # Get previous task adapters (excluding current task being evaluated)
                    previous_adapters = {}
                    for prev_task_idx in range(task_idx):
                        prev_task_name = self.task_configs[prev_task_idx].name
                        prev_adapter = self.adapter_manager.get_adapter(prev_task_name)
                        if prev_adapter is not None:
                            previous_adapters[prev_task_name] = prev_adapter

                    if previous_adapters:
                        # Use forward with lateral input
                        logits = self.base_model.forward_with_lateral_input(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            task_name=task_name,
                            current_adapter=peft_model,
                            previous_adapters=previous_adapters,
                        )
                    else:
                        # No previous adapters available, use standard forward
                        base_outputs = peft_model.base_model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            output_hidden_states=True,
                        )
                        pooled_output = base_outputs.last_hidden_state[:, 0]
                        logits = self.base_model.task_heads[task_name](pooled_output)
                else:
                    # Standard forward pass
                    base_outputs = peft_model.base_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                    )
                    pooled_output = base_outputs.last_hidden_state[:, 0]
                    logits = self.base_model.task_heads[task_name](pooled_output)

                predictions = torch.argmax(logits, dim=-1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Update metrics
        self.metrics.evaluate_on_task(
            task_idx=task_idx,
            task_name=task_name,
            predictions=np.array(all_predictions),
            labels=np.array(all_labels),
            current_task_idx=current_task_idx,
        )

        # Compute accuracy
        accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))

        return {"accuracy": float(accuracy)}

    def train_sequence(self, resume: bool = False):
        """
        Train on the full sequence of tasks.

        Args:
            resume: Whether to resume from latest checkpoint
        """
        print("\n" + "="*80, flush=True)
        print("STARTING CONTINUAL LEARNING TRAINING", flush=True)
        print("="*80, flush=True)
        print(f"Device: {self.device}", flush=True)
        print(f"Tasks: {[t.name for t in self.task_configs]}", flush=True)
        print(f"Batch size: {self.batch_size}, Learning rate: {self.learning_rate}", flush=True)
        print(f"Epochs per task: {self.epochs}", flush=True)
        print(f"LoRA rank: {self.adapter_manager.r}, alpha: {self.adapter_manager.lora_alpha}", flush=True)
        print("="*80 + "\n", flush=True)

        # Check for existing checkpoint if resume requested
        resume_from_checkpoint = False
        latest_checkpoint_path = None
        if resume:
            latest_checkpoint = self.get_latest_checkpoint()
            if latest_checkpoint:
                print(f"Found checkpoint: {latest_checkpoint}", flush=True)
                latest_checkpoint_path = str(latest_checkpoint)
                # Load checkpoint metadata early to get correct task index
                self._load_checkpoint_metadata(latest_checkpoint_path)
                # Load full checkpoint state (models, adapters) before starting
                self._load_checkpoint_models(latest_checkpoint_path)
                resume_from_checkpoint = True
            else:
                print("No checkpoint found, starting from scratch", flush=True)

        # Save config
        config = ExperimentConfig(
            model_name=self.base_model.model_name,
            num_tasks=len(self.task_configs),
            task_sequence=self.task_names,
            seed=self.seed,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            epochs=self.epochs,
            lora_r=self.adapter_manager.r,
            lora_alpha=self.adapter_manager.lora_alpha,
            lambda_ortho=self.lambda_ortho if self.use_orthogonal else 0.0,
            lambda_ewc=self.lambda_ewc if self.use_ewc else 0.0,
            replay_ratio=self.replay_ratio,
            use_ewc=self.use_ewc,
            use_orthogonal=self.use_orthogonal,
            use_replay=self.use_replay,
            use_lateral=self.use_lateral,
        )
        self.tracker.save_config(config)

        # Train each task sequentially
        start_task = self.current_task_idx if resume_from_checkpoint else 0

        # If resuming, check if current task is completed and move to next
        if resume_from_checkpoint and start_task < len(self.task_configs):
            current_task_name = self.task_configs[start_task].name
            if current_task_name in self.trained_tasks:
                print(f"Task {start_task} ({current_task_name}) already in trained_tasks, moving to next", flush=True)
                start_task += 1

        print(f"Starting/resuming from task {start_task}", flush=True)

        for task_idx in range(start_task, len(self.task_configs)):
            task_config = self.task_configs[task_idx]
            task_name = task_config.name

            # Check if we should resume for this specific task
            should_resume = (resume_from_checkpoint and task_idx == start_task and task_idx == self.current_task_idx)

            # Load datasets
            train_dataset = load_task_dataset(task_config, split="train[:80%]")
            val_dataset = load_task_dataset(task_config, split="train[80%:]")
            test_dataset = load_task_dataset(task_config, split="test")

            # Create data loaders
            train_loader = create_dataloader(
                train_dataset,
                self.base_model.tokenizer,
                task_config,
                batch_size=self.batch_size,
                shuffle=True,
            )
            val_loader = create_dataloader(
                val_dataset,
                self.base_model.tokenizer,
                task_config,
                batch_size=self.batch_size,
                shuffle=False,
            )
            test_loader = create_dataloader(
                test_dataset,
                self.base_model.tokenizer,
                task_config,
                batch_size=self.batch_size,
                shuffle=False,
            )

            # Evaluate on previous tasks before training
            # Skip if resuming and this task is just starting (evaluations already done)
            if task_idx > 0 and not (should_resume and self.start_epoch == 0 and self.start_batch == 0):
                print("\nEvaluating on previous tasks...")
                for prev_idx in range(task_idx):
                    prev_config = self.task_configs[prev_idx]
                    prev_test_dataset = load_task_dataset(prev_config, split="test")
                    prev_test_loader = create_dataloader(
                        prev_test_dataset,
                        self.base_model.tokenizer,
                        prev_config,
                        batch_size=self.batch_size,
                        shuffle=False,
                    )
                    metrics = self.evaluate_task(prev_idx, prev_test_loader, task_idx - 1)
                    print(f"  {prev_config.name}: {metrics['accuracy']:.4f}")

            # Train on current task
            self.train_task(task_idx, train_loader, val_loader, resume_from_checkpoint=should_resume)

            # Evaluate on all tasks seen so far
            print("\nEvaluating on all tasks...")
            for eval_idx in range(task_idx + 1):
                eval_config = self.task_configs[eval_idx]
                eval_test_dataset = load_task_dataset(eval_config, split="test")
                eval_test_loader = create_dataloader(
                    eval_test_dataset,
                    self.base_model.tokenizer,
                    eval_config,
                    batch_size=self.batch_size,
                    shuffle=False,
                )
                metrics = self.evaluate_task(eval_idx, eval_test_loader, task_idx)
                print(f"  {eval_config.name}: {metrics['accuracy']:.4f}")

        # Save final results
        summary_metrics = self.metrics.get_summary()
        task_metrics = {
            i: self.metrics.get_task_metrics(i)
            for i in range(len(self.task_configs))
        }

        self.tracker.save_results(
            accuracy_matrix=self.metrics.get_accuracy_matrix(),
            f1_matrix=self.metrics.get_f1_matrix(),
            summary_metrics=summary_metrics,
            task_metrics=task_metrics,
        )

        print("\n" + "="*60)
        print("Final Results:")
        print("="*60)
        print(f"Average Accuracy: {summary_metrics['average_accuracy']:.4f}")
        print(f"Backward Transfer: {summary_metrics['backward_transfer']:.4f}")
        print(f"Forward Transfer: {summary_metrics['forward_transfer']:.4f}")
        print(f"Forgetting: {summary_metrics['forgetting']:.4f}")

        return summary_metrics
