"""Main trainer for continual learning experiments."""

import time
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

    def _set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        import random
        random.seed(seed)

    def train_task(
        self,
        task_idx: int,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ):
        """
        Train on a single task.

        Args:
            task_idx: Index of the task
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
        """
        task_config = self.task_configs[task_idx]
        task_name = task_config.name

        print(f"\n{'='*60}", flush=True)
        print(f"Training on Task {task_idx + 1}: {task_name}", flush=True)
        print(f"{'='*60}", flush=True)

        # Add task head
        print(f"Adding task head for {task_name} with {task_config.num_classes} classes...", flush=True)
        self.base_model.add_task_head(task_name, task_config.num_classes)

        # Add LoRA adapter for this task
        print(f"Adding LoRA adapter for {task_name}...", flush=True)
        peft_model = self.adapter_manager.add_task_adapter(task_name)
        print(f"LoRA adapter added successfully!", flush=True)
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
        print(f"\nStarting training for {self.epochs} epochs with {len(train_loader)} batches per epoch...", flush=True)

        for epoch in range(self.epochs):
            print(f"\n--- Epoch {epoch + 1}/{self.epochs} ---", flush=True)
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, batch in enumerate(train_loader):
                if batch_idx % 10 == 0:
                    print(f"  Batch {batch_idx}/{len(train_loader)}", end='\r', flush=True)
                
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

                # Forward pass through base model
                base_outputs = peft_model.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )

                # Get logits from task head
                # Use last_hidden_state from BaseModelOutput
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

                epoch_loss += loss_dict["total_loss"].item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            print(f"\n  Epoch {epoch + 1}/{self.epochs} completed! Avg Loss: {avg_loss:.4f}", flush=True)

            # Log metrics
            self.tracker.log_metrics(
                step=epoch,
                task_idx=task_idx,
                task_name=task_name,
                metrics={"loss": avg_loss},
            )

        training_time = time.time() - start_time

        # Save adapter state
        self.adapter_manager.save_adapter_state(task_name)

        # Compute Fisher for EWC after training
        if self.use_ewc:
            print("Computing Fisher information matrix...", flush=True)
            self.ewc.compute_fisher(train_loader, task_name, str(self.device))
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

                # Forward pass through base model
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

    def train_sequence(self):
        """Train on the full sequence of tasks."""
        print("\n" + "="*80, flush=True)
        print("STARTING CONTINUAL LEARNING TRAINING", flush=True)
        print("="*80, flush=True)
        print(f"Device: {self.device}", flush=True)
        print(f"Tasks: {[t.name for t in self.task_configs]}", flush=True)
        print(f"Batch size: {self.batch_size}, Learning rate: {self.learning_rate}", flush=True)
        print(f"Epochs per task: {self.epochs}", flush=True)
        print(f"LoRA rank: {self.adapter_manager.r}, alpha: {self.adapter_manager.lora_alpha}", flush=True)
        print("="*80 + "\n", flush=True)

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
        for task_idx in range(len(self.task_configs)):
            task_config = self.task_configs[task_idx]
            task_name = task_config.name

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
            if task_idx > 0:
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
            self.train_task(task_idx, train_loader, val_loader)

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
