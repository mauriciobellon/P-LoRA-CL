"""Experiment tracker for logging and saving results."""

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""

    model_name: str
    num_tasks: int
    task_sequence: List[str]
    seed: int
    batch_size: int
    learning_rate: float
    epochs: int
    lora_r: int
    lora_alpha: int
    lambda_ortho: float
    lambda_ewc: float
    replay_ratio: float
    use_ewc: bool
    use_orthogonal: bool
    use_replay: bool
    use_lateral: bool


class ExperimentTracker:
    """Track experiment progress and save results."""

    def __init__(self, experiment_dir: str, experiment_name: str):
        """
        Initialize experiment tracker.

        Args:
            experiment_dir: Directory to save experiment data
            experiment_name: Name of the experiment
        """
        self.experiment_dir = Path(experiment_dir)
        self.experiment_name = experiment_name
        self.run_dir = self.experiment_dir / experiment_name

        # Create directories
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)
        (self.run_dir / "logs").mkdir(exist_ok=True)
        (self.run_dir / "results").mkdir(exist_ok=True)

        # Store metrics
        self.metrics_history: List[Dict[str, Any]] = []
        self.config: Optional[ExperimentConfig] = None

    def save_config(self, config: ExperimentConfig):
        """Save experiment configuration."""
        self.config = config
        config_path = self.run_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(asdict(config), f, indent=2)

    def log_metrics(
        self,
        step: int,
        task_idx: int,
        task_name: str,
        metrics: Dict[str, float],
    ):
        """
        Log metrics at a step.

        Args:
            step: Training step
            task_idx: Index of current task
            task_name: Name of current task
            metrics: Dictionary of metrics to log
        """
        log_entry = {
            "step": step,
            "task_idx": task_idx,
            "task_name": task_name,
            "timestamp": datetime.now().isoformat(),
            **metrics,
        }
        self.metrics_history.append(log_entry)

        # Save incremental log
        log_path = self.run_dir / "logs" / f"metrics_{task_idx}.json"
        with open(log_path, "w") as f:
            json.dump(self.metrics_history, f, indent=2)

    def save_checkpoint(
        self,
        model_state: Dict[str, Any],
        task_idx: int,
        metrics: Optional[Dict[str, float]] = None,
    ):
        """
        Save model checkpoint.

        Args:
            model_state: Model state dictionary
            task_idx: Index of current task
            metrics: Optional metrics to save with checkpoint
        """
        checkpoint = {
            "model_state": model_state,
            "task_idx": task_idx,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }
        checkpoint_path = self.run_dir / "checkpoints" / f"task_{task_idx}.pt"
        torch.save(checkpoint, checkpoint_path)

    def save_results(
        self,
        accuracy_matrix: np.ndarray,
        f1_matrix: np.ndarray,
        summary_metrics: Dict[str, float],
        task_metrics: Dict[int, Dict[str, float]],
    ):
        """
        Save final results.

        Args:
            accuracy_matrix: Accuracy matrix R
            f1_matrix: F1 matrix
            summary_metrics: Summary metrics (ACC, BWT, FWT, Forgetting)
            task_metrics: Metrics per task
        """
        results = {
            "accuracy_matrix": accuracy_matrix.tolist(),
            "f1_matrix": f1_matrix.tolist(),
            "summary_metrics": summary_metrics,
            "task_metrics": {str(k): v for k, v in task_metrics.items()},
            "timestamp": datetime.now().isoformat(),
        }

        results_path = self.run_dir / "results" / "final_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        # Also save as numpy arrays
        np.save(self.run_dir / "results" / "accuracy_matrix.npy", accuracy_matrix)
        np.save(self.run_dir / "results" / "f1_matrix.npy", f1_matrix)

    def log_computational_cost(
        self,
        task_idx: int,
        task_name: str,
        training_time: float,
        vram_peak: Optional[float] = None,
        num_parameters: Optional[int] = None,
        tokens_processed: Optional[int] = None,
    ):
        """
        Log computational costs.

        Args:
            task_idx: Index of task
            task_name: Name of task
            training_time: Training time in seconds
            vram_peak: Peak VRAM usage in GB
            num_parameters: Number of parameters
            tokens_processed: Number of tokens processed
        """
        cost_entry = {
            "task_idx": task_idx,
            "task_name": task_name,
            "training_time_seconds": training_time,
            "vram_peak_gb": vram_peak,
            "num_parameters": num_parameters,
            "tokens_processed": tokens_processed,
        }

        cost_path = self.run_dir / "results" / "computational_costs.json"
        costs = []
        if cost_path.exists():
            with open(cost_path, "r") as f:
                costs = json.load(f)

        costs.append(cost_entry)

        with open(cost_path, "w") as f:
            json.dump(costs, f, indent=2)


