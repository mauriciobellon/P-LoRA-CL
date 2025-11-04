"""Evaluation metrics for continual learning."""

from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


class CLMetrics:
    """
    Metrics tracker for continual learning experiments.

    Tracks ACC, BWT, FWT, Forgetting as per Lopez-Paz & Ranzato (2017).
    """

    def __init__(self, num_tasks: int):
        """
        Initialize metrics tracker.

        Args:
            num_tasks: Number of tasks in the sequence
        """
        self.num_tasks = num_tasks
        # R[i, j] = accuracy on task j after training on task i
        self.accuracy_matrix = np.zeros((num_tasks, num_tasks))
        self.f1_matrix = np.zeros((num_tasks, num_tasks))

    def update(
        self,
        task_idx: int,
        task_name: str,
        predictions: np.ndarray,
        labels: np.ndarray,
    ):
        """
        Update metrics after training on a task.

        Args:
            task_idx: Index of the task just trained (0-indexed)
            task_name: Name of the task
            predictions: Model predictions
            labels: True labels
        """
        acc = accuracy_score(labels, predictions)
        f1_macro = f1_score(labels, predictions, average="macro", zero_division=0)
        f1_micro = f1_score(labels, predictions, average="micro", zero_division=0)

        self.accuracy_matrix[task_idx, task_idx] = acc
        self.f1_matrix[task_idx, task_idx] = f1_macro

    def evaluate_on_task(
        self,
        task_idx: int,
        task_name: str,
        predictions: np.ndarray,
        labels: np.ndarray,
        current_task_idx: int,
    ):
        """
        Evaluate on a task after training on current_task_idx.

        Args:
            task_idx: Index of the task being evaluated
            task_name: Name of the task
            predictions: Model predictions
            labels: True labels
            current_task_idx: Index of the task just trained
        """
        acc = accuracy_score(labels, predictions)
        f1_macro = f1_score(labels, predictions, average="macro", zero_division=0)

        self.accuracy_matrix[current_task_idx, task_idx] = acc
        self.f1_matrix[current_task_idx, task_idx] = f1_macro

    def compute_average_accuracy(self) -> float:
        """
        Compute Average Accuracy (ACC).

        Returns:
            Average accuracy across all tasks after final training
        """
        final_row = self.accuracy_matrix[-1, :]
        return float(np.mean(final_row))

    def compute_backward_transfer(self) -> float:
        """
        Compute Backward Transfer (BWT).

        Positive values indicate improvement, negative indicate forgetting.

        Returns:
            Backward transfer metric
        """
        if self.num_tasks < 2:
            return 0.0

        bwt_sum = 0.0
        for i in range(self.num_tasks - 1):
            peak_acc = self.accuracy_matrix[i, i]
            final_acc = self.accuracy_matrix[-1, i]
            bwt_sum += final_acc - peak_acc

        return float(bwt_sum / (self.num_tasks - 1))

    def compute_forward_transfer(self) -> float:
        """
        Compute Forward Transfer (FWT).

        Measures how well the model performs on future tasks before training them.

        Returns:
            Forward transfer metric
        """
        if self.num_tasks < 2:
            return 0.0

        fwt_sum = 0.0
        count = 0
        for i in range(self.num_tasks):
            for j in range(i + 1, self.num_tasks):
                # Accuracy on task j after training on task i
                acc = self.accuracy_matrix[i, j]
                # Baseline: random accuracy (1/num_classes)
                # For simplicity, we use 0.5 as baseline for binary tasks
                # This should be adjusted based on task specifics
                baseline = 0.5  # Simplified baseline
                fwt_sum += acc - baseline
                count += 1

        return float(fwt_sum / count) if count > 0 else 0.0

    def compute_forgetting(self) -> float:
        """
        Compute average forgetting.

        Returns:
            Average forgetting across all tasks
        """
        forgetting_sum = 0.0
        for i in range(self.num_tasks):
            peak_acc = self.accuracy_matrix[i, i]
            final_acc = self.accuracy_matrix[-1, i]
            forgetting_sum += peak_acc - final_acc

        return float(forgetting_sum / self.num_tasks)

    def get_accuracy_matrix(self) -> np.ndarray:
        """Get the accuracy matrix R."""
        return self.accuracy_matrix.copy()

    def get_f1_matrix(self) -> np.ndarray:
        """Get the F1 matrix."""
        return self.f1_matrix.copy()

    def get_task_metrics(self, task_idx: int) -> Dict[str, float]:
        """
        Get metrics for a specific task.

        Args:
            task_idx: Index of the task

        Returns:
            Dictionary of metrics
        """
        peak_acc = self.accuracy_matrix[task_idx, task_idx]
        final_acc = self.accuracy_matrix[-1, task_idx]
        forgetting = peak_acc - final_acc

        return {
            "peak_accuracy": float(peak_acc),
            "final_accuracy": float(final_acc),
            "forgetting": float(forgetting),
            "peak_f1": float(self.f1_matrix[task_idx, task_idx]),
            "final_f1": float(self.f1_matrix[-1, task_idx]),
        }

    def get_summary(self) -> Dict[str, float]:
        """
        Get summary of all metrics.

        Returns:
            Dictionary with all aggregated metrics
        """
        return {
            "average_accuracy": self.compute_average_accuracy(),
            "backward_transfer": self.compute_backward_transfer(),
            "forward_transfer": self.compute_forward_transfer(),
            "forgetting": self.compute_forgetting(),
        }


