"""Visualization scripts for generating plots and tables."""

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")


def load_results(experiment_dir: str, experiment_name: str) -> Dict:
    """
    Load experiment results.

    Args:
        experiment_dir: Experiment directory
        experiment_name: Experiment name

    Returns:
        Dictionary with results
    """
    results_path = Path(experiment_dir) / experiment_name / "results" / "final_results.json"
    with open(results_path, "r") as f:
        return json.load(f)


def plot_accuracy_evolution(
    results: Dict,
    output_path: str,
    task_names: Optional[List[str]] = None,
):
    """
    Plot accuracy evolution across tasks.

    Args:
        results: Results dictionary
        output_path: Path to save plot
        task_names: List of task names
    """
    accuracy_matrix = np.array(results["accuracy_matrix"])
    num_tasks = accuracy_matrix.shape[0]

    if task_names is None:
        task_names = [f"Task {i+1}" for i in range(num_tasks)]

    plt.figure(figsize=(10, 6))

    for task_idx in range(num_tasks):
        accuracies = accuracy_matrix[:, task_idx]
        plt.plot(
            range(1, num_tasks + 1),
            accuracies,
            marker="o",
            label=task_names[task_idx],
            linewidth=2,
        )

    plt.xlabel("Training Task", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Accuracy Evolution Across Tasks", fontsize=14, fontweight="bold")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_metrics_comparison(
    results_list: List[Dict],
    method_names: List[str],
    output_path: str,
):
    """
    Plot comparison of metrics across methods.

    Args:
        results_list: List of results dictionaries
        method_names: List of method names
        output_path: Path to save plot
    """
    metrics = ["average_accuracy", "backward_transfer", "forward_transfer", "forgetting"]
    metric_labels = ["ACC", "BWT", "FWT", "Forgetting"]

    data = []
    for method_name, results in zip(method_names, results_list):
        summary = results["summary_metrics"]
        for metric, label in zip(metrics, metric_labels):
            data.append({
                "Method": method_name,
                "Metric": label,
                "Value": summary[metric],
            })

    df = pd.DataFrame(data)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        metric_data = df[df["Metric"] == label]
        axes[idx].bar(metric_data["Method"], metric_data["Value"])
        axes[idx].set_title(label, fontweight="bold")
        axes[idx].set_ylabel("Value")
        axes[idx].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_ablation(
    results_list: List[Dict],
    ablation_names: List[str],
    output_path: str,
):
    """
    Plot ablation study results.

    Args:
        results_list: List of results dictionaries
        ablation_names: List of ablation configuration names
        output_path: Path to save plot
    """
    metrics = ["average_accuracy", "backward_transfer", "forward_transfer", "forgetting"]
    metric_labels = ["ACC", "BWT", "FWT", "Forgetting"]

    data = []
    for ablation_name, results in zip(ablation_names, results_list):
        summary = results["summary_metrics"]
        for metric, label in zip(metrics, metric_labels):
            data.append({
                "Configuration": ablation_name,
                "Metric": label,
                "Value": summary[metric],
            })

    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(ablation_names))
    width = 0.2

    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        metric_data = df[df["Metric"] == label]
        values = [
            metric_data[metric_data["Configuration"] == name]["Value"].values[0]
            for name in ablation_names
        ]
        ax.bar(x + idx * width, values, width, label=label)

    ax.set_xlabel("Configuration", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title("Ablation Study Results", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(ablation_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_metrics_table(
    results: Dict,
    output_path: str,
):
    """
    Generate LaTeX table of metrics.

    Args:
        results: Results dictionary
        output_path: Path to save table
    """
    summary = results["summary_metrics"]

    # Create table
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Metric & Value \\\\",
        "\\midrule",
    ]

    lines.append(f"Average Accuracy & {summary['average_accuracy']:.4f} \\\\")
    lines.append(f"Backward Transfer & {summary['backward_transfer']:.4f} \\\\")
    lines.append(f"Forward Transfer & {summary['forward_transfer']:.4f} \\\\")
    lines.append(f"Forgetting & {summary['forgetting']:.4f} \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Metrics Summary}",
        "\\end{table}",
    ])

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def generate_accuracy_matrix_table(
    results: Dict,
    output_path: str,
    task_names: Optional[List[str]] = None,
):
    """
    Generate LaTeX table of accuracy matrix.

    Args:
        results: Results dictionary
        output_path: Path to save table
        task_names: List of task names
    """
    accuracy_matrix = np.array(results["accuracy_matrix"])
    num_tasks = accuracy_matrix.shape[0]

    if task_names is None:
        task_names = [f"Task {i+1}" for i in range(num_tasks)]

    # Create table
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\begin{tabular}{l" + "c" * num_tasks + "}",
        "\\toprule",
        "Task & " + " & ".join([f"After T{i+1}" for i in range(num_tasks)]) + " \\\\",
        "\\midrule",
    ]

    for task_idx in range(num_tasks):
        row = [task_names[task_idx]]
        for col_idx in range(num_tasks):
            if col_idx >= task_idx:
                row.append(f"{accuracy_matrix[col_idx, task_idx]:.4f}")
            else:
                row.append("--")
        lines.append(" & ".join(row) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Accuracy Matrix R}",
        "\\end{table}",
    ])

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def main():
    """Main visualization entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Visualize experiment results")
    parser.add_argument("--experiment-dir", type=str, default="experiments")
    parser.add_argument("--experiment-name", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="plots")
    parser.add_argument("--task-names", nargs="+", help="Task names")

    args = parser.parse_args()

    # Load results
    results = load_results(args.experiment_dir, args.experiment_name)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Generate plots
    plot_accuracy_evolution(
        results,
        str(output_dir / "accuracy_evolution.png"),
        args.task_names,
    )

    # Generate tables
    generate_metrics_table(
        results,
        str(output_dir / "metrics_table.tex"),
    )
    generate_accuracy_matrix_table(
        results,
        str(output_dir / "accuracy_matrix.tex"),
        args.task_names,
    )

    print(f"Plots and tables saved to {output_dir}")


if __name__ == "__main__":
    main()
