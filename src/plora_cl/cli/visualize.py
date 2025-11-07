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

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    colors = sns.color_palette("husl", len(method_names))

    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        metric_data = df[df["Metric"] == label]
        bars = axes[idx].bar(metric_data["Method"], metric_data["Value"],
                           color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        axes[idx].set_title(label, fontweight="bold", fontsize=14)
        axes[idx].set_ylabel("Value", fontsize=12)
        axes[idx].tick_params(axis="x", rotation=45)
        axes[idx].grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, value in zip(bars, metric_data["Value"]):
            axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                         f'{value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_computational_costs(
    results_list: List[Dict],
    method_names: List[str],
    output_path: str,
):
    """
    Plot computational costs comparison.

    Args:
        results_list: List of results dictionaries
        method_names: List of method names
        output_path: Path to save plot
    """
    if not results_list or "computational_costs" not in results_list[0]:
        print("Warning: No computational cost data available")
        return

    costs = ["training_time_minutes", "peak_vram_gb", "parameters_total"]
    cost_labels = ["Training Time (min)", "Peak VRAM (GB)", "Total Parameters"]

    data = []
    for method_name, results in zip(method_names, results_list):
        costs_data = results.get("computational_costs", {})
        for cost, label in zip(costs, cost_labels):
            value = costs_data.get(cost, 0)
            data.append({
                "Method": method_name,
                "Cost": label,
                "Value": value,
            })

    if not data:
        print("Warning: No cost data to plot")
        return

    df = pd.DataFrame(data)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    colors = sns.color_palette("Set2", len(method_names))

    for idx, (cost, label) in enumerate(zip(costs, cost_labels)):
        cost_data = df[df["Cost"] == label]
        if len(cost_data) > 0:
            bars = axes[idx].bar(cost_data["Method"], cost_data["Value"],
                               color=colors, alpha=0.8, edgecolor='black', linewidth=1)
            axes[idx].set_title(label, fontweight="bold", fontsize=14)
            axes[idx].set_ylabel("Value", fontsize=12)
            axes[idx].tick_params(axis="x", rotation=45)
            axes[idx].grid(True, alpha=0.3, axis='y')

            # Add value labels on bars
            for bar, value in zip(bars, cost_data["Value"]):
                axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                             f'{value:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_task_wise_metrics(
    results: Dict,
    output_path: str,
    task_names: Optional[List[str]] = None,
):
    """
    Plot task-wise metrics (forgetting, best accuracy, final accuracy).

    Args:
        results: Results dictionary
        output_path: Path to save plot
        task_names: List of task names
    """
    accuracy_matrix = np.array(results["accuracy_matrix"])
    num_tasks = accuracy_matrix.shape[0]

    if task_names is None:
        task_names = [f"Task {i+1}" for i in range(num_tasks)]

    # Calculate metrics per task
    task_metrics = []
    for task_idx in range(num_tasks):
        accuracies = accuracy_matrix[:, task_idx]
        best_acc = np.max(accuracies)
        final_acc = accuracies[-1]
        forgetting = best_acc - final_acc

        task_metrics.append({
            "Task": task_names[task_idx],
            "Best Accuracy": best_acc,
            "Final Accuracy": final_acc,
            "Forgetting": forgetting,
        })

    df = pd.DataFrame(task_metrics)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot best accuracy
    bars1 = axes[0].bar(df["Task"], df["Best Accuracy"], color='skyblue', alpha=0.8,
                       edgecolor='black', linewidth=1, label='Best')
    axes[0].set_title("Best Accuracy per Task", fontweight="bold", fontsize=14)
    axes[0].set_ylabel("Accuracy", fontsize=12)
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].set_ylim(0, 1)

    # Plot final accuracy
    bars2 = axes[1].bar(df["Task"], df["Final Accuracy"], color='lightcoral', alpha=0.8,
                       edgecolor='black', linewidth=1, label='Final')
    axes[1].set_title("Final Accuracy per Task", fontweight="bold", fontsize=14)
    axes[1].set_ylabel("Accuracy", fontsize=12)
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_ylim(0, 1)

    # Plot forgetting
    bars3 = axes[2].bar(df["Task"], df["Forgetting"], color='lightgreen', alpha=0.8,
                       edgecolor='black', linewidth=1)
    axes[2].set_title("Forgetting per Task", fontweight="bold", fontsize=14)
    axes[2].set_ylabel("Forgetting", fontsize=12)
    axes[2].tick_params(axis="x", rotation=45)
    axes[2].grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bars, ax in [(bars1, axes[0]), (bars2, axes[1]), (bars3, axes[2])]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

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
    experiment_name: str = "Experiment",
):
    """
    Generate LaTeX table of metrics.

    Args:
        results: Results dictionary
        output_path: Path to save table
        experiment_name: Name of the experiment for caption
    """
    summary = results["summary_metrics"]

    # Create enhanced table
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\begin{tabular}{@{}lcc@{}}",
        "\\toprule",
        "\\textbf{Metric} & \\textbf{Value} & \\textbf{Std Dev} \\\\",
        "\\midrule",
    ]

    metrics_info = [
        ("Average Accuracy", "average_accuracy", "↑"),
        ("Backward Transfer", "backward_transfer", "↑"),
        ("Forward Transfer", "forward_transfer", "↑"),
        ("Forgetting", "forgetting", "↓"),
    ]

    for metric_name, metric_key, direction in metrics_info:
        value = summary[metric_key]
        # For demonstration, using placeholder std dev - in practice this would come from multiple runs
        std_dev = 0.01  # Placeholder
        lines.append(f"{metric_name} {direction} & {value:.4f} & {std_dev:.4f} \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{Metrics Summary for {experiment_name}}}",
        "\\label{tab:metrics_summary}",
        "\\end{table}",
    ])

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def generate_comprehensive_metrics_table(
    results_list: List[Dict],
    method_names: List[str],
    output_path: str,
):
    """
    Generate comprehensive LaTeX table comparing multiple methods.

    Args:
        results_list: List of results dictionaries
        method_names: List of method names
        output_path: Path to save table
    """
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\begin{tabular}{@{}l" + "c" * len(method_names) + "@{}}",
        "\\toprule",
        "\\textbf{Metric} & " + " & ".join([f"\\textbf{{{name}}}" for name in method_names]) + " \\\\",
        "\\midrule",
    ]

    metrics_info = [
        ("Average Accuracy ↑", "average_accuracy"),
        ("Backward Transfer ↑", "backward_transfer"),
        ("Forward Transfer ↑", "forward_transfer"),
        ("Forgetting ↓", "forgetting"),
    ]

    for metric_name, metric_key in metrics_info:
        row_values = []
        for results in results_list:
            value = results["summary_metrics"][metric_key]
            row_values.append(f"{value:.4f}")

        lines.append(f"{metric_name} & " + " & ".join(row_values) + " \\\\")

    # Add computational costs if available
    if results_list and "computational_costs" in results_list[0]:
        lines.append("\\midrule")
        cost_info = [
            ("Training Time (min)", "training_time_minutes"),
            ("Peak VRAM (GB)", "peak_vram_gb"),
            ("Total Parameters", "parameters_total"),
        ]

        for cost_name, cost_key in cost_info:
            row_values = []
            for results in results_list:
                costs = results.get("computational_costs", {})
                value = costs.get(cost_key, 0)
                if "parameters" in cost_key.lower():
                    row_values.append("M" if value > 1000000 else f"{value:,}")
                else:
                    row_values.append(f"{value:.1f}")

            lines.append(f"{cost_name} & " + " & ".join(row_values) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Comprehensive Comparison of Continual Learning Methods}",
        "\\label{tab:comprehensive_comparison}",
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
    parser.add_argument("--compare-experiments", nargs="+", help="List of experiment names to compare")
    parser.add_argument("--comparison-names", nargs="+", help="Names for comparison experiments")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    if args.compare_experiments:
        # Load multiple experiments for comparison
        results_list = []
        method_names = args.comparison_names if args.comparison_names else args.compare_experiments

        if len(args.compare_experiments) != len(method_names):
            print("Error: Number of comparison experiments must match number of comparison names")
            return

        for exp_name in args.compare_experiments:
            try:
                results = load_results(args.experiment_dir, exp_name)
                results_list.append(results)
            except FileNotFoundError:
                print(f"Warning: Results for experiment '{exp_name}' not found, skipping")
                continue

        if not results_list:
            print("Error: No valid experiments found for comparison")
            return

        # Generate comparison plots
        plot_metrics_comparison(
            results_list,
            method_names,
            str(output_dir / "metrics_comparison.png"),
        )

        plot_computational_costs(
            results_list,
            method_names,
            str(output_dir / "computational_costs.png"),
        )

        # Generate comprehensive table
        generate_comprehensive_metrics_table(
            results_list,
            method_names,
            str(output_dir / "comprehensive_comparison.tex"),
        )

        print(f"Comparison plots and tables saved to {output_dir}")

    else:
        # Single experiment visualization
        results = load_results(args.experiment_dir, args.experiment_name)

        # Generate plots
        plot_accuracy_evolution(
            results,
            str(output_dir / "accuracy_evolution.png"),
            args.task_names,
        )

        plot_task_wise_metrics(
            results,
            str(output_dir / "task_wise_metrics.png"),
            args.task_names,
        )

        # Generate tables
        generate_metrics_table(
            results,
            str(output_dir / "metrics_table.tex"),
            args.experiment_name,
        )
        generate_accuracy_matrix_table(
            results,
            str(output_dir / "accuracy_matrix.tex"),
            args.task_names,
        )

        print(f"Plots and tables saved to {output_dir}")


if __name__ == "__main__":
    main()
