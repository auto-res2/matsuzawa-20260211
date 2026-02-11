"""
Evaluation script for comparing multiple runs.
Independent script that fetches results from WandB and generates comparison visualizations.
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb


def fetch_run_data(entity: str, project: str, run_id: str) -> Dict:
    """
    Fetch run data from WandB API.
    
    Args:
        entity: WandB entity
        project: WandB project
        run_id: Run ID
        
    Returns:
        Dictionary with config, summary, and history
    """
    api = wandb.Api()
    
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
        
        data = {
            "run_id": run_id,
            "config": dict(run.config),
            "summary": dict(run.summary),
            "history": run.history().to_dict('records') if len(run.history()) > 0 else [],
        }
        
        return data
    except Exception as e:
        print(f"Warning: Could not fetch run {run_id}: {e}")
        return None


def export_per_run_metrics(data: Dict, results_dir: Path) -> None:
    """
    Export per-run metrics to JSON.
    
    Args:
        data: Run data from WandB
        results_dir: Directory to save metrics
    """
    run_dir = results_dir / data["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = {
        "run_id": data["run_id"],
        "accuracy": data["summary"].get("accuracy", 0),
        "correct": data["summary"].get("correct", 0),
        "total": data["summary"].get("total", 0),
        "demo_context_tokens": data["summary"].get("demo_context_tokens", 0),
        "method": data["config"].get("method", {}).get("name", "unknown"),
    }
    
    metrics_file = run_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Exported: {metrics_file}")


def create_per_run_figures(data: Dict, results_dir: Path) -> None:
    """
    Create per-run visualizations.
    
    Args:
        data: Run data from WandB
        results_dir: Directory to save figures
    """
    run_dir = results_dir / data["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Accuracy bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    accuracy = data["summary"].get("accuracy", 0)
    ax.bar(["Accuracy"], [accuracy], color="steelblue")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{data['run_id']}")
    
    fig_file = run_dir / "accuracy_bar.png"
    plt.tight_layout()
    plt.savefig(fig_file, dpi=150)
    plt.close()
    
    print(f"Exported: {fig_file}")


def export_aggregated_metrics(all_data: List[Dict], results_dir: Path) -> None:
    """
    Export aggregated comparison metrics.
    
    Args:
        all_data: List of run data dictionaries
        results_dir: Base results directory
    """
    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    # Organize by method type
    proposed = [d for d in all_data if d["config"].get("method", {}).get("type") == "proposed"]
    comparative = [d for d in all_data if d["config"].get("method", {}).get("type") == "comparative"]
    
    # Calculate metrics
    metrics_by_run = {}
    for d in all_data:
        metrics_by_run[d["run_id"]] = {
            "accuracy": d["summary"].get("accuracy", 0),
            "method": d["config"].get("method", {}).get("name", "unknown"),
            "type": d["config"].get("method", {}).get("type", "unknown"),
        }
    
    # Best proposed and baseline
    best_proposed = max([d["summary"].get("accuracy", 0) for d in proposed]) if proposed else 0
    best_baseline = max([d["summary"].get("accuracy", 0) for d in comparative]) if comparative else 0
    gap = best_proposed - best_baseline
    
    aggregated = {
        "primary_metric": "accuracy",
        "metrics_by_run": metrics_by_run,
        "best_proposed": best_proposed,
        "best_baseline": best_baseline,
        "gap": gap,
        "num_proposed": len(proposed),
        "num_baseline": len(comparative),
    }
    
    agg_file = comparison_dir / "aggregated_metrics.json"
    with open(agg_file, "w") as f:
        json.dump(aggregated, f, indent=2)
    
    print(f"Exported: {agg_file}")


def create_comparison_figures(all_data: List[Dict], results_dir: Path) -> None:
    """
    Create comparison visualizations across runs.
    
    Args:
        all_data: List of run data dictionaries
        results_dir: Base results directory
    """
    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data
    run_ids = [d["run_id"] for d in all_data]
    accuracies = [d["summary"].get("accuracy", 0) for d in all_data]
    methods = [d["config"].get("method", {}).get("name", "unknown") for d in all_data]
    types = [d["config"].get("method", {}).get("type", "unknown") for d in all_data]
    
    # Bar chart: accuracy by method
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["steelblue" if t == "proposed" else "coral" for t in types]
    bars = ax.bar(range(len(run_ids)), accuracies, color=colors)
    ax.set_xticks(range(len(run_ids)))
    ax.set_xticklabels([m[:15] for m in methods], rotation=45, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Comparison by Method")
    ax.set_ylim(0, 1)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="steelblue", label="Proposed"),
        Patch(facecolor="coral", label="Baseline"),
    ]
    ax.legend(handles=legend_elements)
    
    fig_file = comparison_dir / "accuracy_comparison.png"
    plt.tight_layout()
    plt.savefig(fig_file, dpi=150)
    plt.close()
    
    print(f"Exported: {fig_file}")
    
    # Box plot if multiple seeds per method
    if len(all_data) >= 3:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        proposed_acc = [d["summary"].get("accuracy", 0) for d in all_data if d["config"].get("method", {}).get("type") == "proposed"]
        baseline_acc = [d["summary"].get("accuracy", 0) for d in all_data if d["config"].get("method", {}).get("type") == "comparative"]
        
        data_to_plot = []
        labels = []
        if proposed_acc:
            data_to_plot.append(proposed_acc)
            labels.append("Proposed")
        if baseline_acc:
            data_to_plot.append(baseline_acc)
            labels.append("Baseline")
        
        if data_to_plot:
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], ["steelblue", "coral"]):
                patch.set_facecolor(color)
            
            ax.set_ylabel("Accuracy")
            ax.set_title("Accuracy Distribution by Method Type")
            ax.set_ylim(0, 1)
            
            fig_file = comparison_dir / "accuracy_boxplot.png"
            plt.tight_layout()
            plt.savefig(fig_file, dpi=150)
            plt.close()
            
            print(f"Exported: {fig_file}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate and compare runs")
    parser.add_argument("results_dir", type=str, help="Results directory")
    parser.add_argument("run_ids", type=str, help="JSON string list of run IDs")
    parser.add_argument("--entity", type=str, default="airas", help="WandB entity")
    parser.add_argument("--project", type=str, default="2026-02-11", help="WandB project")
    args = parser.parse_args()
    
    # Parse run IDs
    run_ids = json.loads(args.run_ids)
    results_dir = Path(args.results_dir)
    
    print(f"Evaluating {len(run_ids)} runs...")
    print(f"Entity: {args.entity}, Project: {args.project}")
    
    # Fetch data for all runs
    all_data = []
    for run_id in run_ids:
        print(f"\nFetching {run_id}...")
        data = fetch_run_data(args.entity, args.project, run_id)
        if data:
            all_data.append(data)
            
            # Export per-run metrics and figures
            export_per_run_metrics(data, results_dir)
            create_per_run_figures(data, results_dir)
    
    # Export aggregated metrics and comparison figures
    if all_data:
        print("\nCreating comparison visualizations...")
        export_aggregated_metrics(all_data, results_dir)
        create_comparison_figures(all_data, results_dir)
    
    print("\n✓ Evaluation complete")


if __name__ == "__main__":
    main()
