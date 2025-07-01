#!/usr/bin/env python3
"""
ToMnet Figure 5 Visualization

This script reproduces the visualizations from Figure 5 with goal-directed agents.

Figure 5 shows:
- (b) N_past vs average posterior probability assigned to the true action
- (d) 2D embedding space of ToMnet for preferred objects with N_past=0

Usage:
    python visualize_figure5.py [--results_path RESULTS_PATH] [--save_plots] [--output_dir OUTPUT_DIR]
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.decomposition import PCA
import argparse

# Set style
try:
    plt.style.use("seaborn-v0_8")
except OSError:
    # Fallback for older matplotlib versions
    try:
        plt.style.use("seaborn")
    except:
        pass  # Use default style
sns.set_palette("husl")

# Object type colors for visualization (different from action colors)
OBJECT_COLORS = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "cyan"]
OBJECT_NAMES = ["Object 0", "Object 1", "Object 2", "Object 3", "Object 4", "Object 5", "Object 6", "Object 7", "Object 8"]


def load_evaluation_results(results_path="result/figure5/figure5_results.pkl"):
    """Load evaluation results from pickle file"""
    print(f"Loading results from: {results_path}")

    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path}")

    with open(results_path, "rb") as f:
        results = pickle.load(f)

    # Check the data structure
    print("Results loaded successfully!")
    print("Available models:", list(results.keys()))
    
    for model_name, model_results in results.items():
        print(f"\nModel: {model_name}")
        if "figure5b" in model_results:
            n_points = len(model_results["figure5b"]["n_past_values"])
            print(f"  Figure 5b data: {n_points} N_past values")
        if "figure5d" in model_results:
            n_samples = model_results["figure5d"]["n_samples"]
            print(f"  Figure 5d data: {n_samples} embedding samples")

    return results


def plot_figure5b_n_past_vs_likelihood(results):
    """Plot N_past vs average posterior probability assigned to true action (Figure 5b)"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Plot data for each model
    colors = plt.cm.Set1(np.linspace(0, 1, len(results)))
    
    for i, (model_name, model_results) in enumerate(results.items()):
        if "figure5b" not in model_results:
            print(f"Warning: No Figure 5b data found for model {model_name}")
            continue
            
        figure5b_data = model_results["figure5b"]
        
        n_past_values = np.array(figure5b_data["n_past_values"])
        avg_likelihoods = np.array(figure5b_data["avg_action_likelihoods"])
        std_likelihoods = np.array(figure5b_data["std_action_likelihoods"])
        n_samples = np.array(figure5b_data["n_samples"])
        
        # Filter out points with no samples
        valid_mask = n_samples > 0
        if not np.any(valid_mask):
            print(f"Warning: No valid data points for model {model_name}")
            continue
            
        n_past_valid = n_past_values[valid_mask]
        avg_likelihoods_valid = avg_likelihoods[valid_mask]
        std_likelihoods_valid = std_likelihoods[valid_mask]
        
        # Plot with error bars
        ax.errorbar(
            n_past_valid,
            avg_likelihoods_valid,
            yerr=std_likelihoods_valid,
            marker='o',
            markersize=8,
            linewidth=2,
            capsize=5,
            label=f"{model_name}",
            color=colors[i],
            alpha=0.8,
        )

    ax.set_xlabel("N_past (Number of Past Episodes)", fontsize=12)
    ax.set_ylabel("Average Posterior Probability\nAssigned to True Action", fontsize=12)
    ax.set_title(
        "Figure 5b: N_past vs Action Prediction Likelihood\n(Goal-Directed Agents)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    # Set integer ticks for x-axis
    ax.set_xticks(range(0, 11))

    # Add annotation
    ax.text(
        0.02,
        0.98,
        "Goal-directed agents:\nMore past episodes → Better predictions",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
    )

    return fig


def plot_figure5d_embedding_space(results):
    """Plot 2D embedding space colored by preferred objects with N_past=0 (Figure 5d)"""
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 6))
    
    # Handle single model case
    if len(results) == 1:
        axes = [axes]
    
    for idx, (model_name, model_results) in enumerate(results.items()):
        ax = axes[idx] if len(results) > 1 else axes[0]
        
        if "figure5d" not in model_results:
            print(f"Warning: No Figure 5d data found for model {model_name}")
            ax.text(0.5, 0.5, f"No Figure 5d data\nfor {model_name}", 
                   transform=ax.transAxes, ha='center', va='center',
                   bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.8))
            continue
            
        figure5d_data = model_results["figure5d"]
        
        embeddings_2d = figure5d_data["embeddings_2d"]
        preferred_objects = figure5d_data["preferred_objects"]
        n_samples = figure5d_data["n_samples"]
        
        if n_samples == 0 or len(embeddings_2d) == 0:
            ax.text(0.5, 0.5, f"No embedding data\nfor {model_name}", 
                   transform=ax.transAxes, ha='center', va='center',
                   bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.8))
            continue
        
        # Get unique preferred objects
        unique_objects = np.unique(preferred_objects)
        n_objects = len(unique_objects)
        
        # Plot scatter with colors representing different preferred objects
        for obj_idx in unique_objects:
            mask = preferred_objects == obj_idx
            if np.any(mask):
                color_idx = int(obj_idx) % len(OBJECT_COLORS)
                ax.scatter(
                    embeddings_2d[mask, 0],
                    embeddings_2d[mask, 1],
                    c=OBJECT_COLORS[color_idx],
                    alpha=0.7,
                    s=60,
                    edgecolors="black",
                    linewidth=0.5,
                    label=f"Prefers Object {int(obj_idx)}",
                )

        ax.set_xlabel("Normalized $e_1$", fontsize=12)
        ax.set_ylabel("Normalized $e_2$", fontsize=12)
        ax.set_title(
            f"{model_name}\n2D Character Embeddings (N_past=0)\nColored by Preferred Objects",
            fontsize=12,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="best")

        # Add annotation about the clustering
        ax.text(
            0.02,
            0.98,
            f"Agents clustered by object preference\nTotal samples: {n_samples}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8),
        )

    # Overall title
    fig.suptitle(
        "Figure 5d: 2D Embedding Space for Preferred Objects (N_past=0)",
        fontsize=16,
        fontweight="bold",
        y=0.95,
    )
    
    plt.tight_layout()
    return fig


def main():
    """Main function to generate all Figure 5 plots"""
    parser = argparse.ArgumentParser(description="Visualize ToMnet Figure 5 results")
    parser.add_argument(
        "--experiment",
        default="figure5",
        help="Experiment type (default: figure5)",
    )
    parser.add_argument(
        "--results_path",
        default=None,
        help="Path to evaluation results pickle file (default: result/{experiment}/figure5_results.pkl)",
    )
    parser.add_argument(
        "--save_plots",
        action="store_true",
        help="Save plots to files instead of displaying",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to save plots (default: plots/{experiment})",
    )

    args = parser.parse_args()

    # Set default paths if not provided
    if args.results_path is None:
        args.results_path = f"result/{args.experiment}/figure5_results.pkl"
    if args.output_dir is None:
        args.output_dir = f"plots/{args.experiment}"

    # Load results
    try:
        results = load_evaluation_results(args.results_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Results file not found: {args.results_path}")

    print("\n=== Generating Figure 5 Visualizations ===")

    # Create output directory if saving plots
    if args.save_plots:
        os.makedirs(args.output_dir, exist_ok=True)

    # Generate plots
    print("Generating Figure 5b: N_past vs Action Likelihood")
    fig5b = plot_figure5b_n_past_vs_likelihood(results)

    print("Generating Figure 5d: 2D Embedding Space for Preferred Objects")
    fig5d = plot_figure5d_embedding_space(results)

    # Save or display plots
    if args.save_plots:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"\nSaving plots to {args.output_dir}/")
        
        fig5b.savefig(
            f"{args.output_dir}/figure5b_n_past_vs_likelihood.png",
            dpi=300,
            bbox_inches="tight",
        )
        fig5d.savefig(
            f"{args.output_dir}/figure5d_embedding_space.png",
            dpi=300,
            bbox_inches="tight",
        )
        
        print("Plots saved successfully!")
    else:
        print("\nDisplaying plots...")
        plt.show()

    print("\n=== Visualization Complete ===")


if __name__ == "__main__":
    main()