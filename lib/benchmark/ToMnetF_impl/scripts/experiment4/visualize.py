import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import json
import pickle
import os
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random
from config import Config

"""
Publication-quality visualization for ToMnetF (Experiment 4)
Extended with random positions and goal rewards visualization
@Author Filip Borowiak
"""


def generate_past_episodes_from_batch(
    trajectories, goals, batch_size, n_past_min, n_past_max, max_n_past
):
    """
    Generate past episodes by randomly sampling from other trajectories in the batch
    with the same goal, using fully vectorized operations for efficiency

    Args:
        trajectories: Batch of trajectories [batch_size, depth, height, width, time_step]
        goals: Batch of goal labels [batch_size]
        batch_size: Size of current batch
        n_past_min: Minimum number of past episodes to sample
        n_past_max: Maximum number of past episodes to sample
        max_n_past: Maximum number of past episodes for consistent tensor shape

    Returns:
        past_episodes_batch: [batch_size, max_n_past, depth, height, width, time_step]
    """
    device = trajectories.device
    depth, height, width, time_step = trajectories.shape[1:]

    # Initialize past episodes tensor
    past_episodes_batch = torch.zeros(
        (batch_size, max_n_past, depth, height, width, time_step),
        dtype=trajectories.dtype,
        device=device,
    )

    # Generate random n_past values for all samples at once
    n_past_values = torch.randint(
        n_past_min, n_past_max + 1, (batch_size,), device=device
    )

    # Create goal similarity matrix (batch_size x batch_size)
    # same_goal_mask[i, j] = True if sample i and j have the same goal
    goals_expanded = goals.unsqueeze(1)  # [batch_size, 1]
    same_goal_mask = goals_expanded == goals.unsqueeze(0)  # [batch_size, batch_size]

    # Exclude self-matches by setting diagonal to False
    same_goal_mask.fill_diagonal_(False)

    # Create random sampling matrix for all samples at once
    # For each sample, we create random indices for selecting past episodes
    rand_matrix = torch.rand(batch_size, batch_size, device=device)

    # Mask out invalid sources (different goals or self)
    rand_matrix = rand_matrix * same_goal_mask.float()

    # For each sample, sort the random values to get sampling order
    sorted_vals, sorted_indices = torch.sort(rand_matrix, dim=1, descending=True)

    # Process all samples in parallel
    for ep_idx in range(max_n_past):
        # Create mask for samples that need this episode
        needs_episode = n_past_values > ep_idx

        if needs_episode.any():
            # Make sure we don't exceed batch dimension
            effective_ep_idx = min(ep_idx, batch_size - 1)

            # Get the source indices for this episode position
            source_indices = sorted_indices[needs_episode, effective_ep_idx]

            # Check if source is valid (non-zero in sorted_vals means same goal)
            valid_sources = sorted_vals[needs_episode, effective_ep_idx] > 0

            # Create indices for assignment
            target_indices = torch.where(needs_episode)[0]
            valid_targets = target_indices[valid_sources]
            valid_sources_idx = source_indices[valid_sources]

            # Vectorized copy of trajectories
            if len(valid_targets) > 0:
                past_episodes_batch[valid_targets, ep_idx] = trajectories[
                    valid_sources_idx
                ]

    return past_episodes_batch


def plot_accuracy_by_n_past(results_by_n_past, output_dir=None, show_confidence=True):
    """
    Plot action accuracy as a function of N_past values

    Args:
        results_by_n_past: Dictionary with N_past values as keys and metrics as values
        output_dir: Directory to save plots
        show_confidence: Whether to show confidence intervals
    """
    plt.style.use("seaborn-v0_8")

    # Extract data
    n_past_values = sorted(results_by_n_past.keys())
    accuracies = [results_by_n_past[n]["accuracy"] for n in n_past_values]
    f1_scores = [results_by_n_past[n]["f1_score"] for n in n_past_values]

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot accuracy
    ax1.plot(
        n_past_values,
        accuracies,
        "o-",
        linewidth=2,
        markersize=8,
        color="steelblue",
        label="Action Accuracy",
    )
    ax1.set_xlabel("Number of Past Episodes (N_past)", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.set_title("Action Accuracy vs N_past", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.5, max(n_past_values) + 0.5)
    ax1.set_ylim(0, 1.0)

    # Add value labels on points
    for i, (n, acc) in enumerate(zip(n_past_values, accuracies)):
        ax1.annotate(
            f"{acc:.3f}",
            (n, acc),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=10,
        )

    # Plot F1 score
    ax2.plot(
        n_past_values,
        f1_scores,
        "o-",
        linewidth=2,
        markersize=8,
        color="darkgreen",
        label="F1 Score",
    )
    ax2.set_xlabel("Number of Past Episodes (N_past)", fontsize=12)
    ax2.set_ylabel("F1 Score", fontsize=12)
    ax2.set_title("F1 Score vs N_past", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.5, max(n_past_values) + 0.5)
    ax2.set_ylim(0, 1.0)

    # Add value labels on points
    for i, (n, f1) in enumerate(zip(n_past_values, f1_scores)):
        ax2.annotate(
            f"{f1:.3f}",
            (n, f1),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=10,
        )

    plt.tight_layout()

    # Save plot
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(
            os.path.join(output_dir, "accuracy_by_n_past.png"),
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()

    # Print summary statistics
    print("\nAccuracy by N_past Summary:")
    print("-" * 40)
    for n in n_past_values:
        print(
            f"N_past={n:2d}: Accuracy={accuracies[n_past_values.index(n)]:.4f}, "
            f"F1={f1_scores[n_past_values.index(n)]:.4f}"
        )

    best_n_past = n_past_values[np.argmax(accuracies)]
    print(f"\nBest N_past: {best_n_past} (Accuracy: {max(accuracies):.4f})")


def plot_accuracy_heatmap_by_n_past(results_by_n_past, output_dir=None):
    """
    Create a heatmap showing per-action accuracy by N_past values

    Args:
        results_by_n_past: Dictionary with N_past values as keys and metrics as values
        output_dir: Directory to save plots
    """
    plt.style.use("seaborn-v0_8")

    # Extract data and calculate per-action accuracy
    n_past_values = sorted(results_by_n_past.keys())
    action_names = ["Up", "Down", "Left", "Right"]  # Assuming 4 actions

    accuracy_matrix = []

    for n_past in n_past_values:
        predictions = results_by_n_past[n_past]["predictions"]
        targets = results_by_n_past[n_past]["targets"]

        # Calculate per-action accuracy
        action_accuracies = []
        for action in range(4):
            action_mask = np.array(targets) == action
            if np.sum(action_mask) > 0:
                action_acc = np.mean(np.array(predictions)[action_mask] == action)
                action_accuracies.append(action_acc)
            else:
                action_accuracies.append(0.0)

        accuracy_matrix.append(action_accuracies)

    accuracy_matrix = np.array(accuracy_matrix)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(
        accuracy_matrix,
        xticklabels=action_names,
        yticklabels=[f"N_past={n}" for n in n_past_values],
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        cbar_kws={"label": "Accuracy"},
        ax=ax,
    )

    ax.set_title("Per-Action Accuracy by N_past", fontsize=14, fontweight="bold")
    ax.set_xlabel("Action Type", fontsize=12)
    ax.set_ylabel("Number of Past Episodes", fontsize=12)

    plt.tight_layout()

    # Save plot
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(
            os.path.join(output_dir, "accuracy_heatmap_by_n_past.png"),
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()


# Set style for publication-quality plots
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


def visualize_successor_representation(
    sr_map, maze_map, output_path, title="Successor Representation"
):
    """
    Visualize successor representation as a heatmap overlay on the maze

    Args:
        sr_map: numpy array of shape (height, width) with SR values
        maze_map: numpy array of shape (height, width) with maze layout
        output_path: Path to save the visualization
        title: Title for the plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    gammas = [0.5, 0.9, 0.99]

    for i, (ax, gamma) in enumerate(zip(axes, gammas)):
        # Create a combined visualization
        combined_map = np.zeros_like(maze_map, dtype=float)

        # Set walls to -1 for clear distinction
        combined_map[maze_map == 0] = -1  # Walls

        # Overlay SR values on non-wall areas
        non_wall_mask = maze_map != 0
        if len(sr_map.shape) == 3:  # Multiple gamma values
            sr_values = sr_map[i]
        else:
            sr_values = sr_map

        combined_map[non_wall_mask] = sr_values[non_wall_mask]

        # Create heatmap
        im = ax.imshow(combined_map, cmap="viridis", interpolation="nearest")
        ax.set_title(f"SR (γ={gamma})", fontsize=12, fontweight="bold")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Visit Probability", rotation=270, labelpad=15)

        # Add grid
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(maze_map.shape[1]))
        ax.set_yticks(range(maze_map.shape[0]))

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def visualize_consumption_predictions(consumption_pred, consumption_true, output_path):
    """
    Visualize consumption predictions vs true labels

    Args:
        consumption_pred: Predicted consumption probabilities (batch_size, 4)
        consumption_true: True consumption labels (batch_size, 4)
        output_path: Path to save the visualization
    """
    goals = ["Goal A", "Goal B", "Goal C", "Goal D"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, (ax, goal) in enumerate(zip(axes, goals)):
        # Create scatter plot
        ax.scatter(consumption_true[:, i], consumption_pred[:, i], alpha=0.6, s=20)

        # Add diagonal line for perfect prediction
        min_val = min(consumption_true[:, i].min(), consumption_pred[:, i].min())
        max_val = max(consumption_true[:, i].max(), consumption_pred[:, i].max())
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            alpha=0.8,
            label="Perfect Prediction",
        )

        ax.set_xlabel("True Consumption")
        ax.set_ylabel("Predicted Consumption")
        ax.set_title(f"{goal} Consumption Prediction")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Consumption Prediction Accuracy", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def visualize_maze_trajectory_with_sr(
    maze_map, trajectory_positions, sr_map, output_path
):
    """
    Visualize maze with agent trajectory and SR overlay

    Args:
        maze_map: numpy array of maze layout
        trajectory_positions: List of (x, y) positions
        sr_map: SR map to overlay
        output_path: Path to save visualization
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    gammas = [0.5, 0.9, 0.99]

    for i, (ax, gamma) in enumerate(zip(axes, gammas)):
        # Start with maze layout
        display_map = maze_map.copy().astype(float)

        # Overlay SR values
        if len(sr_map.shape) == 3:
            sr_values = sr_map[i]
        else:
            sr_values = sr_map

        # Normalize SR values for better visualization
        sr_normalized = (sr_values - sr_values.min()) / (
            sr_values.max() - sr_values.min() + 1e-8
        )

        # Create visualization
        im = ax.imshow(sr_normalized, cmap="YlOrRd", alpha=0.7)

        # Overlay maze walls
        wall_mask = maze_map == 0
        ax.imshow(np.where(wall_mask, 0, np.nan), cmap="gray", alpha=0.8)

        # Plot trajectory
        if trajectory_positions:
            traj_x = [
                pos[1] for pos in trajectory_positions
            ]  # Note: x,y swapped for matplotlib
            traj_y = [pos[0] for pos in trajectory_positions]
            ax.plot(
                traj_x, traj_y, "b-", linewidth=3, alpha=0.8, label="Agent Trajectory"
            )
            ax.plot(traj_x[0], traj_y[0], "go", markersize=10, label="Start")
            ax.plot(traj_x[-1], traj_y[-1], "ro", markersize=10, label="End")

        ax.set_title(f"SR (γ={gamma}) with Trajectory")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(
        "Successor Representation with Agent Trajectory", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_training_curves(history_path, output_dir, experiment_no):
    """
    Plot training and validation curves including component losses

    Args:
        history_path: Path to training history JSON
        output_dir: Output directory for plots
        experiment_no: Experiment number
    """

    with open(history_path, "r") as f:
        history = json.load(f)

    # Check if component losses are available
    has_component_losses = "train_action_loss" in history

    if has_component_losses:
        # Create 3x2 subplot grid for comprehensive visualization
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))
        ax1, ax2 = axes[0]
        ax3, ax4 = axes[1]
        ax5, ax6 = axes[2]
    else:
        # Original 1x2 layout
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    epochs = history["epoch"]

    # Accuracy plot
    ax1.plot(
        epochs,
        history["train_accuracy"],
        label="Training",
        linewidth=2,
        marker="o",
        markersize=4,
    )
    ax1.plot(
        epochs,
        history["val_accuracy"],
        label="Validation",
        linewidth=2,
        marker="s",
        markersize=4,
    )
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Accuracy (%)", fontsize=12)
    ax1.set_title("Model Accuracy", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max(epochs))

    # Total Loss plot
    ax2.plot(
        epochs,
        history["train_loss"],
        label="Training",
        linewidth=2,
        marker="o",
        markersize=4,
    )
    ax2.plot(
        epochs,
        history["val_loss"],
        label="Validation",
        linewidth=2,
        marker="s",
        markersize=4,
    )
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Loss", fontsize=12)
    ax2.set_title("Total Loss", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max(epochs))

    if has_component_losses:
        # Action Loss plot
        ax3.plot(
            epochs,
            history["train_action_loss"],
            label="Training",
            linewidth=2,
            marker="o",
            markersize=4,
            color="green",
        )
        ax3.plot(
            epochs,
            history["val_action_loss"],
            label="Validation",
            linewidth=2,
            marker="s",
            markersize=4,
            color="lightgreen",
        )
        ax3.set_xlabel("Epoch", fontsize=12)
        ax3.set_ylabel("Loss", fontsize=12)
        ax3.set_title("Action Loss", fontsize=14, fontweight="bold")
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, max(epochs))

        # Consumption Loss plot
        ax4.plot(
            epochs,
            history["train_consumption_loss"],
            label="Training",
            linewidth=2,
            marker="o",
            markersize=4,
            color="red",
        )
        ax4.plot(
            epochs,
            history["val_consumption_loss"],
            label="Validation",
            linewidth=2,
            marker="s",
            markersize=4,
            color="salmon",
        )
        ax4.set_xlabel("Epoch", fontsize=12)
        ax4.set_ylabel("Loss", fontsize=12)
        ax4.set_title("Consumption Loss", fontsize=14, fontweight="bold")
        ax4.legend(fontsize=11)
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(0, max(epochs))

        # SR Loss plot
        ax5.plot(
            epochs,
            history["train_sr_loss"],
            label="Training",
            linewidth=2,
            marker="o",
            markersize=4,
            color="purple",
        )
        ax5.plot(
            epochs,
            history["val_sr_loss"],
            label="Validation",
            linewidth=2,
            marker="s",
            markersize=4,
            color="plum",
        )
        ax5.set_xlabel("Epoch", fontsize=12)
        ax5.set_ylabel("Loss", fontsize=12)
        ax5.set_title("Successor Representation Loss", fontsize=14, fontweight="bold")
        ax5.legend(fontsize=11)
        ax5.grid(True, alpha=0.3)
        ax5.set_xlim(0, max(epochs))

        # Epoch Time plot
        if "epoch_time" in history:
            ax6.plot(
                epochs,
                history["epoch_time"],
                label="Time per epoch",
                linewidth=2,
                marker="o",
                markersize=4,
                color="orange",
            )
            ax6.set_xlabel("Epoch", fontsize=12)
            ax6.set_ylabel("Time (seconds)", fontsize=12)
            ax6.set_title("Training Time per Epoch", fontsize=14, fontweight="bold")
            ax6.legend(fontsize=11)
            ax6.grid(True, alpha=0.3)
            ax6.set_xlim(0, max(epochs))

            # Add average time annotation
            avg_time = sum(history["epoch_time"]) / len(history["epoch_time"])
            ax6.axhline(y=avg_time, color="red", linestyle="--", alpha=0.7)
            ax6.text(
                max(epochs) * 0.02,
                avg_time * 1.05,
                f"Avg: {avg_time:.2f}s",
                color="red",
                fontsize=10,
                fontweight="bold",
            )
        else:
            ax6.text(
                0.5,
                0.5,
                "Epoch time data not available",
                ha="center",
                va="center",
                transform=ax6.transAxes,
            )
            ax6.set_visible(False)

    plt.tight_layout()

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"training_curves_exp{experiment_no}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Training curves saved to: {plot_path}")


def plot_confusion_matrix(predictions_path, output_dir, experiment_no):
    """
    Plot confusion matrix

    Args:
        predictions_path: Path to predictions pickle file
        output_dir: Output directory for plots
        experiment_no: Experiment number
    """

    with open(predictions_path, "rb") as f:
        data = pickle.load(f)

    predictions = np.array(data["predictions"])
    targets = np.array(data["targets"])

    # Calculate confusion matrix
    cm = confusion_matrix(targets, predictions)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Action labels
    action_labels = ["UP", "RIGHT", "DOWN", "LEFT"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Raw counts
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=action_labels,
        yticklabels=action_labels,
        ax=ax1,
        cbar_kws={"label": "Count"},
    )
    ax1.set_title("Confusion Matrix (Counts)", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Predicted Action", fontsize=12)
    ax1.set_ylabel("True Action", fontsize=12)

    # Normalized
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        xticklabels=action_labels,
        yticklabels=action_labels,
        ax=ax2,
        cbar_kws={"label": "Proportion"},
    )
    ax2.set_title("Confusion Matrix (Normalized)", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Predicted Action", fontsize=12)
    ax2.set_ylabel("True Action", fontsize=12)

    plt.tight_layout()

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"confusion_matrix_exp{experiment_no}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Confusion matrix saved to: {plot_path}")


def plot_action_likelihood(predictions_path, output_dir, experiment_no):
    """
    Plot action likelihood distributions

    Args:
        predictions_path: Path to predictions pickle file
        output_dir: Output directory for plots
        experiment_no: Experiment number
    """

    with open(predictions_path, "rb") as f:
        data = pickle.load(f)

    predictions = np.array(data["predictions"])
    targets = np.array(data["targets"])
    probabilities = np.array(data["probabilities"])

    # Get likelihood for correct predictions
    correct_mask = predictions == targets
    correct_likelihoods = probabilities[correct_mask, targets[correct_mask]]
    incorrect_likelihoods = probabilities[~correct_mask, targets[~correct_mask]]

    # Action-wise likelihood distributions
    action_labels = ["UP", "RIGHT", "DOWN", "LEFT"]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    axes = [ax1, ax2, ax3, ax4]

    for action in range(4):
        ax = axes[action]

        # Get likelihoods for this action
        action_mask = targets == action
        action_correct = correct_mask & action_mask
        action_incorrect = (~correct_mask) & action_mask

        if np.sum(action_correct) > 0:
            correct_probs = probabilities[action_correct, action]
            ax.hist(correct_probs, bins=30, alpha=0.7, label="Correct", density=True)

        if np.sum(action_incorrect) > 0:
            incorrect_probs = probabilities[action_incorrect, action]
            ax.hist(
                incorrect_probs, bins=30, alpha=0.7, label="Incorrect", density=True
            )

        ax.set_title(f"Action: {action_labels[action]}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted Likelihood", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Action Likelihood Distributions", fontsize=16, fontweight="bold")
    plt.tight_layout()

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"action_likelihood_exp{experiment_no}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Action likelihood plot saved to: {plot_path}")


def plot_cross_species_results(results_path, output_dir, experiment_no):
    """
    Plot cross-species evaluation results

    Args:
        results_path: Path to cross-species results JSON
        output_dir: Output directory for plots
        experiment_no: Experiment number
    """

    with open(results_path, "r") as f:
        results = json.load(f)

    cross_species = results["cross_species_results"]

    # Prepare data for plotting
    models = list(cross_species.keys())
    datasets = list(cross_species[models[0]].keys()) if models else []

    accuracy_matrix = np.zeros((len(models), len(datasets)))
    f1_matrix = np.zeros((len(models), len(datasets)))

    for i, model in enumerate(models):
        for j, dataset in enumerate(datasets):
            accuracy_matrix[i, j] = cross_species[model][dataset]["accuracy"]
            f1_matrix[i, j] = cross_species[model][dataset]["f1_score"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Accuracy heatmap
    sns.heatmap(
        accuracy_matrix,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        xticklabels=datasets,
        yticklabels=models,
        ax=ax1,
        cbar_kws={"label": "Accuracy"},
    )
    ax1.set_title("Cross-Species Accuracy", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Test Dataset", fontsize=12)
    ax1.set_ylabel("Model", fontsize=12)

    # F1 score heatmap
    sns.heatmap(
        f1_matrix,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        xticklabels=datasets,
        yticklabels=models,
        ax=ax2,
        cbar_kws={"label": "F1 Score"},
    )
    ax2.set_title("Cross-Species F1 Score", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Test Dataset", fontsize=12)
    ax2.set_ylabel("Model", fontsize=12)

    plt.tight_layout()

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(
        output_dir, f"cross_species_results_exp{experiment_no}.png"
    )
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Cross-species results plot saved to: {plot_path}")


def plot_character_embeddings(
    model, data_loader, device, output_dir, experiment_no, n_samples=1000
):
    """
    Plot character embeddings using dimensionality reduction

    Args:
        model: Trained ToMnet model
        data_loader: Data loader
        device: Computing device
        output_dir: Output directory for plots
        experiment_no: Experiment number
        n_samples: Number of samples to use
    """

    # Initialize config for past episodes generation
    config = Config()

    model.eval()
    embeddings = []
    actions = []

    sample_count = 0
    with torch.no_grad():
        for batch in data_loader:
            if sample_count >= n_samples:
                break

            # Handle both 3-element and 4-element batches
            if len(batch) == 4:
                traj, curr, act, goals = batch
            else:
                traj, curr, act = batch
                # Create dummy goals if not available
                goals = torch.zeros(traj.size(0), dtype=torch.long, device=device)

            traj, curr = traj.to(device), curr.to(device)
            goals = goals.to(device)
            act = act.squeeze(-1).type(torch.long)

            # Generate past episodes from batch trajectories with same goal
            past_episodes_batch = generate_past_episodes_from_batch(
                trajectories=traj,
                goals=goals,
                batch_size=traj.size(0),
                n_past_min=config.n_past_min,
                n_past_max=config.n_past_max,
                max_n_past=config.n_past_max,
            )

            # Get character embeddings using past episodes
            e_char = model.char_net(past_episodes_batch)

            # Take the last timestep embedding
            char_embedding = e_char.cpu().numpy()

            for i in range(len(act)):
                if sample_count >= n_samples:
                    break

                embeddings.append(char_embedding[i])
                actions.append(act[i].item())
                sample_count += 1

    embeddings = np.array(embeddings)
    actions = np.array(actions)

    # Dimensionality reduction
    if embeddings.shape[1] > 2:
        # PCA
        pca = PCA(n_components=2)
        embeddings_pca = pca.fit_transform(embeddings)

        # t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_tsne = tsne.fit_transform(embeddings)
    else:
        embeddings_pca = embeddings
        embeddings_tsne = embeddings

    # Action labels and colors
    action_labels = ["UP", "RIGHT", "DOWN", "LEFT"]
    colors = ["red", "blue", "green", "orange"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # PCA plot
    for action in range(4):
        mask = actions == action
        if np.sum(mask) > 0:
            ax1.scatter(
                embeddings_pca[mask, 0],
                embeddings_pca[mask, 1],
                c=colors[action],
                label=action_labels[action],
                alpha=0.6,
                s=20,
            )

    ax1.set_title("Character Embeddings (PCA)", fontsize=14, fontweight="bold")
    ax1.set_xlabel(f"PC1 (var: {pca.explained_variance_ratio_[0]:.3f})", fontsize=12)
    ax1.set_ylabel(f"PC2 (var: {pca.explained_variance_ratio_[1]:.3f})", fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # t-SNE plot
    for action in range(4):
        mask = actions == action
        if np.sum(mask) > 0:
            ax2.scatter(
                embeddings_tsne[mask, 0],
                embeddings_tsne[mask, 1],
                c=colors[action],
                label=action_labels[action],
                alpha=0.6,
                s=20,
            )

    ax2.set_title("Character Embeddings (t-SNE)", fontsize=14, fontweight="bold")
    ax2.set_xlabel("t-SNE 1", fontsize=12)
    ax2.set_ylabel("t-SNE 2", fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"character_embeddings_exp{experiment_no}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Character embeddings plot saved to: {plot_path}")


def create_additional_visualizations(
    model, val_loader, plot_dir, experiment_no, device, has_n_past, config=None
):
    """
    Create and save additional visualizations including successor representation
    Moved from train.py for better organization
    """

    print("Creating additional visualizations...")

    # Import Config if not provided
    if config is None:
        from config import Config

        config = Config()

    # Get a sample batch for visualization
    model.eval()
    with torch.no_grad():
        sample_batch = next(iter(val_loader))

        # Parse data based on what's available
        traj, curr, act, goals = (
            sample_batch[0],
            sample_batch[1],
            sample_batch[2],
            sample_batch[3],
        )
        traj, curr, act, goals = (
            traj.to(device),
            curr.to(device),
            act.to(device),
            goals.to(device),
        )

        data_idx = 4  # Updated since we now have goals at index 3

        # Handle consumption and SR labels
        if len(sample_batch) > data_idx:
            consumption_target = sample_batch[data_idx].to(device)
            sr_target = sample_batch[data_idx + 1].to(device)
            data_idx += 2
        else:
            # Create dummy targets
            batch_size = act.size(0)
            consumption_target = torch.zeros(batch_size, 4).to(device)
            sr_target = torch.zeros(batch_size, 3, 13, 13).to(device)

        # Handle N_past data - generate from batch trajectories with same goal
        model_inputs = [traj, curr]
        if has_n_past:
            # Generate past episodes from other trajectories in the batch with same goal
            # Use config values instead of hardcoded defaults
            past_episodes_batch = generate_past_episodes_from_batch(
                trajectories=traj,
                goals=goals,
                batch_size=traj.size(0),
                n_past_min=config.n_past_min,  # From config
                n_past_max=config.n_past_max,  # From config
                max_n_past=config.n_past_max,  # From config
            )
            model_inputs.append(past_episodes_batch)

        # Get model predictions
        action_pred, consumption_pred, sr_pred = model(model_inputs)

        # 1. Visualize Successor Representation (training version)
        visualize_successor_representation_train(
            sr_pred, sr_target, plot_dir, experiment_no
        )

        # 2. Visualize Consumption Predictions (training version)
        visualize_consumption_predictions_train(
            consumption_pred, consumption_target, plot_dir, experiment_no
        )

        # 3. Visualize Action Predictions (training version)
        visualize_action_predictions_train(action_pred, act, plot_dir, experiment_no)

        # 4. Visualize Past Episodes if available
        if has_n_past:
            visualize_past_episodes_train(past_episodes_batch, plot_dir, experiment_no)

    print(f"Additional visualizations saved to: {plot_dir}")


def visualize_successor_representation_train(
    sr_pred, sr_target, plot_dir, experiment_no
):
    """Visualize successor representation predictions and targets (training version)"""

    # Take first sample from batch for visualization
    sr_pred_sample = sr_pred[0].cpu().numpy()  # Shape: (3, 13, 13)
    sr_target_sample = sr_target[0].cpu().numpy()  # Shape: (3, 13, 13)

    gamma_values = [0.5, 0.9, 0.99]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        "Successor Representation: Predictions vs Targets (Training)", fontsize=16
    )

    for i, gamma in enumerate(gamma_values):
        # Prediction
        im1 = axes[0, i].imshow(
            sr_pred_sample[i], cmap="viridis", interpolation="nearest"
        )
        axes[0, i].set_title(f"Prediction γ={gamma}")
        axes[0, i].set_xlabel("X coordinate")
        axes[0, i].set_ylabel("Y coordinate")
        plt.colorbar(im1, ax=axes[0, i])

        # Target
        im2 = axes[1, i].imshow(
            sr_target_sample[i], cmap="viridis", interpolation="nearest"
        )
        axes[1, i].set_title(f"Target γ={gamma}")
        axes[1, i].set_xlabel("X coordinate")
        axes[1, i].set_ylabel("Y coordinate")
        plt.colorbar(im2, ax=axes[1, i])

    plt.tight_layout()
    sr_path = os.path.join(
        plot_dir, f"exp{experiment_no}_successor_representation_train.png"
    )
    plt.savefig(sr_path, dpi=150, bbox_inches="tight")
    plt.close()


def visualize_consumption_predictions_train(
    consumption_pred, consumption_target, plot_dir, experiment_no
):
    """Visualize consumption predictions (training version)"""

    # Apply sigmoid to get probabilities
    consumption_prob = torch.sigmoid(consumption_pred).cpu().numpy()
    consumption_target_np = consumption_target.cpu().numpy()

    # Take first few samples for visualization
    n_samples = min(10, consumption_pred.size(0))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot predictions
    goals = ["A", "B", "C", "D"]
    x_pos = np.arange(len(goals))

    # Average probabilities across samples
    mean_pred = np.mean(consumption_prob[:n_samples], axis=0)
    mean_target = np.mean(consumption_target_np[:n_samples], axis=0)

    ax1.bar(x_pos, mean_pred, alpha=0.7, label="Predictions")
    ax1.set_xlabel("Goals")
    ax1.set_ylabel("Consumption Probability")
    ax1.set_title("Average Consumption Predictions (Training)")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(goals)
    ax1.grid(True, alpha=0.3)

    ax2.bar(x_pos, mean_target, alpha=0.7, color="orange", label="Targets")
    ax2.set_xlabel("Goals")
    ax2.set_ylabel("Consumption Probability")
    ax2.set_title("Average Consumption Targets (Training)")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(goals)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    consumption_path = os.path.join(
        plot_dir, f"exp{experiment_no}_consumption_predictions_train.png"
    )
    plt.savefig(consumption_path, dpi=150, bbox_inches="tight")
    plt.close()


def visualize_action_predictions_train(
    action_pred, action_target, plot_dir, experiment_no
):
    """Visualize action prediction distribution (training version)"""

    # Apply softmax to get probabilities
    action_prob = torch.softmax(action_pred, dim=1).cpu().numpy()
    action_target_np = action_target.cpu().numpy()

    # Action mapping
    actions = ["UP", "RIGHT", "DOWN", "LEFT"]

    # Create action distribution plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Predicted action distribution (average across batch)
    mean_action_prob = np.mean(action_prob, axis=0)
    ax1.bar(actions, mean_action_prob, alpha=0.7, color="skyblue")
    ax1.set_xlabel("Actions")
    ax1.set_ylabel("Probability")
    ax1.set_title("Average Action Predictions (Training)")
    ax1.grid(True, alpha=0.3)

    # Target action distribution
    action_counts = np.bincount(action_target_np.astype(int), minlength=4)
    action_dist = action_counts / action_counts.sum()
    ax2.bar(actions, action_dist, alpha=0.7, color="lightcoral")
    ax2.set_xlabel("Actions")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Target Action Distribution (Training)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    action_path = os.path.join(
        plot_dir, f"exp{experiment_no}_action_predictions_train.png"
    )
    plt.savefig(action_path, dpi=150, bbox_inches="tight")
    plt.close()


def visualize_past_episodes_train(past_episodes, plot_dir, experiment_no):
    """Visualize past episodes structure (training version)"""

    # Take first sample from batch
    past_eps_sample = (
        past_episodes[0].cpu().numpy()
    )  # Shape: (n_past_max, depth, height, width, time_step)
    n_past_max, depth, height, width, time_step = past_eps_sample.shape

    # Count non-zero episodes
    non_zero_episodes = []
    for ep_idx in range(n_past_max):
        episode = past_eps_sample[ep_idx]
        if np.sum(episode) > 0:
            non_zero_episodes.append(ep_idx)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Past Episodes Analysis (Training Sample)", fontsize=16)

    # Plot 1: Episode usage
    episode_usage = [1 if i in non_zero_episodes else 0 for i in range(n_past_max)]
    axes[0, 0].bar(range(n_past_max), episode_usage)
    axes[0, 0].set_xlabel("Episode Index")
    axes[0, 0].set_ylabel("Used (1) / Unused (0)")
    axes[0, 0].set_title(
        f"Episode Usage ({len(non_zero_episodes)}/{n_past_max} episodes used)"
    )
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Episode lengths (if any episodes exist)
    if non_zero_episodes:
        episode_lengths = []
        for ep_idx in non_zero_episodes:
            episode = past_eps_sample[ep_idx]
            # Count time steps with non-zero values
            length = 0
            for t in range(time_step):
                if np.sum(episode[:, :, :, t]) > 0:
                    length = t + 1
            episode_lengths.append(length)

        axes[0, 1].bar(range(len(episode_lengths)), episode_lengths)
        axes[0, 1].set_xlabel("Episode Index (used only)")
        axes[0, 1].set_ylabel("Episode Length")
        axes[0, 1].set_title("Trajectory Lengths of Used Episodes")
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(
            0.5,
            0.5,
            "No episodes used",
            ha="center",
            va="center",
            transform=axes[0, 1].transAxes,
        )
        axes[0, 1].set_title("Episode Lengths")

    # Plot 3: Visualize first used episode (agent positions over time)
    if non_zero_episodes:
        first_ep_idx = non_zero_episodes[0]
        episode = past_eps_sample[first_ep_idx]

        # Extract agent positions over time (channel 1 is agent position)
        agent_positions = episode[1, :, :, :]  # Shape: (height, width, time_step)

        # Find agent positions for each time step
        positions_over_time = []
        for t in range(time_step):
            pos = np.where(agent_positions[:, :, t] == 1)
            if len(pos[0]) > 0:
                positions_over_time.append((pos[0][0], pos[1][0]))

        if positions_over_time:
            x_pos, y_pos = zip(*positions_over_time)
            axes[1, 0].plot(x_pos, y_pos, "b-o", markersize=4, linewidth=2)
            axes[1, 0].set_xlim(0, height - 1)
            axes[1, 0].set_ylim(0, width - 1)
            axes[1, 0].set_xlabel("X coordinate")
            axes[1, 0].set_ylabel("Y coordinate")
            axes[1, 0].set_title(f"Agent Trajectory (Episode {first_ep_idx})")
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].invert_yaxis()
        else:
            axes[1, 0].text(
                0.5,
                0.5,
                "No valid positions",
                ha="center",
                va="center",
                transform=axes[1, 0].transAxes,
            )
    else:
        axes[1, 0].text(
            0.5,
            0.5,
            "No episodes to display",
            ha="center",
            va="center",
            transform=axes[1, 0].transAxes,
        )

    # Plot 4: Action distribution across all past episodes
    if non_zero_episodes:
        action_counts = np.zeros(4)  # 4 actions
        for ep_idx in non_zero_episodes:
            episode = past_eps_sample[ep_idx]
            # Sum across action channels (channels 6-9 are actions)
            for action_ch in range(4):
                action_counts[action_ch] += np.sum(episode[6 + action_ch])

        if action_counts.sum() > 0:
            action_counts = action_counts / action_counts.sum()
            actions = ["UP", "RIGHT", "DOWN", "LEFT"]
            axes[1, 1].bar(actions, action_counts)
            axes[1, 1].set_xlabel("Actions")
            axes[1, 1].set_ylabel("Frequency")
            axes[1, 1].set_title("Action Distribution (All Past Episodes)")
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(
                0.5,
                0.5,
                "No actions found",
                ha="center",
                va="center",
                transform=axes[1, 1].transAxes,
            )
    else:
        axes[1, 1].text(
            0.5,
            0.5,
            "No episodes to analyze",
            ha="center",
            va="center",
            transform=axes[1, 1].transAxes,
        )

    plt.tight_layout()
    past_episodes_path = os.path.join(
        plot_dir, f"exp{experiment_no}_past_episodes_train.png"
    )
    plt.savefig(past_episodes_path, dpi=150, bbox_inches="tight")
    plt.close()


def create_summary_report(
    experiment_no,
    result_dir="../../result/experiment4",
    plot_dir="../../plots/experiment4",
    model=None,
    val_loader=None,
    device=None,
    has_n_past=False,
    config=None,
):
    """
    Create a summary report with all visualizations

    Args:
        experiment_no: Experiment number
        result_dir: Results directory
        plot_dir: Plots directory
        model: Trained model (optional, for additional visualizations)
        val_loader: Validation data loader (optional)
        device: Computing device (optional)
        has_n_past: Whether N_past data is available (optional)
        config: Config object (optional)
    """

    print(f"Creating summary report for experiment {experiment_no}")

    # Training curves
    history_path = os.path.join(result_dir, f"exp{experiment_no}_training_history.json")
    if os.path.exists(history_path):
        plot_training_curves(history_path, plot_dir, experiment_no)

    # Confusion matrix and action likelihood
    predictions_path = os.path.join(
        result_dir,
        f"exp{experiment_no}_best_processed_data_exp{experiment_no}",
        "predictions.pkl",
    )
    if os.path.exists(predictions_path):
        plot_confusion_matrix(predictions_path, plot_dir, experiment_no)
        plot_action_likelihood(predictions_path, plot_dir, experiment_no)

    # Cross-species results
    cross_species_path = os.path.join(
        result_dir, f"cross_species_evaluation_exp{experiment_no}.json"
    )
    if os.path.exists(cross_species_path):
        plot_cross_species_results(cross_species_path, plot_dir, experiment_no)

    print(f"Summary report completed. Plots saved to: {plot_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create publication-quality visualizations for ToMnetF"
    )
    parser.add_argument(
        "--experiment_no", type=int, default=4, help="Experiment number"
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="../../result/experiment4",
        help="Results directory",
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default="../../plots/experiment4",
        help="Plots directory",
    )
    parser.add_argument(
        "--history_path",
        type=str,
        default=None,
        help="Path to training history JSON file",
    )
    parser.add_argument(
        "--predictions_path",
        type=str,
        default=None,
        help="Path to predictions pickle file",
    )
    parser.add_argument(
        "--cross_species_path",
        type=str,
        default=None,
        help="Path to cross-species results JSON file",
    )
    parser.add_argument(
        "--plot_type",
        type=str,
        default="all",
        choices=[
            "all",
            "training",
            "confusion",
            "likelihood",
            "cross_species",
            "embeddings",
        ],
        help="Type of plot to generate",
    )

    args = parser.parse_args()

    if args.plot_type == "all":
        create_summary_report(
            experiment_no=args.experiment_no,
            result_dir=args.result_dir,
            plot_dir=args.plot_dir,
        )
    if args.plot_type in ["all", "training"] and args.history_path:
        plot_training_curves(args.history_path, args.plot_dir, args.experiment_no)
    if args.plot_type in ["all", "confusion"] and args.predictions_path:
        plot_confusion_matrix(args.predictions_path, args.plot_dir, args.experiment_no)
    if args.plot_type in ["all", "likelihood"] and args.predictions_path:
        plot_action_likelihood(args.predictions_path, args.plot_dir, args.experiment_no)
    if args.plot_type in ["all", "cross_species"] and args.cross_species_path:
        plot_cross_species_results(
            args.cross_species_path, args.plot_dir, args.experiment_no
        )
    if args.plot_type in ["all", "embeddings"]:
        # Need to load model and data for embeddings visualization
        print("Creating character embeddings visualization...")
        from config import Config
        from torch.utils.data import TensorDataset

        config = Config()

        # Load model
        model_path = os.path.join(config.model_dir, f"exp{args.experiment_no}_best.pth")
        if os.path.exists(model_path):
            import sys

            sys.path.append("..")
            from tomnet import ToMnet

            device = config.device if torch.cuda.is_available() else "cpu"
            model_kwargs = config.get_model_kwargs()
            model = ToMnet(**model_kwargs)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()

            # Load test data
            test_data_path = os.path.join(
                config.data_dir, f"processed_data_exp{args.experiment_no}.pkl"
            )
            with open(test_data_path, "rb") as f:
                test_data = pickle.load(f)
            test_dataset = TensorDataset(
                test_data["data_trajectories"],
                test_data["data_current_state"],
                test_data["data_actions"],
                test_data["data_labels"],  # Include goals
            )
            test_loader = DataLoader(
                test_dataset, batch_size=config.batch_size, shuffle=False
            )

            # Create visualization
            plot_character_embeddings(
                model,
                test_loader,
                device,
                args.plot_dir,
                args.experiment_no,
                n_samples=1000,
            )

    print(f"Visualization completed for experiment {args.experiment_no}")
