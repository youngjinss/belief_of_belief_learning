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

"""
Publication-quality visualization for ToMnetF (Experiment 2)
Extended with SR and consumption visualization
@Author Filip Borowiak
"""

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
    Plot training and validation curves

    Args:
        history_path: Path to training history JSON
        output_dir: Output directory for plots
        experiment_no: Experiment number
    """

    with open(history_path, "r") as f:
        history = json.load(f)

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

    # Loss plot
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
    ax2.set_title("Model Loss", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max(epochs))

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

    model.eval()
    embeddings = []
    actions = []

    sample_count = 0
    with torch.no_grad():
        for traj, curr, act in data_loader:
            if sample_count >= n_samples:
                break

            traj, curr = traj.to(device), curr.to(device)
            act = act.squeeze(-1).type(torch.long)

            # Get character embeddings
            e_char = model.char_net(traj)

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


def create_summary_report(
    experiment_no,
    result_dir="../../result/experiment2",
    plot_dir="../../plots/experiment2",
):
    """
    Create a summary report with all visualizations

    Args:
        experiment_no: Experiment number
        result_dir: Results directory
        plot_dir: Plots directory
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
        "--experiment_no", type=int, default=1, help="Experiment number"
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="../../result/experiment2",
        help="Results directory",
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default="../../plots/experiment2",
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
    elif args.plot_type == "training" and args.history_path:
        plot_training_curves(args.history_path, args.plot_dir, args.experiment_no)
    elif args.plot_type == "confusion" and args.predictions_path:
        plot_confusion_matrix(args.predictions_path, args.plot_dir, args.experiment_no)
    elif args.plot_type == "likelihood" and args.predictions_path:
        plot_action_likelihood(args.predictions_path, args.plot_dir, args.experiment_no)
    elif args.plot_type == "cross_species" and args.cross_species_path:
        plot_cross_species_results(
            args.cross_species_path, args.plot_dir, args.experiment_no
        )
    else:
        print(f"Invalid combination of plot_type and required paths")
        print(f"For {args.plot_type}, required paths were not provided or don't exist")

    print(f"Visualization completed for experiment {args.experiment_no}")
