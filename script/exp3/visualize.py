import os
import json
import pickle
import matplotlib.pyplot as plt

import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from config import Config
from train import prepare_data_for_training, generate_past_episodes_from_batch
from data_generation import DataReader
from evaluate import load_model

"""
Visualization tools for KeyDoor ToMnet experiment
Adapted from ToMnetF experiment5 for KeyDoor environment
"""


def plot_accuracy_by_n_past(
    results_by_n_past,
    output_dir=None,
):
    """
    Plot action accuracy as a function of N_past values

    Args:
        results_by_n_past: Dictionary with N_past values as keys and metrics as values
        output_dir: Directory to save plots
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
    ax1.set_title("KeyDoor: Action Accuracy vs N_past", fontsize=14, fontweight="bold")
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
    ax2.set_title("KeyDoor: F1 Score vs N_past", fontsize=14, fontweight="bold")
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
            os.path.join(output_dir, "keydoor_accuracy_by_n_past.png"),
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()

    # Print summary statistics
    print("\nKeyDoor Accuracy by N_past Summary:")
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
    # KeyDoor has 7 actions: left, right, forward, pick_up, drop, toggle, done
    action_names = ["Left", "Right", "Forward", "Pick_up", "Drop", "Toggle", "Done"]

    accuracy_matrix = []

    for n_past in n_past_values:
        predictions = results_by_n_past[n_past]["predictions"]
        targets = results_by_n_past[n_past]["targets"]

        # Calculate per-action accuracy
        action_accuracies = []
        for action in range(7):  # 7 actions in KeyDoor
            action_mask = np.array(targets) == action
            if np.sum(action_mask) > 0:
                action_acc = np.mean(np.array(predictions)[action_mask] == action)
                action_accuracies.append(action_acc)
            else:
                action_accuracies.append(0.0)

        accuracy_matrix.append(action_accuracies)

    accuracy_matrix = np.array(accuracy_matrix)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 6))

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

    ax.set_title(
        "KeyDoor: Per-Action Accuracy by N_past", fontsize=14, fontweight="bold"
    )
    ax.set_xlabel("Action Type", fontsize=12)
    ax.set_ylabel("Number of Past Episodes", fontsize=12)

    plt.tight_layout()

    # Save plot
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(
            os.path.join(output_dir, "keydoor_accuracy_heatmap_by_n_past.png"),
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()


def plot_training_curves(history_path, output_dir, experiment_no=3):
    """
    Plot training curves from training history

    Args:
        history_path: Path to training history JSON file
        output_dir: Directory to save plots
        experiment_no: Experiment number
    """
    plt.style.use("seaborn-v0_8")

    # Load training history
    if not os.path.exists(history_path):
        print(f"Training history not found at: {history_path}")
        return

    with open(history_path, "r") as f:
        history = json.load(f)

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        f"KeyDoor ToMnet Training History (Experiment {experiment_no})", fontsize=16
    )

    # Loss curves
    axes[0, 0].plot(
        history["epoch"], history["train_loss"], label="Train Loss", marker="o"
    )
    axes[0, 0].plot(history["epoch"], history["val_loss"], label="Val Loss", marker="s")
    axes[0, 0].set_title("Total Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Action accuracy curves
    axes[0, 1].plot(
        history["epoch"],
        history["train_action_accuracy"],
        label="Train Action Acc",
        marker="o",
    )
    axes[0, 1].plot(
        history["epoch"],
        history["val_action_accuracy"],
        label="Val Action Acc",
        marker="s",
    )
    axes[0, 1].set_title("Action Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Goal accuracy curves
    axes[1, 0].plot(
        history["epoch"],
        history["train_goal_accuracy"],
        label="Train Goal Acc",
        marker="o",
    )
    axes[1, 0].plot(
        history["epoch"], history["val_goal_accuracy"], label="Val Goal Acc", marker="s"
    )
    axes[1, 0].set_title("Goal Accuracy")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Loss components
    axes[1, 1].plot(
        history["epoch"],
        history["train_action_loss"],
        label="Train Action Loss",
        marker="o",
    )
    axes[1, 1].plot(
        history["epoch"],
        history["train_goal_loss"],
        label="Train Goal Loss",
        marker="s",
    )
    axes[1, 1].plot(
        history["epoch"],
        history["val_action_loss"],
        label="Val Action Loss",
        marker="^",
    )
    axes[1, 1].plot(
        history["epoch"], history["val_goal_loss"], label="Val Goal Loss", marker="v"
    )
    axes[1, 1].set_title("Loss Components")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(
            os.path.join(output_dir, f"keydoor_training_curves_exp{experiment_no}.png"),
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()

    # Print training summary
    print(f"\nKeyDoor Training Summary (Experiment {experiment_no}):")
    print("-" * 50)
    print(f"Total epochs: {len(history['epoch'])}")
    print(f"Best validation loss: {min(history['val_loss']):.4f}")
    print(f"Best validation action accuracy: {max(history['val_action_accuracy']):.4f}")
    print(f"Best validation goal accuracy: {max(history['val_goal_accuracy']):.4f}")


def plot_confusion_matrix(predictions_path, output_dir, experiment_no=3):
    """
    Plot confusion matrix from predictions

    Args:
        predictions_path: Path to predictions pickle file
        output_dir: Directory to save plots
        experiment_no: Experiment number
    """
    plt.style.use("seaborn-v0_8")

    # Load predictions
    if not os.path.exists(predictions_path):
        print(f"Predictions not found at: {predictions_path}")
        return

    with open(predictions_path, "rb") as f:
        predictions_data = pickle.load(f)

    targets = np.array(predictions_data["targets"])
    predictions = np.array(predictions_data["predictions"])

    # Create confusion matrix
    cm = confusion_matrix(targets, predictions)

    # KeyDoor action names
    action_names = ["Left", "Right", "Forward", "Pick_up", "Drop", "Toggle", "Done"]

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=action_names,
        yticklabels=action_names,
        ax=ax,
    )

    ax.set_title(
        f"KeyDoor: Confusion Matrix (Experiment {experiment_no})",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Predicted Action", fontsize=12)
    ax.set_ylabel("True Action", fontsize=12)

    plt.tight_layout()

    # Save plot
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(
            os.path.join(
                output_dir, f"keydoor_confusion_matrix_exp{experiment_no}.png"
            ),
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()

    # Print confusion matrix statistics
    print(f"\nKeyDoor Confusion Matrix Statistics (Experiment {experiment_no}):")
    print("-" * 60)
    for i, action in enumerate(action_names):
        if i < len(cm):
            precision = cm[i, i] / cm[:, i].sum() if cm[:, i].sum() > 0 else 0
            recall = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
            print(f"{action:8s}: Precision={precision:.3f}, Recall={recall:.3f}")


def plot_action_likelihood(predictions_path, output_dir, experiment_no=3):
    """
    Plot action likelihood distributions

    Args:
        predictions_path: Path to predictions pickle file
        output_dir: Directory to save plots
        experiment_no: Experiment number
    """
    plt.style.use("seaborn-v0_8")

    # Load predictions
    if not os.path.exists(predictions_path):
        print(f"Predictions not found at: {predictions_path}")
        return

    with open(predictions_path, "rb") as f:
        predictions_data = pickle.load(f)

    targets = np.array(predictions_data["targets"])
    probabilities = np.array(predictions_data["probabilities"])

    # KeyDoor action names
    action_names = ["Left", "Right", "Forward", "Pick_up", "Drop", "Toggle", "Done"]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(
        f"KeyDoor: Action Likelihood Distributions (Experiment {experiment_no})",
        fontsize=16,
    )

    for action in range(7):  # 7 actions in KeyDoor
        row = action // 4
        col = action % 4
        ax = axes[row, col]

        # Get likelihood values for this action
        action_mask = targets == action
        if np.sum(action_mask) > 0:
            likelihoods = probabilities[action_mask, action]

            # Plot histogram
            ax.hist(
                likelihoods, bins=30, alpha=0.7, color=f"C{action}", edgecolor="black"
            )
            ax.set_title(f"{action_names[action]}")
            ax.set_xlabel("Likelihood")
            ax.set_ylabel("Frequency")
            ax.grid(True, alpha=0.3)

            # Add statistics
            mean_likelihood = np.mean(likelihoods)
            ax.axvline(
                mean_likelihood,
                color="red",
                linestyle="--",
                label=f"Mean: {mean_likelihood:.3f}",
            )
            ax.legend()
        else:
            ax.text(
                0.5, 0.5, "No samples", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_title(f"{action_names[action]}")

    # Remove empty subplot
    if len(axes.flat) > 7:
        axes.flat[7].remove()

    plt.tight_layout()

    # Save plot
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(
            os.path.join(
                output_dir, f"keydoor_action_likelihood_exp{experiment_no}.png"
            ),
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()


def plot_character_embeddings(
    model, test_loader, device, output_dir, experiment_no=3, n_samples=1000
):
    """
    Plot character embeddings using PCA and t-SNE

    Args:
        model: Trained ToMnet model
        test_loader: Test data loader
        device: Computing device
        output_dir: Directory to save plots
        experiment_no: Experiment number
        n_samples: Number of samples to visualize
    """
    plt.style.use("seaborn-v0_8")

    model.eval()
    embeddings = []
    goals = []

    sample_count = 0
    with torch.no_grad():
        for batch in test_loader:
            if sample_count >= n_samples:
                break

            if len(batch) >= 3:
                trajectories, actions, batch_goals = batch[:3]
                trajectories = trajectories.to(device)
                batch_goals = batch_goals.to(device)

                batch_size = trajectories.size(0)

                # Generate past episodes
                past_episodes = generate_past_episodes_from_batch(
                    trajectories, batch_goals, batch_size, 1, 1, 1
                )

                # Get character embeddings
                char_embeddings = model.char_net(past_episodes)

                embeddings.extend(char_embeddings.cpu().numpy())
                goals.extend(batch_goals.cpu().numpy())

                sample_count += len(batch_goals)

    if len(embeddings) == 0:
        print("No embeddings to visualize")
        return

    embeddings = np.array(embeddings)
    goals = np.array(goals)

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"KeyDoor: Character Embeddings (Experiment {experiment_no})", fontsize=16
    )

    # Goal colors for KeyDoor (4 goals: A, B, C, D)
    goal_colors = ["red", "green", "blue", "yellow"]
    goal_names = ["Goal A (Red)", "Goal B (Green)", "Goal C (Blue)", "Goal D (Yellow)"]

    # PCA visualization
    if embeddings.shape[1] > 2:
        pca = PCA(n_components=2)
        embeddings_pca = pca.fit_transform(embeddings)

        for goal in range(4):
            mask = goals == goal
            if np.sum(mask) > 0:
                ax1.scatter(
                    embeddings_pca[mask, 0],
                    embeddings_pca[mask, 1],
                    c=goal_colors[goal],
                    label=goal_names[goal],
                    alpha=0.6,
                )

        ax1.set_title("PCA Visualization")
        ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
        ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # t-SNE visualization
    if len(embeddings) > 50:  # t-SNE needs sufficient samples
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_tsne = tsne.fit_transform(embeddings[:1000])  # Limit for performance
        goals_tsne = goals[:1000]

        for goal in range(4):
            mask = goals_tsne == goal
            if np.sum(mask) > 0:
                ax2.scatter(
                    embeddings_tsne[mask, 0],
                    embeddings_tsne[mask, 1],
                    c=goal_colors[goal],
                    label=goal_names[goal],
                    alpha=0.6,
                )

        ax2.set_title("t-SNE Visualization")
        ax2.set_xlabel("t-SNE 1")
        ax2.set_ylabel("t-SNE 2")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(
            os.path.join(
                output_dir, f"keydoor_character_embeddings_exp{experiment_no}.png"
            ),
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()

    print(f"\nKeyDoor Character Embeddings Analysis (Experiment {experiment_no}):")
    print("-" * 60)
    print(f"Total samples: {len(embeddings)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")

    # Per-goal statistics
    for goal in range(4):
        mask = goals == goal
        count = np.sum(mask)
        print(f"{goal_names[goal]}: {count} samples")


def create_additional_visualizations(
    model, test_loader, output_dir, experiment_no, device, save_plots=True, config=None
):
    """
    Create additional visualizations for KeyDoor experiment

    Args:
        model: Trained ToMnet model
        test_loader: Test data loader
        output_dir: Directory to save plots
        experiment_no: Experiment number
        device: Computing device
        save_plots: Whether to save plots
        config: Configuration object
    """
    if config is None:
        config = Config()

    print("Creating additional KeyDoor visualizations...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Plot character embeddings
    plot_character_embeddings(
        model,
        test_loader,
        device,
        output_dir,
        experiment_no,
        n_samples=config.evaluation_config.get("n_samples", 1000),
    )

    print("Additional visualizations completed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize KeyDoor ToMnet results")
    parser.add_argument(
        "--config_override",
        action="store_true",
        help="Override config with command line arguments",
    )
    parser.add_argument("--result_dir", type=str, help="Directory containing results")
    parser.add_argument("--plot_dir", type=str, help="Directory to save plots")
    parser.add_argument(
        "--experiment_no", type=int, default=3, help="Experiment number"
    )
    parser.add_argument(
        "--plot_type",
        type=str,
        choices=["training", "confusion", "likelihood", "embeddings", "n_past", "all"],
        default="all",
        help="Type of plot to create",
    )

    args = parser.parse_args()

    config = Config()

    # Override config with command line arguments if specified
    if args.config_override:
        config.update_from_args(args)

    results_dir = args.result_dir or config.result_dir
    plot_dir = args.plot_dir or config.plot_dir
    experiment_no = args.experiment_no or config.experiment_no

    # Create plot directory
    os.makedirs(plot_dir, exist_ok=True)

    print(f"Creating KeyDoor visualizations for experiment {experiment_no}")
    print(f"Results directory: {results_dir}")
    print(f"Plot directory: {plot_dir}")

    # Plot training curves
    if args.plot_type in ["training", "all"]:
        history_files = [
            os.path.join(results_dir, "training_history.json"),
            os.path.join(results_dir, f"exp{experiment_no}_*/training_history.json"),
        ]

        import glob

        for pattern in history_files:
            matching_files = glob.glob(pattern)
            for history_file in matching_files:
                if os.path.exists(history_file):
                    plot_training_curves(history_file, plot_dir, experiment_no)
                    break

    # Plot confusion matrix
    if args.plot_type in ["confusion", "all"]:
        pred_files = [
            os.path.join(results_dir, "predictions.pkl"),
            os.path.join(results_dir, f"exp{experiment_no}_*/predictions.pkl"),
        ]

        import glob

        for pattern in pred_files:
            matching_files = glob.glob(pattern)
            for pred_file in matching_files:
                if os.path.exists(pred_file):
                    plot_confusion_matrix(pred_file, plot_dir, experiment_no)
                    break

    # Plot action likelihood
    if args.plot_type in ["likelihood", "all"]:
        pred_files = [
            os.path.join(results_dir, "predictions.pkl"),
            os.path.join(results_dir, f"exp{experiment_no}_*/predictions.pkl"),
        ]

        import glob

        for pattern in pred_files:
            matching_files = glob.glob(pattern)
            for pred_file in matching_files:
                if os.path.exists(pred_file):
                    plot_action_likelihood(pred_file, plot_dir, experiment_no)
                    break

    # Plot N_past results
    if args.plot_type in ["n_past", "all"]:
        n_past_file = os.path.join(results_dir, "n_past_evaluation_results.json")
        if os.path.exists(n_past_file):
            with open(n_past_file, "r") as f:
                n_past_results = json.load(f)

            # Convert string keys back to integers
            results_by_n_past = {}
            for key, value in n_past_results.items():
                results_by_n_past[int(key)] = value

            plot_accuracy_by_n_past(results_by_n_past, plot_dir)
            plot_accuracy_heatmap_by_n_past(results_by_n_past, plot_dir)

    print("Visualization completed!")
