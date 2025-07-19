import os
import json
import pickle
import warnings

# Set matplotlib backend before importing pyplot to avoid display issues
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt

import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Suppress sklearn and matplotlib warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from config import Config
from utils import prepare_data_for_training, generate_past_episodes_from_batch
from data_generation import DataGenerator as DataReader, DataGenerator
from utils import set_seed, load_chunked_data_for_training, load_test_data_all_combinations, combine_all_combinations_data

# Set seed using Config default value
config = Config()
set_seed(config.seed)

# Remove circular import - load_model will be imported locally when needed

"""
Visualization tools for AchieverBlocker ToMnet experiment
Adapted from ToMnetF experiment5 for multi-agent AchieverBlocker environment
"""

# No caching - keep it simple and reliable


# Data loading functions moved to utils.py


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
    title_prefix = "Single-Agent" if config and config.is_single_agent_mode() else "AchieverBlocker"
    ax1.set_title(
        f"{title_prefix}: Action Accuracy vs N_past", fontsize=14, fontweight="bold"
    )
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
    title_prefix = "Single-Agent" if config and config.is_single_agent_mode() else "AchieverBlocker"
    ax2.set_title(f"{title_prefix}: F1 Score vs N_past", fontsize=14, fontweight="bold")
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
            os.path.join(output_dir, "achieverblocker_accuracy_by_n_past.png"),
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()

    # Print summary statistics
    title_prefix = "Single-Agent" if config and config.is_single_agent_mode() else "AchieverBlocker"
    print(f"\n{title_prefix} Accuracy by N_past Summary:")
    print("-" * 40)
    for n in n_past_values:
        print(
            f"N_past={n:2d}: Accuracy={accuracies[n_past_values.index(n)]:.4f}, "
            f"F1={f1_scores[n_past_values.index(n)]:.4f}"
        )

    best_n_past = n_past_values[np.argmax(accuracies)]
    print(f"\nBest N_past: {best_n_past} (Accuracy: {max(accuracies):.4f})")


def plot_accuracy_heatmap_by_n_past(results_by_n_past, output_dir=None, config=None):
    """
    Create a heatmap showing per-action accuracy by N_past values

    Args:
        results_by_n_past: Dictionary with N_past values as keys and metrics as values
        output_dir: Directory to save plots
        config: Config object containing action configuration
    """
    plt.style.use("seaborn-v0_8")

    # Extract data and calculate per-action accuracy
    n_past_values = sorted(results_by_n_past.keys())

    # Get action information from config or use defaults
    if config is not None:
        action_config = config.get_action_config()
        num_actions = action_config.get("num_actions", 7)
        action_names = action_config.get(
            "action_names",
            ["Up", "Right", "Down", "Left", "Stay", "Pickup", "Toggle"][:num_actions],
        )
    else:
        # Use default values if config is not provided (AchieverBlocker has mixed actions)
        num_actions = 8  # Combined achiever (0-6) and blocker (0-5) actions
        action_names = [
            "Up",
            "Right",
            "Down",
            "Left",
            "Stay",
            "Pickup",
            "Toggle",
            "Broken",
        ][:num_actions]

    accuracy_matrix = []

    for n_past in n_past_values:
        # Check if detailed predictions and targets are available
        if (
            "predictions" in results_by_n_past[n_past]
            and "targets" in results_by_n_past[n_past]
        ):
            predictions = results_by_n_past[n_past]["predictions"]
            targets = results_by_n_past[n_past]["targets"]

            # Calculate per-action accuracy
            action_accuracies = []
            for action in range(num_actions):
                action_mask = np.array(targets) == action
                if np.sum(action_mask) > 0:
                    action_acc = np.mean(np.array(predictions)[action_mask] == action)
                    action_accuracies.append(action_acc)
                else:
                    action_accuracies.append(0.0)

            accuracy_matrix.append(action_accuracies)
        else:
            # If detailed predictions not available, use overall accuracy for all actions
            overall_accuracy = results_by_n_past[n_past]["accuracy"]
            # Use the same accuracy for all actions (not ideal but prevents crash)
            action_accuracies = [
                overall_accuracy / num_actions
            ] * num_actions  # Distribute equally
            accuracy_matrix.append(action_accuracies)

            # Only print warning once
            if n_past == n_past_values[0]:
                print(
                    "Warning: N_past results don't contain detailed predictions/targets."
                )
                print(
                    "Using overall accuracy distributed across all actions for heatmap."
                )

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

    title_prefix = "Single-Agent" if config and config.is_single_agent_mode() else "AchieverBlocker"
    ax.set_title(
        f"{title_prefix}: Per-Action Accuracy by N_past", fontsize=14, fontweight="bold"
    )
    ax.set_xlabel("Action Type", fontsize=12)
    ax.set_ylabel("Number of Past Episodes", fontsize=12)

    plt.tight_layout()

    # Save plot
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(
            os.path.join(output_dir, "achieverblocker_accuracy_heatmap_by_n_past.png"),
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()


def plot_training_curves(history_path, output_dir, config=None, experiment_no=None):
    """
    Plot training curves from training history

    Args:
        history_path: Path to training history JSON file
        output_dir: Directory to save plots
        config: Configuration object containing experiment settings
        experiment_no: Experiment number (defaults to config.experiment_no)
    """
    if config is None:
        config = Config()

    if experiment_no is None:
        experiment_no = config.experiment_no
    plt.style.use("seaborn-v0_8")

    # Load training history
    if not os.path.exists(history_path):
        print(f"Training history not found at: {history_path}")
        return

    with open(history_path, "r") as f:
        history = json.load(f)

    # Check if component losses are available
    has_component_losses = (
        "train_action_loss" in history
        and "train_consumption_loss" in history
        and "train_sr_loss" in history
    )

    if has_component_losses:
        # Create 3x2 subplot grid for comprehensive visualization (matching experiment 5)
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))
        ax1, ax2 = axes[0]
        ax3, ax4 = axes[1]
        ax5, ax6 = axes[2]
    else:
        # Fallback to 2x2 layout for basic metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        ax1, ax2 = axes[0]
        ax3, ax4 = axes[1]

    title_prefix = "Single-Agent" if config and config.is_single_agent_mode() else "AchieverBlocker"
    fig.suptitle(
        f"{title_prefix} ToMnet Training History (Experiment {experiment_no})",
        fontsize=16,
    )

    epochs = history["epoch"]

    # Total accuracy plot
    ax1.plot(
        epochs,
        history["train_action_accuracy"],
        label="Training",
        linewidth=2,
        marker="o",
        markersize=4,
    )
    ax1.plot(
        epochs,
        history["val_action_accuracy"],
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

        # Goal accuracy plot (moved to 6th position)
        ax6.plot(
            epochs,
            history["train_goal_accuracy"],
            label="Training",
            linewidth=2,
            marker="o",
            markersize=4,
            color="orange",
        )
        ax6.plot(
            epochs,
            history["val_goal_accuracy"],
            label="Validation",
            linewidth=2,
            marker="s",
            markersize=4,
            color="moccasin",
        )
        ax6.set_xlabel("Epoch", fontsize=12)
        ax6.set_ylabel("Accuracy", fontsize=12)
        ax6.set_title("Goal Accuracy", fontsize=14, fontweight="bold")
        ax6.legend(fontsize=11)
        ax6.grid(True, alpha=0.3)
        ax6.set_xlim(0, max(epochs))
    else:
        # Fallback to basic goal accuracy and loss components
        # Goal accuracy curves
        ax3.plot(
            history["epoch"],
            history["train_goal_accuracy"],
            label="Train Goal Acc",
            marker="o",
        )
        ax3.plot(
            history["epoch"],
            history["val_goal_accuracy"],
            label="Val Goal Acc",
            marker="s",
        )
        ax3.set_title("Goal Accuracy")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Accuracy")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Loss components
        ax4.plot(
            history["epoch"],
            history["train_action_loss"] if "train_action_loss" in history else [],
            label="Train Action Loss",
            marker="o",
        )
        ax4.plot(
            history["epoch"],
            history["train_goal_loss"] if "train_goal_loss" in history else [],
            label="Train Goal Loss",
            marker="s",
        )
        ax4.plot(
            history["epoch"],
            history["val_action_loss"] if "val_action_loss" in history else [],
            label="Val Action Loss",
            marker="^",
        )
        ax4.plot(
            history["epoch"],
            history["val_goal_loss"] if "val_goal_loss" in history else [],
            label="Val Goal Loss",
            marker="v",
        )
        ax4.set_title("Loss Components")
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Loss")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(
            os.path.join(
                output_dir, f"achieverblocker_training_curves_exp{experiment_no}.png"
            ),
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()

    # Print training summary
    title_prefix = "Single-Agent" if config and config.is_single_agent_mode() else "AchieverBlocker"
    print(f"\n{title_prefix} Training Summary (Experiment {experiment_no}):")
    print("-" * 50)
    print(f"Total epochs: {len(history['epoch'])}")
    print(f"Best validation loss: {min(history['val_loss']):.4f}")
    print(f"Best validation action accuracy: {max(history['val_action_accuracy']):.4f}")
    print(f"Best validation goal accuracy: {max(history['val_goal_accuracy']):.4f}")

    # Print component loss summaries if available
    if has_component_losses:
        if "val_consumption_loss" in history:
            print(
                f"Best validation consumption loss: {min(history['val_consumption_loss']):.4f}"
            )
        if "val_sr_loss" in history:
            print(f"Best validation SR loss: {min(history['val_sr_loss']):.4f}")


def plot_confusion_matrix(
    predictions_path, output_dir, config=None, experiment_no=None
):
    """
    Plot confusion matrix from predictions

    Args:
        predictions_path: Path to predictions pickle file
        output_dir: Directory to save plots
        config: Configuration object containing experiment settings
        experiment_no: Experiment number (defaults to config.experiment_no)
    """
    if config is None:
        config = Config()

    if experiment_no is None:
        experiment_no = config.experiment_no

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

    # Get action information from config
    action_config = config.get_action_config()
    action_names = action_config.get(
        "action_names",
        ["Up", "Right", "Down", "Left", "Stay", "Pickup", "Toggle", "Broken"],
    )

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

    title_prefix = "Single-Agent" if config and config.is_single_agent_mode() else "AchieverBlocker"
    ax.set_title(
        f"{title_prefix}: Confusion Matrix (Experiment {experiment_no})",
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
                output_dir, f"achieverblocker_confusion_matrix_exp{experiment_no}.png"
            ),
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()

    # Print confusion matrix statistics
    title_prefix = "Single-Agent" if config and config.is_single_agent_mode() else "AchieverBlocker"
    print(
        f"\n{title_prefix} Confusion Matrix Statistics (Experiment {experiment_no}):"
    )
    print("-" * 60)
    for i, action in enumerate(action_names):
        if i < len(cm):
            precision = cm[i, i] / cm[:, i].sum() if cm[:, i].sum() > 0 else 0
            recall = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
            print(f"{action:8s}: Precision={precision:.3f}, Recall={recall:.3f}")


def plot_action_likelihood(
    predictions_path, output_dir, config=None, experiment_no=None
):
    """
    Plot action likelihood distributions

    Args:
        predictions_path: Path to predictions pickle file
        output_dir: Directory to save plots
        config: Configuration object containing experiment settings
        experiment_no: Experiment number (defaults to config.experiment_no)
    """
    if config is None:
        config = Config()

    if experiment_no is None:
        experiment_no = config.experiment_no

    plt.style.use("seaborn-v0_8")

    # Load predictions
    if not os.path.exists(predictions_path):
        print(f"Predictions not found at: {predictions_path}")
        return

    with open(predictions_path, "rb") as f:
        predictions_data = pickle.load(f)

    targets = np.array(predictions_data["targets"])
    probabilities = np.array(predictions_data["probabilities"])

    # Get action information from config
    action_config = config.get_action_config()
    action_names = action_config.get(
        "action_names",
        ["Up", "Right", "Down", "Left", "Stay", "Pickup", "Toggle", "Broken"],
    )

    # Create figure with subplots
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    title_prefix = "Single-Agent" if config and config.is_single_agent_mode() else "AchieverBlocker"
    fig.suptitle(
        f"{title_prefix}: Action Likelihood Distributions (Experiment {experiment_no})",
        fontsize=16,
    )

    num_actions = len(action_names)
    for action in range(num_actions):
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
                output_dir, f"achieverblocker_action_likelihood_exp{experiment_no}.png"
            ),
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()


def plot_character_embeddings(
    model,
    test_loader,
    device,
    output_dir,
    config=None,
    experiment_no=None,
    n_samples=None,
):
    """
    Plot character embeddings using PCA and t-SNE with separate agent analysis
    Creates three types of plots:
    1. Agent-based coloring (achiever vs blocker)
    2. Goal-based coloring (red, green, blue, yellow)
    3. Separate plots for achiever goals and blocker goals

    Args:
        model: Trained ToMnet model
        test_loader: Test data loader or can be None to load from config
        device: Computing device
        output_dir: Directory to save plots
        config: Configuration object containing experiment settings
        experiment_no: Experiment number (defaults to config.experiment_no)
        n_samples: Number of samples to visualize (None for all samples)
    """
    if config is None:
        config = Config()

    if experiment_no is None:
        experiment_no = config.experiment_no

    # Get visualization config
    vis_config = config.get_visualization_config()
    agent_colors = vis_config["agent_colors"]
    agent_names = vis_config["agent_names"]
    goal_colors = vis_config["goal_colors"]
    goal_names = vis_config["goal_names"]
    goal_letters = vis_config["goal_letters"]
    embedding_plots = vis_config["embedding_plots"]

    plt.style.use("seaborn-v0_8")

    # Load processed test data from all combinations to get agent information
    print("Loading processed test data from all combinations to extract agent and goal information...")

    # Generate base path from config based on single-agent vs multi-agent mode
    if config.is_single_agent_mode():
        # For single-agent mode, use the first achiever type's test directory
        achiever_type = list(config.achiever_types.keys())[0] 
        test_data_dir = os.path.dirname(config.get_training_data_path(achiever_type, None, is_test=True))
    else:
        # For multi-agent mode, use environment base path
        env_name = config.get_env_name()
        test_data_dir = f"./data/{env_name}"

    # Load test data for all combinations efficiently
    all_test_data = load_test_data_all_combinations(config, test_data_dir_base=test_data_dir)
    
    # Combine data from all combinations
    processed_data = combine_all_combinations_data(all_test_data)
    
    print(f"Loaded combined test data from all combinations with {processed_data['trajectories'].shape[0]} samples")
    print(f"Achiever types: {list(config.achiever_types.keys())}")
    if config.is_single_agent_mode():
        print(f"Single-agent mode: No blockers")
    else:
        print(f"Blocker types: {list(config.blocker_types.keys())}")
    
    # Extract agent labels, goal labels, and types from processed tensors
    # agents tensor: 0=achiever, 1=blocker
    agent_indices = processed_data["agents"]
    agent_labels = np.array(
        ["achiever" if idx == 0 else "blocker" for idx in agent_indices]
    )

    # goals tensor: one-hot encoded [A, B, C, D]
    goals_tensor = processed_data["goals"]
    goal_labels = np.argmax(goals_tensor, axis=1)  # Convert one-hot to indices

    # types tensor: 0 for achievers (always), 0 or 1 for blockers (0=randomly select, 1=rule-based)
    types_tensor = processed_data["types"]
    type_labels = types_tensor.astype(int)

    print(f"Agent distribution: {np.unique(agent_labels, return_counts=True)}")
    print(f"Goal distribution: {np.unique(goal_labels, return_counts=True)}")

    # Extract character embeddings using the model
    model.eval()
    embeddings = []

    # Get the data tensors - handle both chunked and original formats
    if isinstance(processed_data["trajectories"], torch.Tensor):
        # Original tensor format
        trajectories_tensor = processed_data["trajectories"]
        goals_tensor = processed_data["goals"]
        goal_ranks_tensor = processed_data["goal_ranks"]
        agents_tensor = processed_data["agents"]
        types_tensor_full = processed_data["types"]
    else:
        # Chunked format (numpy arrays)
        trajectories_tensor = torch.from_numpy(processed_data["trajectories"]).float()
        goals_tensor = torch.from_numpy(processed_data["goals"]).float()
        goal_ranks_tensor = torch.from_numpy(processed_data["goal_ranks"]).long()
        agents_tensor = torch.from_numpy(processed_data["agents"]).long()
        types_tensor_full = torch.from_numpy(processed_data["types"]).long()

    if n_samples is not None:
        # Limit samples if specified
        indices = np.random.choice(
            len(agent_labels), min(n_samples, len(agent_labels)), replace=False
        )
        trajectories_tensor = trajectories_tensor[indices]
        goals_tensor = goals_tensor[indices]
        goal_ranks_tensor = goal_ranks_tensor[indices]
        agents_tensor = agents_tensor[indices]
        types_tensor_full = types_tensor_full[indices]
        agent_labels = agent_labels[indices]
        goal_labels = goal_labels[indices]
        type_labels = type_labels[indices]

    print(f"Extracting character embeddings for {len(agent_labels)} samples...")

    # Optimized batch processing for character embeddings
    batch_size = 32  # Process in batches for better GPU utilization
    n_past_config = config.get_n_past_evaluation_config()

    with torch.no_grad():
        for start_idx in range(0, len(agent_labels), batch_size):
            end_idx = min(start_idx + batch_size, len(agent_labels))

            if start_idx % (batch_size * 10) == 0:
                print(
                    f"Processing batch {start_idx//batch_size + 1}/{(len(agent_labels) + batch_size - 1)//batch_size}"
                )

            try:
                # Get batch tensors
                batch_trajectories = trajectories_tensor[start_idx:end_idx].to(device)
                batch_goal_ranks = goal_ranks_tensor[start_idx:end_idx].to(device)
                batch_agents = agents_tensor[start_idx:end_idx].to(device)
                current_batch_size = end_idx - start_idx

                # Generate past episodes for this batch
                past_episodes = generate_past_episodes_from_batch(
                    trajectories=batch_trajectories,
                    goal_ranks=batch_goal_ranks,
                    agents=batch_agents,
                    batch_size=current_batch_size,
                    n_past_min=n_past_config["n_past_min"],
                    n_past_max=n_past_config["n_past_max"],
                    max_n_past=n_past_config["n_past_max"],
                    rank_threshold=config.get_data_config().get("rank_threshold", 4),
                )

                # Extract character embeddings for the entire batch
                char_embeddings = model.get_character_embedding(past_episodes)

                # Add batch embeddings to list
                for emb in char_embeddings.cpu().numpy():
                    embeddings.append(emb.flatten())

            except Exception as e:
                print(f"Error processing batch starting at {start_idx}: {e}")
                # Add zero embeddings as placeholders for the entire batch
                embedding_dim = 64  # Default embedding dimension
                for _ in range(current_batch_size):
                    embeddings.append(np.zeros(embedding_dim))
                continue

    embeddings = np.array(embeddings)
    print(f"Extracted embeddings shape: {embeddings.shape}")

    if len(embeddings) == 0:
        print("No embeddings to visualize!")
        return

    # Create plots based on mode
    if config.is_single_agent_mode():
        # Single-agent mode: only create goal-based and achiever type embeddings
        _plot_goal_based_embeddings(
            embeddings, agent_labels, goal_labels, config, output_dir, experiment_no
        )
        _plot_type_based_embeddings_for_achiever(
            embeddings,
            agent_labels,
            goal_labels, 
            type_labels,
            config,
            output_dir,
            experiment_no,
        )
    else:
        # Multi-agent mode: create all types of plots
        _plot_agent_based_embeddings(
            embeddings, agent_labels, goal_labels, config, output_dir, experiment_no
        )
        _plot_goal_based_embeddings(
            embeddings, agent_labels, goal_labels, config, output_dir, experiment_no
        )
        _plot_separate_agent_goal_embeddings(
            embeddings, agent_labels, goal_labels, config, output_dir, experiment_no
        )
        _plot_type_based_embeddings_for_blockers(
            embeddings,
            agent_labels,
            goal_labels,
            type_labels,
            config,
            output_dir,
            experiment_no,
        )
        _plot_type_based_embeddings_for_achiever(
            embeddings,
            agent_labels,
            goal_labels,
            type_labels,
            config,
            output_dir,
            experiment_no,
        )


def _plot_agent_based_embeddings(
    embeddings, agent_labels, goal_labels, config, output_dir, experiment_no
):
    """Plot embeddings colored by agent type (achiever vs blocker)"""
    vis_config = config.get_visualization_config()
    agent_colors = vis_config["agent_colors"]
    agent_names = vis_config["agent_names"]
    embedding_plots = vis_config["embedding_plots"]

    print("\nCreating agent-based embedding plots...")

    # Create figure with PCA and t-SNE subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=embedding_plots["pca_figsize"])
    fig.suptitle(
        f"Character Embeddings by Agent Type (Experiment {experiment_no})", fontsize=16
    )

    # PCA visualization
    if embeddings.shape[1] > 2:
        print("Computing PCA...")
        pca = PCA(n_components=2)
        embeddings_pca = pca.fit_transform(embeddings)

        unique_agents = np.unique(agent_labels)
        for i, agent in enumerate(unique_agents):
            mask = agent_labels == agent
            agent_count = np.sum(mask)
            if agent_count > 0:
                color = agent_colors[i] if i < len(agent_colors) else f"C{i}"
                name = agent_names[i] if i < len(agent_names) else agent
                ax1.scatter(
                    embeddings_pca[mask, 0],
                    embeddings_pca[mask, 1],
                    c=color,
                    label=f"{name} (n={agent_count})",
                    alpha=embedding_plots["alpha"],
                    s=embedding_plots["marker_size"],
                )

        ax1.set_title(f"PCA by Agent Type")
        ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
        ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # t-SNE visualization with optimized parameters
    print("Computing t-SNE (this may take a while)...")
    # Use faster t-SNE parameters for better performance
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=min(30, len(embeddings) // 4),  # Adaptive perplexity
        n_iter=300,  # Reduced iterations for speed
        early_exaggeration=12,
        learning_rate="auto",
    )
    embeddings_tsne = tsne.fit_transform(embeddings)

    unique_agents = np.unique(agent_labels)
    for i, agent in enumerate(unique_agents):
        mask = agent_labels == agent
        agent_count = np.sum(mask)
        if agent_count > 0:
            color = agent_colors[i] if i < len(agent_colors) else f"C{i}"
            name = agent_names[i] if i < len(agent_names) else agent
            ax2.scatter(
                embeddings_tsne[mask, 0],
                embeddings_tsne[mask, 1],
                c=color,
                label=f"{name} (n={agent_count})",
                alpha=embedding_plots["alpha"],
                s=embedding_plots["marker_size"],
            )

    ax2.set_title(f"t-SNE by Agent Type")
    ax2.set_xlabel("t-SNE 1")
    ax2.set_ylabel("t-SNE 2")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(
        os.path.join(
            output_dir, f"character_embeddings_by_agent_exp{experiment_no}.png"
        ),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def _plot_goal_based_embeddings(
    embeddings, agent_labels, goal_labels, config, output_dir, experiment_no
):
    """Plot embeddings colored by goal type (red, green, blue, yellow)"""
    vis_config = config.get_visualization_config()
    goal_colors = vis_config["goal_colors"]
    goal_names = vis_config["goal_names"]
    embedding_plots = vis_config["embedding_plots"]

    print("\nCreating goal-based embedding plots...")

    # Create figure with PCA and t-SNE subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=embedding_plots["pca_figsize"])
    fig.suptitle(
        f"Character Embeddings by Goal Type (Experiment {experiment_no})", fontsize=16
    )

    # PCA visualization
    if embeddings.shape[1] > 2:
        print("Computing PCA...")
        pca = PCA(n_components=2)
        embeddings_pca = pca.fit_transform(embeddings)

        unique_goals = np.unique(goal_labels)
        for goal in unique_goals:
            mask = goal_labels == goal
            goal_count = np.sum(mask)
            if goal_count > 0:
                color = goal_colors[goal] if goal < len(goal_colors) else f"C{goal}"
                name = goal_names[goal] if goal < len(goal_names) else f"Goal {goal}"
                ax1.scatter(
                    embeddings_pca[mask, 0],
                    embeddings_pca[mask, 1],
                    c=color,
                    label=f"{name} (n={goal_count})",
                    alpha=embedding_plots["alpha"],
                    s=embedding_plots["marker_size"],
                )

        ax1.set_title(f"PCA by Goal Type")
        ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
        ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # t-SNE visualization with optimized parameters
    print("Computing t-SNE (this may take a while)...")
    # Use faster t-SNE parameters for better performance
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=min(30, len(embeddings) // 4),  # Adaptive perplexity
        n_iter=300,  # Reduced iterations for speed
        early_exaggeration=12,
        learning_rate="auto",
    )
    embeddings_tsne = tsne.fit_transform(embeddings)

    unique_goals = np.unique(goal_labels)
    for goal in unique_goals:
        mask = goal_labels == goal
        goal_count = np.sum(mask)
        if goal_count > 0:
            color = goal_colors[goal] if goal < len(goal_colors) else f"C{goal}"
            name = goal_names[goal] if goal < len(goal_names) else f"Goal {goal}"
            ax2.scatter(
                embeddings_tsne[mask, 0],
                embeddings_tsne[mask, 1],
                c=color,
                label=f"{name} (n={goal_count})",
                alpha=embedding_plots["alpha"],
                s=embedding_plots["marker_size"],
            )

    ax2.set_title(f"t-SNE by Goal Type")
    ax2.set_xlabel("t-SNE 1")
    ax2.set_ylabel("t-SNE 2")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(
        os.path.join(
            output_dir, f"character_embeddings_by_goal_exp{experiment_no}.png"
        ),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def _plot_separate_agent_goal_embeddings(
    embeddings, agent_labels, goal_labels, config, output_dir, experiment_no
):
    """Plot separate embeddings for achiever goals and blocker goals"""
    vis_config = config.get_visualization_config()
    goal_colors = vis_config["goal_colors"]
    goal_names = vis_config["goal_names"]
    embedding_plots = vis_config["embedding_plots"]

    print("\nCreating separate agent-goal embedding plots...")

    # Create figure with 2x2 subplots (PCA and t-SNE for each agent)
    fig, axes = plt.subplots(2, 2, figsize=embedding_plots["combined_figsize"])
    fig.suptitle(
        f"Character Embeddings: Achiever vs Blocker Goals (Experiment {experiment_no})",
        fontsize=16,
    )

    agents = ["achiever", "blocker"]

    for agent_idx, agent in enumerate(agents):
        agent_mask = agent_labels == agent
        agent_embeddings = embeddings[agent_mask]
        agent_goals = goal_labels[agent_mask]

        if len(agent_embeddings) == 0:
            print(f"No embeddings found for {agent}")
            continue

        print(f"Processing {agent}: {len(agent_embeddings)} embeddings")

        # PCA for this agent
        ax_pca = axes[agent_idx, 0]
        if agent_embeddings.shape[1] > 2:
            pca = PCA(n_components=2)
            agent_embeddings_pca = pca.fit_transform(agent_embeddings)

            unique_goals = np.unique(agent_goals)
            for goal in unique_goals:
                goal_mask = agent_goals == goal
                goal_count = np.sum(goal_mask)
                if goal_count > 0:
                    color = goal_colors[goal] if goal < len(goal_colors) else f"C{goal}"
                    name = (
                        goal_names[goal] if goal < len(goal_names) else f"Goal {goal}"
                    )
                    ax_pca.scatter(
                        agent_embeddings_pca[goal_mask, 0],
                        agent_embeddings_pca[goal_mask, 1],
                        c=color,
                        label=f"{name} (n={goal_count})",
                        alpha=embedding_plots["alpha"],
                        s=embedding_plots["marker_size"],
                    )

            ax_pca.set_title(f"PCA: {agent.capitalize()} Goals")
            ax_pca.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
            ax_pca.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
            ax_pca.legend(fontsize=8)
            ax_pca.grid(True, alpha=0.3)

        # t-SNE for this agent
        ax_tsne = axes[agent_idx, 1]
        tsne = TSNE(n_components=2, random_state=42)
        agent_embeddings_tsne = tsne.fit_transform(agent_embeddings)

        unique_goals = np.unique(agent_goals)
        for goal in unique_goals:
            goal_mask = agent_goals == goal
            goal_count = np.sum(goal_mask)
            if goal_count > 0:
                color = goal_colors[goal] if goal < len(goal_colors) else f"C{goal}"
                name = (
                    goal_names[goal] if goal < len(goal_names) else f"Goal {goal}"
                )
                ax_tsne.scatter(
                    agent_embeddings_tsne[goal_mask, 0],
                    agent_embeddings_tsne[goal_mask, 1],
                    c=color,
                    label=f"{name} (n={goal_count})",
                    alpha=embedding_plots["alpha"],
                    s=embedding_plots["marker_size"],
                )

        ax_tsne.set_title(f"t-SNE: {agent.capitalize()} Goals")
        ax_tsne.set_xlabel("t-SNE 1")
        ax_tsne.set_ylabel("t-SNE 2")
        ax_tsne.legend(fontsize=8)
        ax_tsne.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(
        os.path.join(
            output_dir, f"character_embeddings_separate_agents_exp{experiment_no}.png"
        ),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"All embedding plots saved to {output_dir}")


def _plot_type_based_embeddings_for_blockers(
    embeddings,
    agent_labels,
    goal_labels,
    type_labels,
    config,
    output_dir,
    experiment_no,
):
    """Plot embeddings colored by Type, constrained to Blocker agents only"""
    vis_config = config.get_visualization_config()
    embedding_plots = vis_config["embedding_plots"]

    print("\nCreating Type-based embedding plots for Blocker agents...")

    # Filter for blocker agents only
    blocker_mask = agent_labels == "blocker"
    blocker_embeddings = embeddings[blocker_mask]
    blocker_types = type_labels[blocker_mask]

    if len(blocker_embeddings) == 0:
        print("No blocker samples found for Type visualization")
        return

    print(f"Found {len(blocker_embeddings)} blocker samples for Type visualization")
    print(f"Type distribution: {np.unique(blocker_types, return_counts=True)}")

    # Create figure with PCA and t-SNE subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=embedding_plots["pca_figsize"])
    fig.suptitle(
        f"Character Embeddings by Blocker Type (Experiment {experiment_no})",
        fontsize=16,
    )

    # Type colors and names
    type_colors = ["lightcoral", "darkgreen"]  # 0=randomly select, 1=rule-based
    type_names = ["Randomly Select", "Rule-based"]

    # PCA visualization
    if blocker_embeddings.shape[1] > 2:
        print("Computing PCA for blocker types...")
        pca = PCA(n_components=2)
        embeddings_pca = pca.fit_transform(blocker_embeddings)

        unique_types = np.unique(blocker_types)
        for i, blocker_type in enumerate(unique_types):
            mask = blocker_types == blocker_type
            type_count = np.sum(mask)
            if type_count > 0:
                color = type_colors[i] if i < len(type_colors) else f"C{i}"
                name = type_names[i] if i < len(type_names) else f"Type {blocker_type}"
                ax1.scatter(
                    embeddings_pca[mask, 0],
                    embeddings_pca[mask, 1],
                    c=color,
                    label=f"{name} (n={type_count})",
                    alpha=embedding_plots["alpha"],
                    s=embedding_plots["marker_size"],
                )

        ax1.set_title(f"PCA by Blocker Type")
        ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
        ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # t-SNE visualization
    print("Computing t-SNE for blocker types...")
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=min(30, len(blocker_embeddings) // 4),
    )
    embeddings_tsne = tsne.fit_transform(blocker_embeddings)

    unique_types = np.unique(blocker_types)
    for i, blocker_type in enumerate(unique_types):
        mask = blocker_types == blocker_type
        type_count = np.sum(mask)
        if type_count > 0:
            color = type_colors[i] if i < len(type_colors) else f"C{i}"
            name = type_names[i] if i < len(type_names) else f"Type {blocker_type}"
            ax2.scatter(
                embeddings_tsne[mask, 0],
                embeddings_tsne[mask, 1],
                c=color,
                label=f"{name} (n={type_count})",
                alpha=embedding_plots["alpha"],
                s=embedding_plots["marker_size"],
            )

    ax2.set_title("t-SNE by Blocker Type")
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
                output_dir, f"character_embeddings_blocker_type_exp{experiment_no}.png"
            ),
            dpi=300,
            bbox_inches="tight",
        )
        print(f"Blocker Type embedding plot saved to {output_dir}")

    plt.close()


def _plot_type_based_embeddings_for_achiever(
    embeddings,
    agent_labels,
    goal_labels,
    type_labels,
    config,
    output_dir,
    experiment_no,
):
    """Plot embeddings colored by Type, constrained to Achiever agents only"""
    vis_config = config.get_visualization_config()
    embedding_plots = vis_config["embedding_plots"]

    print("\nCreating Type-based embedding plots for Achiever agents...")

    # Filter for achiever agents only
    achiever_mask = agent_labels == "achiever"
    achiever_embeddings = embeddings[achiever_mask]
    achiever_types = type_labels[achiever_mask]

    if len(achiever_embeddings) == 0:
        print("No achiever samples found for Type visualization")
        return

    print(f"Found {len(achiever_embeddings)} achiever samples for Type visualization")
    print(f"Type distribution: {np.unique(achiever_types, return_counts=True)}")

    # Create figure with PCA and t-SNE subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=embedding_plots["pca_figsize"])
    fig.suptitle(
        f"Character Embeddings by Achiever Type (Experiment {experiment_no})",
        fontsize=16,
    )

    # Type colors and names for achievers
    type_colors = ["lightblue", "darkblue"]  # 0=random/lv0va, 1=strategic/lv1va
    type_names = ["Random Achiever", "Strategic Achiever"]

    # PCA visualization
    if achiever_embeddings.shape[1] > 2:
        print("Computing PCA for achiever types...")
        pca = PCA(n_components=2)
        embeddings_pca = pca.fit_transform(achiever_embeddings)

        unique_types = np.unique(achiever_types)
        for i, achiever_type in enumerate(unique_types):
            mask = achiever_types == achiever_type
            type_count = np.sum(mask)
            if type_count > 0:
                color = type_colors[i] if i < len(type_colors) else f"C{i}"
                name = type_names[i] if i < len(type_names) else f"Type {achiever_type}"
                ax1.scatter(
                    embeddings_pca[mask, 0],
                    embeddings_pca[mask, 1],
                    c=color,
                    label=f"{name} (n={type_count})",
                    alpha=embedding_plots["alpha"],
                    s=embedding_plots["marker_size"],
                )

        ax1.set_title(f"PCA by Achiever Type")
        ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
        ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # t-SNE visualization
    print("Computing t-SNE for achiever types...")
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=min(30, len(achiever_embeddings) // 4),
    )
    embeddings_tsne = tsne.fit_transform(achiever_embeddings)

    unique_types = np.unique(achiever_types)
    for i, achiever_type in enumerate(unique_types):
        mask = achiever_types == achiever_type
        type_count = np.sum(mask)
        if type_count > 0:
            color = type_colors[i] if i < len(type_colors) else f"C{i}"
            name = type_names[i] if i < len(type_names) else f"Type {achiever_type}"
            ax2.scatter(
                embeddings_tsne[mask, 0],
                embeddings_tsne[mask, 1],
                c=color,
                label=f"{name} (n={type_count})",
                alpha=embedding_plots["alpha"],
                s=embedding_plots["marker_size"],
            )

    ax2.set_title("t-SNE by Achiever Type")
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
                output_dir, f"character_embeddings_achiever_type_exp{experiment_no}.png"
            ),
            dpi=300,
            bbox_inches="tight",
        )
        print(f"Achiever Type embedding plot saved to {output_dir}")

    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize AchieverBlocker ToMnet results"
    )
    parser.add_argument(
        "--config_override",
        action="store_true",
        help="Override config with command line arguments",
    )
    parser.add_argument("--result_dir", type=str, help="Directory containing results")
    parser.add_argument("--plot_dir", type=str, help="Directory to save plots")
    parser.add_argument(
        "--experiment_no", type=int, default=5, help="Experiment number"
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

    results_dir = args.result_dir or getattr(config, "result_dir", "results/exp6")
    plot_dir = args.plot_dir or getattr(config, "plot_dir", "results/exp6/plots")
    experiment_no = args.experiment_no or getattr(config, "experiment_no", 5)

    # Create plot directory
    os.makedirs(plot_dir, exist_ok=True)

    title_prefix = "Single-Agent" if config.is_single_agent_mode() else "AchieverBlocker"
    print(f"Creating {title_prefix} visualizations for experiment {experiment_no}")
    print(f"Results directory: {results_dir}")
    print(f"Plot directory: {plot_dir}")

    # Plot training curves
    if args.plot_type in ["training", "all"]:
        # Get history file paths from config
        history_config = config.get_history_config()
        history_files = history_config.get(
            "history_files",
            [
                os.path.join(results_dir, "training_history.json"),
                os.path.join(
                    results_dir, f"exp{experiment_no}_*/training_history.json"
                ),
            ],
        )

        import glob

        for pattern in history_files:
            matching_files = glob.glob(pattern)
            for history_file in matching_files:
                if os.path.exists(history_file):
                    plot_training_curves(history_file, plot_dir, config, experiment_no)
                    break

    # Plot confusion matrix
    if args.plot_type in ["confusion", "all"]:
        # Get prediction file paths from config
        pred_config = config.get_prediction_config()
        pred_files = pred_config.get(
            "prediction_files",
            [
                os.path.join(results_dir, "predictions.pkl"),
                os.path.join(results_dir, f"exp{experiment_no}_*/predictions.pkl"),
            ],
        )

        import glob

        for pattern in pred_files:
            matching_files = glob.glob(pattern)
            for pred_file in matching_files:
                if os.path.exists(pred_file):
                    plot_confusion_matrix(pred_file, plot_dir, config, experiment_no)
                    break

    # Plot action likelihood
    if args.plot_type in ["likelihood", "all"]:
        # Get prediction file paths from config
        pred_config = config.get_prediction_config()
        pred_files = pred_config.get(
            "prediction_files",
            [
                os.path.join(results_dir, "predictions.pkl"),
                os.path.join(results_dir, f"exp{experiment_no}_*/predictions.pkl"),
            ],
        )

        import glob

        for pattern in pred_files:
            matching_files = glob.glob(pattern)
            for pred_file in matching_files:
                if os.path.exists(pred_file):
                    plot_action_likelihood(pred_file, plot_dir, config, experiment_no)
                    break

    # Plot character embeddings
    if args.plot_type in ["embeddings", "all"]:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Find the best model
        model_path = os.path.join(results_dir, "best_model.pth")
        
        if os.path.exists(model_path):
            # Load model
            from evaluate import load_model
            model_kwargs = config.get_model_kwargs()
            model = load_model(model_path, device, model_kwargs)
            
            # Load test data using the same multi-combination approach as evaluate.py
            from utils import load_test_data_all_combinations, combine_all_combinations_data
            
            # Generate base path from config based on single-agent vs multi-agent mode
            if config.is_single_agent_mode():
                # For single-agent mode, use the first achiever type's test directory
                achiever_type = list(config.achiever_types.keys())[0]
                test_data_dir_base = os.path.dirname(config.get_training_data_path(achiever_type, None, is_test=True))
            else:
                # For multi-agent mode, use environment base path
                env_name = config.get_env_name()
                test_data_dir_base = f"./data/{env_name}"
            
            # Load test data for all combinations efficiently (same as evaluate.py)
            try:
                all_test_data = load_test_data_all_combinations(config, test_data_dir_base=test_data_dir_base)
                # Combine data from all combinations
                test_data = combine_all_combinations_data(all_test_data)
                print(f"Successfully loaded test data from all combinations: {test_data['trajectories'].shape[0]} samples")
            except Exception as e:
                print(f"Failed to load test data from combinations: {e}")
                test_data = None

            if test_data:
                # Convert numpy arrays to tensors for TensorDataset (same as evaluate.py fix)
                test_tensors = {
                    key: torch.from_numpy(data) if isinstance(data, np.ndarray) else torch.tensor(data)
                    for key, data in test_data.items()
                }
                
                # Create test dataset
                test_dataset = TensorDataset(
                    test_tensors["trajectories"],
                    test_tensors["actions"],
                    test_tensors["goals"],
                    test_tensors["goal_ranks"],
                    test_tensors["agents"],
                    test_tensors["types"],
                    test_tensors["consumption_labels"],
                    test_tensors["sr_labels"],
                )
                test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

                # Create character embeddings plot
                plot_character_embeddings(
                    model,
                    test_loader,
                    device,
                    plot_dir,
                    config,
                    experiment_no,
                    n_samples=None,
                )
                print("Character embedding visualization completed!")
            else:
                print("No test games found for character embedding visualization")
        else:
            print(f"Model file not found: {model_path}")

    print("Visualization completed!")


