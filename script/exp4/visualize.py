import os
import json
import pickle
import warnings

# Set matplotlib backend before importing pyplot to avoid display issues
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
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
from train import prepare_data_for_training, generate_past_episodes_from_batch
from data_generation import DataGenerator as DataReader, DataGenerator
# Remove circular import - load_model will be imported locally when needed

"""
Visualization tools for AchieverBlocker ToMnet experiment
Adapted from ToMnetF experiment5 for multi-agent AchieverBlocker environment
"""

# No caching - keep it simple and reliable


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
    ax1.set_title(
        "AchieverBlocker: Action Accuracy vs N_past", fontsize=14, fontweight="bold"
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
    ax2.set_title("AchieverBlocker: F1 Score vs N_past", fontsize=14, fontweight="bold")
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
    print("\nAchieverBlocker Accuracy by N_past Summary:")
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

    ax.set_title(
        "AchieverBlocker: Per-Action Accuracy by N_past", fontsize=14, fontweight="bold"
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

    fig.suptitle(
        f"AchieverBlocker ToMnet Training History (Experiment {experiment_no})",
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
    print(f"\nAchieverBlocker Training Summary (Experiment {experiment_no}):")
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

    ax.set_title(
        f"AchieverBlocker: Confusion Matrix (Experiment {experiment_no})",
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
    print(
        f"\nAchieverBlocker Confusion Matrix Statistics (Experiment {experiment_no}):"
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
    fig.suptitle(
        f"AchieverBlocker: Action Likelihood Distributions (Experiment {experiment_no})",
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

    # Load processed test data to get agent information
    print("Loading processed test data to extract agent and goal information...")

    env_name = config.get_env_name()
    agent_pair = config.get_agent_pair_name()
    test_data_dir = f"./data/{env_name}/{agent_pair}/test"

    processed_test_data_path = os.path.join(
        test_data_dir, f"processed_test_data_exp{experiment_no}.pkl"
    )

    if not os.path.exists(processed_test_data_path):
        print(f"Error: Processed test data not found at {processed_test_data_path}")
        print("Please run the evaluation script first to generate processed data.")
        return

    # Load agent and goal information from original processed data
    data_reader = DataGenerator(
        time_step=config.get_data_config().get("time_step", 500),
        w=config.width,
        h=config.height,
        d=config.get_data_config().get("maze_depth", 9),
        config=config,
    )

    processed_data = data_reader.load_processed_data(processed_test_data_path)
    print(
        f"Loaded processed data with {processed_data['trajectories'].shape[0]} samples"
    )

    # Extract agent labels and goal labels from processed tensors
    # agents tensor: 0=achiever, 1=blocker
    agent_indices = processed_data["agents"].numpy()
    agent_labels = np.array(
        ["achiever" if idx == 0 else "blocker" for idx in agent_indices]
    )

    # goals tensor: one-hot encoded [A, B, C, D]
    goals_tensor = processed_data["goals"].numpy()
    goal_labels = np.argmax(goals_tensor, axis=1)  # Convert one-hot to indices

    print(f"Agent distribution: {np.unique(agent_labels, return_counts=True)}")
    print(f"Goal distribution: {np.unique(goal_labels, return_counts=True)}")

    # Extract character embeddings using the model
    model.eval()
    embeddings = []

    # Get the data tensors
    trajectories_tensor = processed_data["trajectories"]
    goals_tensor = processed_data["goals"]
    goal_ranks_tensor = processed_data["goal_ranks"]
    agents_tensor = processed_data["agents"]

    if n_samples is not None:
        # Limit samples if specified
        indices = np.random.choice(
            len(agent_labels), min(n_samples, len(agent_labels)), replace=False
        )
        trajectories_tensor = trajectories_tensor[indices]
        goals_tensor = goals_tensor[indices]
        goal_ranks_tensor = goal_ranks_tensor[indices]
        agents_tensor = agents_tensor[indices]
        agent_labels = agent_labels[indices]
        goal_labels = goal_labels[indices]

    print(f"Extracting character embeddings for {len(agent_labels)} samples...")

    # Optimized batch processing for character embeddings
    batch_size = 32  # Process in batches for better GPU utilization
    n_past_config = config.get_n_past_evaluation_config()
    
    with torch.no_grad():
        for start_idx in range(0, len(agent_labels), batch_size):
            end_idx = min(start_idx + batch_size, len(agent_labels))
            
            if start_idx % (batch_size * 10) == 0:
                print(f"Processing batch {start_idx//batch_size + 1}/{(len(agent_labels) + batch_size - 1)//batch_size}")

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
                    rank_threshold=1,
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

    # Create the three types of plots
    _plot_agent_based_embeddings(
        embeddings, agent_labels, goal_labels, config, output_dir, experiment_no
    )
    _plot_goal_based_embeddings(
        embeddings, agent_labels, goal_labels, config, output_dir, experiment_no
    )
    _plot_separate_agent_goal_embeddings(
        embeddings, agent_labels, goal_labels, config, output_dir, experiment_no
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
    if len(embeddings) > 50:
        print("Computing t-SNE (this may take a while)...")
        # Use faster t-SNE parameters for better performance
        tsne = TSNE(
            n_components=2, 
            random_state=42, 
            perplexity=min(30, len(embeddings)//4),  # Adaptive perplexity
            n_iter=300,  # Reduced iterations for speed
            early_exaggeration=12,
            learning_rate='auto'
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
    if len(embeddings) > 50:
        print("Computing t-SNE (this may take a while)...")
        # Use faster t-SNE parameters for better performance
        tsne = TSNE(
            n_components=2, 
            random_state=42, 
            perplexity=min(30, len(embeddings)//4),  # Adaptive perplexity
            n_iter=300,  # Reduced iterations for speed
            early_exaggeration=12,
            learning_rate='auto'
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
        if len(agent_embeddings) > 30:  # Minimum samples for t-SNE
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
        else:
            ax_tsne.text(
                0.5,
                0.5,
                f"Insufficient samples\nfor t-SNE ({len(agent_embeddings)})",
                ha="center",
                va="center",
                transform=ax_tsne.transAxes,
            )
            ax_tsne.set_title(f"t-SNE: {agent.capitalize()} Goals")

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


# Keep the rest of the old function for backward compatibility if needed
def plot_character_embeddings_old(
    model,
    test_loader,
    device,
    output_dir,
    config=None,
    experiment_no=None,
    n_samples=None,
):
    """
    Original plot_character_embeddings function (kept for backward compatibility)
    """
    if config is None:
        config = Config()

    if experiment_no is None:
        experiment_no = config.experiment_no
    plt.style.use("seaborn-v0_8")

    model.eval()
    embeddings = []
    goal_labels = []

    sample_count = 0
    batch_count = 0
    successful_batches = 0
    failed_batches = 0

    if n_samples is None:
        print("Starting character embedding extraction for all test samples...")
    else:
        print(f"Starting character embedding extraction for {n_samples} samples...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            batch_count += 1
            if batch_idx % 50 == 0:  # Print every 50 batches instead of every batch
                print(
                    f"Processing batch {batch_idx + 1}/{len(test_loader)}, current sample count: {sample_count}"
                )

            if n_samples is not None and sample_count >= n_samples:
                print(f"Reached target sample count of {n_samples}, stopping...")
                break

            # Handle KeyDoor exp4 batch format (7 elements) - use native exp4 logic
            if len(batch) >= 7:
                (
                    trajectories,
                    actions,
                    goals,
                    goal_ranks,
                    consumption_labels,
                    sr_labels,
                ) = batch
                trajectories = trajectories.to(device)
                goals = goals.to(device)
                goal_ranks = goal_ranks.to(device)

                batch_size = trajectories.size(0)

                # Use KeyDoor exp4 native logic with goal_ranks (don't force exp5 compatibility)
                n_past_config = config.get_n_past_evaluation_config()

                past_episodes = generate_past_episodes_from_batch(
                    trajectories=trajectories,
                    goals=goal_ranks,  # Use goal_ranks as intended in exp4
                    batch_size=batch_size,
                    n_past_min=n_past_config["n_past_min"],
                    n_past_max=n_past_config["n_past_max"],
                    max_n_past=n_past_config["n_past_max"],
                    rank_threshold=1,  # KeyDoor exp4 parameter
                )

                # Use KeyDoor exp4 native character embedding method
                try:
                    char_embeddings = model.get_character_embedding(
                        past_episodes
                    )  # Native exp4 method

                    embeddings.extend(char_embeddings.cpu().numpy())
                    goal_labels.extend(goals.cpu().numpy())

                    sample_count += len(goals)
                    successful_batches += 1
                except Exception as e:
                    print(f"  Error getting character embeddings: {e}")
                    failed_batches += 1
                    continue
            else:
                failed_batches += 1
                continue

    print(
        f"Character embedding extraction completed. Total embeddings: {len(embeddings)}"
    )
    print(f"Batch processing summary:")
    print(f"  Total batches processed: {batch_count}")
    print(f"  Successful batches: {successful_batches}")
    print(f"  Failed batches: {failed_batches}")
    print(f"  Final sample count: {sample_count}")

    if len(embeddings) == 0:
        print("No embeddings to visualize - no valid samples found!")
        return

    embeddings = np.array(embeddings)
    goal_labels = np.array(goal_labels)

    print(f"\nData for visualization:")
    print(f"  Embeddings array shape: {embeddings.shape}")
    print(f"  Goal labels array shape: {goal_labels.shape}")
    print(f"  Unique goals in data: {np.unique(goal_labels)}")
    print(
        f"  Goal distribution: {np.bincount(goal_labels) if len(goal_labels) > 0 else 'N/A'}"
    )

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"AchieverBlocker: Character Embeddings (Experiment {experiment_no})",
        fontsize=16,
    )

    # Get goal information from config
    goal_config = config.get_goal_config()
    num_goals = goal_config.get("num_goals", 4)
    goal_colors = goal_config.get(
        "goal_colors", ["red", "green", "blue", "yellow"][:num_goals]
    )
    goal_names = goal_config.get(
        "goal_names", [f"Goal {chr(65+i)}" for i in range(num_goals)]
    )

    # PCA visualization
    if embeddings.shape[1] > 2:
        try:
            print(f"\nPCA Analysis:")
            print(f"  Input embeddings shape: {embeddings.shape}")

            pca = PCA(n_components=2)
            embeddings_pca = pca.fit_transform(embeddings)
            print(f"  PCA output shape: {embeddings_pca.shape}")
            print(
                f"  Explained variance ratio: PC1={pca.explained_variance_ratio_[0]:.4f}, PC2={pca.explained_variance_ratio_[1]:.4f}"
            )

            # Check for problematic variance (all embeddings are identical)
            if (
                pca.explained_variance_ratio_[0] > 0.99
                and pca.explained_variance_ratio_[1] < 0.01
            ):
                print(f"  WARNING: Character embeddings show no meaningful variance!")
                print(
                    f"  This suggests all embeddings are identical or nearly identical."
                )
                print(
                    f"  The character network may not be learning goal-specific representations."
                )

            # Get unique goals present in the data (aligned with exp5 logic)
            unique_goals = np.unique(goal_labels)
            print(f"  Unique goals for PCA: {unique_goals}")

            pca_sample_counts = {}
            for goal in unique_goals:
                mask = goal_labels == goal
                goal_count = np.sum(mask)
                pca_sample_counts[goal] = goal_count
                print(f"    Goal {goal}: {goal_count} samples")

                if goal_count > 0:
                    # Use 1-based indexing for colors and names (aligned with exp5)
                    color_idx = (
                        int(goal) - 1
                        if goal <= len(goal_colors)
                        else goal % len(goal_colors)
                    )
                    goal_name = (
                        goal_names[int(goal) - 1]
                        if goal <= len(goal_names)
                        else f"Goal {goal}"
                    )
                    ax1.scatter(
                        embeddings_pca[mask, 0],
                        embeddings_pca[mask, 1],
                        c=goal_colors[color_idx],
                        label=f"{goal_name} (n={goal_count})",
                        alpha=0.6,
                    )

            ax1.set_title(f"PCA Visualization (n={len(embeddings)} total)")
            ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
            ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        except Exception as e:
            print(f"Error in PCA visualization: {e}")
            ax1.text(
                0.5,
                0.5,
                f"PCA Error: {str(e)}",
                ha="center",
                va="center",
                transform=ax1.transAxes,
            )
    else:
        # For 2D embeddings, plot directly
        unique_goals = np.unique(goal_labels)
        for goal in unique_goals:
            mask = goal_labels == goal
            if np.sum(mask) > 0:
                # Use 1-based indexing for colors and names (aligned with exp5)
                color_idx = (
                    int(goal) - 1
                    if goal <= len(goal_colors)
                    else goal % len(goal_colors)
                )
                goal_name = (
                    goal_names[int(goal) - 1]
                    if goal <= len(goal_names)
                    else f"Goal {goal}"
                )
                ax1.scatter(
                    embeddings[mask, 0],
                    embeddings[mask, 1],
                    c=goal_colors[color_idx],
                    label=goal_name,
                    alpha=0.6,
                )
        ax1.set_title("Character Embeddings (2D)")
        ax1.set_xlabel("Dimension 1")
        ax1.set_ylabel("Dimension 2")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # t-SNE visualization
    if len(embeddings) > 50:  # t-SNE needs sufficient samples
        try:
            print(f"\nt-SNE Analysis:")
            print(f"  Total available embeddings: {len(embeddings)}")

            # Use all available samples for t-SNE (no limit)
            embeddings_for_tsne = embeddings
            goals_tsne = goal_labels

            print(f"  Using all {len(embeddings)} samples for t-SNE")
            print(f"  t-SNE input shape: {embeddings_for_tsne.shape}")
            print(
                f"  WARNING: t-SNE with {len(embeddings)} samples may take several minutes to compute..."
            )

            tsne = TSNE(n_components=2, random_state=42)
            embeddings_tsne = tsne.fit_transform(embeddings_for_tsne)
            print(f"  t-SNE output shape: {embeddings_tsne.shape}")

            # Get unique goals present in the data (aligned with exp5 logic)
            unique_goals_tsne = np.unique(goals_tsne)
            print(f"  Unique goals for t-SNE: {unique_goals_tsne}")

            tsne_sample_counts = {}
            for goal in unique_goals_tsne:
                mask = goals_tsne == goal
                goal_count = np.sum(mask)
                tsne_sample_counts[goal] = goal_count
                print(f"    Goal {goal}: {goal_count} samples")

                if goal_count > 0:
                    # Use 1-based indexing for colors and names (aligned with exp5)
                    color_idx = (
                        int(goal) - 1
                        if goal <= len(goal_colors)
                        else goal % len(goal_colors)
                    )
                    goal_name = (
                        goal_names[int(goal) - 1]
                        if goal <= len(goal_names)
                        else f"Goal {goal}"
                    )
                    ax2.scatter(
                        embeddings_tsne[mask, 0],
                        embeddings_tsne[mask, 1],
                        c=goal_colors[color_idx],
                        label=f"{goal_name} (n={goal_count})",
                        alpha=0.6,
                    )

            ax2.set_title(f"t-SNE Visualization (n={len(embeddings)})")
            ax2.set_xlabel("t-SNE 1")
            ax2.set_ylabel("t-SNE 2")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        except Exception as e:
            print(f"Error in t-SNE visualization: {e}")
            ax2.text(
                0.5,
                0.5,
                f"t-SNE Error: {str(e)}",
                ha="center",
                va="center",
                transform=ax2.transAxes,
            )
    else:
        print(f"\nt-SNE Analysis:")
        print(f"  Insufficient samples for t-SNE: {len(embeddings)} < 50 required")
        ax2.text(
            0.5,
            0.5,
            f"Not enough samples for t-SNE\n({len(embeddings)} < 50 required)",
            ha="center",
            va="center",
            transform=ax2.transAxes,
            fontsize=12,
        )
        ax2.set_title(f"t-SNE Visualization (n={len(embeddings)} insufficient)")

    plt.tight_layout()

    # Save plot
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(
            os.path.join(
                output_dir,
                f"achieverblocker_character_embeddings_exp{experiment_no}.png",
            ),
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()

    print(
        f"\nAchieverBlocker Character Embeddings Analysis (Experiment {experiment_no}):"
    )
    print("-" * 60)
    print(f"Total samples collected: {len(embeddings)}")
    print(
        f"Embedding dimension: {embeddings.shape[1] if len(embeddings) > 0 else 'N/A'}"
    )
    print(f"Data loader had {len(test_loader)} batches")
    print(f"Batch processing: {successful_batches} successful, {failed_batches} failed")

    # Per-goal statistics
    if len(embeddings) > 0:
        unique_goals = np.unique(goal_labels)
        print(f"\nPer-goal distribution:")
        for goal in unique_goals:
            mask = goal_labels == goal
            count = np.sum(mask)
            percentage = (count / len(goal_labels)) * 100
            # Use 1-based indexing for goal names (aligned with exp5 logic)
            goal_name = (
                goal_names[int(goal) - 1] if goal <= len(goal_names) else f"Goal {goal}"
            )
            print(f"  {goal_name}: {count} samples ({percentage:.1f}%)")

        print(f"\nVisualization summary:")
        print(f"  PCA: Used {len(embeddings)} samples")
        print(
            f"  t-SNE: {'Used ' + str(len(embeddings)) + ' samples' if len(embeddings) > 50 else 'Insufficient samples (' + str(len(embeddings)) + ' < 50)'}"
        )
    else:
        print("\nNo samples available for analysis!")


def create_additional_visualizations(
    model,
    test_loader,
    output_dir,
    device,
    config=None,
    experiment_no=None,
    save_plots=True,
):
    """
    Create additional visualizations for KeyDoor experiment

    Args:
        model: Trained ToMnet model
        test_loader: Test data loader
        output_dir: Directory to save plots
        device: Computing device
        config: Configuration object
        experiment_no: Experiment number (defaults to config.experiment_no)
        save_plots: Whether to save plots
    """
    if config is None:
        config = Config()

    if experiment_no is None:
        experiment_no = config.experiment_no

    print("Creating additional AchieverBlocker visualizations...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Plot character embeddings
    plot_character_embeddings(
        model,
        test_loader,
        device,
        output_dir,
        config,
        experiment_no,
        n_samples=None,
    )

    print("Additional visualizations completed!")


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
        "--experiment_no", type=int, default=4, help="Experiment number"
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

    results_dir = args.result_dir or getattr(config, "result_dir", "results/exp4")
    plot_dir = args.plot_dir or getattr(config, "plot_dir", "results/exp4/plots")
    experiment_no = args.experiment_no or getattr(config, "experiment_no", 4)

    # Create plot directory
    os.makedirs(plot_dir, exist_ok=True)

    print(f"Creating AchieverBlocker visualizations for experiment {experiment_no}")
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

    # Plot N_past results
    if args.plot_type in ["n_past", "all"]:
        n_past_config = config.get_n_past_config()
        n_past_file = n_past_config.get(
            "n_past_results_file",
            os.path.join(results_dir, "n_past_evaluation_results.json"),
        )
        if os.path.exists(n_past_file):
            with open(n_past_file, "r") as f:
                n_past_results = json.load(f)

            # Convert string keys back to integers
            results_by_n_past = {}
            for key, value in n_past_results.items():
                results_by_n_past[int(key)] = value

            plot_accuracy_by_n_past(results_by_n_past, plot_dir)

            # Only create heatmap if detailed predictions are available
            has_predictions = False
            for n_past, metrics in results_by_n_past.items():
                if "predictions" in metrics and "targets" in metrics:
                    has_predictions = True
                    break

            if has_predictions:
                plot_accuracy_heatmap_by_n_past(results_by_n_past, plot_dir)
            else:
                print(
                    "Skipping accuracy heatmap - detailed predictions not available in N_past results"
                )

    # Plot character embeddings
    if args.plot_type in ["embeddings", "all"]:
        print("Creating character embedding visualizations...")

        # Load model and test data for character embedding visualization
        # Import load_model locally to avoid circular import
        from evaluate import load_model
        from data_generation import DataGenerator as DataReader
        from train import prepare_data_for_training
        from torch.utils.data import DataLoader, TensorDataset
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Get model paths from config
        model_config = config.get_model_config()

        # Also check parent directory in case results_dir is timestamped
        parent_results_dir = (
            os.path.dirname(results_dir)
            if os.path.basename(results_dir).replace("_", "").replace("-", "").isdigit()
            else results_dir
        )
        
        # Also search for models in other timestamped directories within parent
        import glob
        parent_timestamped_dirs = []
        if parent_results_dir != results_dir:
            pattern = os.path.join(parent_results_dir, "20*")
            parent_timestamped_dirs = glob.glob(pattern)

        possible_model_paths = model_config.get(
            "possible_model_paths",
            [
                os.path.join(results_dir, "best_model.pth"),
                os.path.join(parent_results_dir, "best_model.pth"),  # Check parent for timestamped dirs
                os.path.join(results_dir, "model.pth"),
                os.path.join(parent_results_dir, "model.pth"),
                os.path.join(results_dir, "figure5_goal_directed_alpha0.01_model.pth"),
            ],
        )
        
        # Add paths from other timestamped directories
        for timestamped_dir in parent_timestamped_dirs:
            possible_model_paths.extend([
                os.path.join(timestamped_dir, "best_model.pth"),
                os.path.join(timestamped_dir, "model.pth"),
            ])

        # Add any additional model paths from config
        if "additional_model_paths" in model_config:
            possible_model_paths.extend(model_config["additional_model_paths"])

        model_path = None
        for path in possible_model_paths:
            if os.path.exists(path):
                model_path = path
                break

        if model_path and os.path.exists(model_path):
            model_kwargs = config.get_model_kwargs()
            model = load_model(model_path, device, model_kwargs)

            # Load test data
            data_config = config.get_data_config()
            data_reader = DataReader(
                time_step=data_config.get("time_step", 500),
                w=config.width,
                h=config.height,
                d=data_config.get("maze_depth", 9),
                config=config,
            )

            # First check if processed test data already exists from evaluation
            processed_test_data_path = None

            # Get test data path from config
            env_name = config.get_env_name()
            agent_pair = config.get_agent_pair_name()
            test_data_dir_from_config = f"./data/{env_name}/{agent_pair}/test"

            possible_processed_paths = [
                os.path.join(
                    results_dir, f"processed_test_data_exp{experiment_no}.pkl"
                ),
                os.path.join(
                    parent_results_dir, f"processed_test_data_exp{experiment_no}.pkl"
                ),
                os.path.join(
                    test_data_dir_from_config,
                    f"processed_test_data_exp{experiment_no}.pkl",
                ),  # From config
                os.path.join(results_dir, "predictions.pkl"),  # Contains processed data
                os.path.join(parent_results_dir, "predictions.pkl"),
                os.path.join(parent_results_dir, "test_evaluation", "predictions.pkl"),
            ]

            for path in possible_processed_paths:
                if os.path.exists(path):
                    processed_test_data_path = path
                    break

            test_data = None

            if processed_test_data_path and processed_test_data_path.endswith(
                "predictions.pkl"
            ):
                # Load from predictions.pkl which contains the processed data
                print(
                    f"Loading processed test data from predictions: {processed_test_data_path}"
                )
                with open(processed_test_data_path, "rb") as f:
                    pred_data = pickle.load(f)
                # predictions.pkl doesn't contain the full tensor data needed for embeddings
                # Fall back to raw test data loading
                processed_test_data_path = None
            elif processed_test_data_path:
                print(f"Loading processed test data: {processed_test_data_path}")
                test_data = data_reader.load_processed_data(processed_test_data_path)

            if test_data is None:
                # Try to find test data directory for raw data processing
                data_config = config.get_data_config()
                test_data_dirs = data_config.get(
                    "test_data_dirs",
                    [
                        test_data_dir_from_config,  # Primary path from config
                        os.path.join(os.path.dirname(results_dir), "test"),
                        (
                            config.test_data_dir
                            if hasattr(config, "test_data_dir")
                            else None
                        ),
                        os.path.join(results_dir, "test"),
                    ],
                )
                # Filter out None values
                test_data_dirs = [d for d in test_data_dirs if d is not None]

                test_data_dir = None
                for tdd in test_data_dirs:
                    if os.path.exists(tdd):
                        test_data_dir = tdd
                        break

            if test_data:
                # Use existing processed test data
                test_dataset = TensorDataset(
                    test_data["trajectories"],
                    test_data["actions"],
                    test_data["goals"],
                    test_data["goal_ranks"],
                    test_data["agents"],
                    test_data["consumption_labels"],
                    test_data["sr_labels"],
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
            elif test_data_dir:
                # Process raw test data
                test_games = data_reader.ReadAllGames(test_data_dir)
                if test_games:
                    data_config = config.get_data_config()
                    test_data = prepare_data_for_training(
                        test_games,
                        min_timestep=6,
                        max_trajectory_length=data_config["time_step"],
                    )
                    test_dataset = TensorDataset(
                        test_data["trajectories"],
                        test_data["actions"],
                        test_data["goals"],
                        test_data["goal_ranks"],
                        test_data["agents"],
                        test_data["consumption_labels"],
                        test_data["sr_labels"],
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
                print(
                    f"Test data directory not found. Tried: {test_data_dirs if 'test_data_dirs' in locals() else 'None'}"
                )
        else:
            print(f"Model file not found: {model_path}")

    print("Visualization completed!")
