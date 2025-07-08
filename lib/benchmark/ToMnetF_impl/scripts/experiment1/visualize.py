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
Publication-quality visualization for ToMnetF
@Author Filip Borowiak
"""

# Set style for publication-quality plots
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


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
    result_dir="../../result/experiment1",
    plot_dir="../../plots/experiment1",
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
        default="../../result/experiment1",
        help="Results directory",
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default="../../plots/experiment1",
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
