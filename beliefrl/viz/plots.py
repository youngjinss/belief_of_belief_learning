"""Embedding and metric plots shared by exp7 and exp8.

These fourteen functions were AST-identical in both experiments and reference
nothing else in visualize.py -- no module-level state, no Config construction,
no local base classes.

Deliberately left behind: the four plots that build a Config() as a fallback
(plot_training_curves, plot_confusion_matrix, plot_action_likelihood,
plot_character_embeddings), since the core cannot know which experiment's Config
to construct; and EmbeddingExtractor, whose base class BaseEmbeddingExtractor
differs between the two experiments.
"""

import os

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

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
    title_prefix = (
        "Single-Agent"
        if config and config.is_single_agent_mode()
        else "AchieverBlocker"
    )
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
    title_prefix = (
        "Single-Agent"
        if config and config.is_single_agent_mode()
        else "AchieverBlocker"
    )
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
    title_prefix = (
        "Single-Agent"
        if config and config.is_single_agent_mode()
        else "AchieverBlocker"
    )
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

            # Handle missing detailed predictions - using overall accuracy distributed across actions
            if n_past == n_past_values[0]:
                print("N_past results don't contain detailed predictions/targets.")
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

    title_prefix = (
        "Single-Agent"
        if config and config.is_single_agent_mode()
        else "AchieverBlocker"
    )
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

def _plot_agent_based_embeddings(
    embeddings, agent_labels, goal_labels, config, output_dir, experiment_no
):
    """Plot embeddings colored by agent type (achiever vs blocker)"""
    vis_config = config.get_visualization_config()
    agent_colors = vis_config["agent_colors"]
    agent_names = vis_config["agent_names"]
    embedding_plots = vis_config["embedding_plots"]

    # Ensure labels are numpy arrays
    if hasattr(agent_labels, "cpu"):
        agent_labels = agent_labels.cpu().numpy()
    if hasattr(goal_labels, "cpu"):
        goal_labels = goal_labels.cpu().numpy()
    agent_labels = np.asarray(agent_labels)
    goal_labels = np.asarray(goal_labels)

    print("\nCreating agent-based embedding plots...")

    # Create figure with PCA and t-SNE subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=embedding_plots["pca_figsize"])
    fig.suptitle(
        f"Character Embeddings by Agent Type (Experiment {experiment_no})", fontsize=16
    )

    # PCA visualization
    if embeddings.shape[1] > 2:
        print("Computing PCA...")
        # Check for NaN or infinite values
        if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
            print("Warning: Found NaN or infinite values in embeddings. Cleaning...")
            # Replace NaN with 0 and clip infinite values
            embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=1e10, neginf=-1e10)
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

    # Ensure goal_labels is numpy array
    if hasattr(goal_labels, "cpu"):
        goal_labels = goal_labels.cpu().numpy()
    goal_labels = np.asarray(goal_labels)

    # Create figure with PCA and t-SNE subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=embedding_plots["pca_figsize"])
    fig.suptitle(
        f"Character Embeddings by Goal Type (Experiment {experiment_no})", fontsize=16
    )

    # PCA visualization
    if embeddings.shape[1] > 2:
        print("Computing PCA...")
        # Check for NaN or infinite values
        if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
            print("Warning: Found NaN or infinite values in embeddings. Cleaning...")
            # Replace NaN with 0 and clip infinite values
            embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=1e10, neginf=-1e10)
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

    # Ensure labels are numpy arrays
    if hasattr(agent_labels, "cpu"):
        agent_labels = agent_labels.cpu().numpy()
    if hasattr(goal_labels, "cpu"):
        goal_labels = goal_labels.cpu().numpy()
    agent_labels = np.asarray(agent_labels)
    goal_labels = np.asarray(goal_labels)

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
            # Check for NaN or infinite values
            if np.any(np.isnan(agent_embeddings)) or np.any(np.isinf(agent_embeddings)):
                print(
                    "Warning: Found NaN or infinite values in agent embeddings. Cleaning..."
                )
                # Replace NaN with 0 and clip infinite values
                agent_embeddings = np.nan_to_num(
                    agent_embeddings, nan=0.0, posinf=1e10, neginf=-1e10
                )
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
                name = goal_names[goal] if goal < len(goal_names) else f"Goal {goal}"
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

    # Ensure labels are numpy arrays
    if hasattr(agent_labels, "cpu"):
        agent_labels = agent_labels.cpu().numpy()
    if hasattr(goal_labels, "cpu"):
        goal_labels = goal_labels.cpu().numpy()
    if hasattr(type_labels, "cpu"):
        type_labels = type_labels.cpu().numpy()
    agent_labels = np.asarray(agent_labels)
    goal_labels = np.asarray(goal_labels)
    type_labels = np.asarray(type_labels)
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
        # Check for NaN or infinite values
        if np.any(np.isnan(blocker_embeddings)) or np.any(np.isinf(blocker_embeddings)):
            print(
                "Warning: Found NaN or infinite values in blocker embeddings. Cleaning..."
            )
            # Replace NaN with 0 and clip infinite values
            blocker_embeddings = np.nan_to_num(
                blocker_embeddings, nan=0.0, posinf=1e10, neginf=-1e10
            )
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
        # Check for NaN or infinite values
        if np.any(np.isnan(achiever_embeddings)) or np.any(
            np.isinf(achiever_embeddings)
        ):
            print(
                "Warning: Found NaN or infinite values in achiever embeddings. Cleaning..."
            )
            # Replace NaN with 0 and clip infinite values
            achiever_embeddings = np.nan_to_num(
                achiever_embeddings, nan=0.0, posinf=1e10, neginf=-1e10
            )
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

def _plot_mental_agent_based_embeddings(
    embeddings, agent_labels, goal_labels, config, output_dir, experiment_no
):
    """Plot mental embeddings colored by agent type (achiever vs blocker)"""
    vis_config = config.get_visualization_config()
    agent_colors = vis_config["agent_colors"]
    agent_names = vis_config["agent_names"]
    embedding_plots = vis_config["embedding_plots"]

    print("\nCreating agent-based mental embedding plots...")

    # Create figure with PCA and t-SNE subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=embedding_plots["pca_figsize"])
    fig.suptitle(
        f"Mental Embeddings by Agent Type (Experiment {experiment_no})", fontsize=16
    )

    # PCA visualization
    if embeddings.shape[1] > 2:
        print("Computing PCA for mental embeddings...")
        # Check for NaN or infinite values
        if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
            print("Warning: Found NaN or infinite values in embeddings. Cleaning...")
            # Replace NaN with 0 and clip infinite values
            embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=1e10, neginf=-1e10)
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
    print("Computing t-SNE for mental embeddings...")
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=min(30, len(embeddings) // 4),
        n_iter=300,
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
        os.path.join(output_dir, f"mental_embeddings_by_agent_exp{experiment_no}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

def _plot_mental_goal_based_embeddings(
    embeddings, agent_labels, goal_labels, config, output_dir, experiment_no
):
    """Plot mental embeddings colored by goal type (red, green, blue, yellow)"""
    vis_config = config.get_visualization_config()
    goal_colors = vis_config["goal_colors"]
    goal_names = vis_config["goal_names"]
    embedding_plots = vis_config["embedding_plots"]

    print("\nCreating goal-based mental embedding plots...")

    # Create figure with PCA and t-SNE subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=embedding_plots["pca_figsize"])
    fig.suptitle(
        f"Mental Embeddings by Goal Type (Experiment {experiment_no})", fontsize=16
    )

    # PCA visualization
    if embeddings.shape[1] > 2:
        print("Computing PCA for mental embeddings...")
        # Check for NaN or infinite values
        if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
            print("Warning: Found NaN or infinite values in embeddings. Cleaning...")
            # Replace NaN with 0 and clip infinite values
            embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=1e10, neginf=-1e10)
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
    print("Computing t-SNE for mental embeddings...")
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=min(30, len(embeddings) // 4),
        n_iter=300,
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
        os.path.join(output_dir, f"mental_embeddings_by_goal_exp{experiment_no}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

def _plot_mental_separate_agent_goal_embeddings(
    embeddings, agent_labels, goal_labels, config, output_dir, experiment_no
):
    """Plot separate mental embeddings for achiever goals and blocker goals"""
    vis_config = config.get_visualization_config()
    goal_colors = vis_config["goal_colors"]
    goal_names = vis_config["goal_names"]
    embedding_plots = vis_config["embedding_plots"]

    print("\nCreating separate agent-goal mental embedding plots...")

    # Create figure with 2x2 subplots (PCA and t-SNE for each agent)
    fig, axes = plt.subplots(2, 2, figsize=embedding_plots["combined_figsize"])
    fig.suptitle(
        f"Mental Embeddings: Achiever vs Blocker Goals (Experiment {experiment_no})",
        fontsize=16,
    )

    agents = ["achiever", "blocker"]

    for agent_idx, agent in enumerate(agents):
        agent_mask = agent_labels == agent
        agent_embeddings = embeddings[agent_mask]
        agent_goals = goal_labels[agent_mask]

        if len(agent_embeddings) == 0:
            print(f"No mental embeddings found for {agent}")
            continue

        print(f"Processing {agent}: {len(agent_embeddings)} mental embeddings")

        # PCA for this agent
        ax_pca = axes[agent_idx, 0]
        if agent_embeddings.shape[1] > 2:
            # Check for NaN or infinite values
            if np.any(np.isnan(agent_embeddings)) or np.any(np.isinf(agent_embeddings)):
                print(
                    "Warning: Found NaN or infinite values in agent embeddings. Cleaning..."
                )
                # Replace NaN with 0 and clip infinite values
                agent_embeddings = np.nan_to_num(
                    agent_embeddings, nan=0.0, posinf=1e10, neginf=-1e10
                )
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
                name = goal_names[goal] if goal < len(goal_names) else f"Goal {goal}"
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
            output_dir, f"mental_embeddings_separate_agents_exp{experiment_no}.png"
        ),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"All mental embedding plots saved to {output_dir}")

def _plot_mental_type_based_embeddings_for_blockers(
    embeddings,
    agent_labels,
    goal_labels,
    type_labels,
    config,
    output_dir,
    experiment_no,
):
    """Plot mental embeddings colored by Type, constrained to Blocker agents only"""

    # Ensure labels are numpy arrays
    if hasattr(agent_labels, "cpu"):
        agent_labels = agent_labels.cpu().numpy()
    if hasattr(goal_labels, "cpu"):
        goal_labels = goal_labels.cpu().numpy()
    if hasattr(type_labels, "cpu"):
        type_labels = type_labels.cpu().numpy()
    agent_labels = np.asarray(agent_labels)
    goal_labels = np.asarray(goal_labels)
    type_labels = np.asarray(type_labels)
    vis_config = config.get_visualization_config()
    embedding_plots = vis_config["embedding_plots"]

    print("\nCreating Type-based mental embedding plots for Blocker agents...")

    # Filter for blocker agents only
    blocker_mask = agent_labels == "blocker"
    blocker_embeddings = embeddings[blocker_mask]
    blocker_types = type_labels[blocker_mask]

    if len(blocker_embeddings) == 0:
        print("No blocker samples found for mental Type visualization")
        return

    print(
        f"Found {len(blocker_embeddings)} blocker samples for mental Type visualization"
    )
    print(f"Type distribution: {np.unique(blocker_types, return_counts=True)}")

    # Create figure with PCA and t-SNE subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=embedding_plots["pca_figsize"])
    fig.suptitle(
        f"Mental Embeddings by Blocker Type (Experiment {experiment_no})",
        fontsize=16,
    )

    # Type colors and names
    type_colors = ["lightcoral", "darkgreen"]  # 0=randomly select, 1=rule-based
    type_names = ["Randomly Select", "Rule-based"]

    # PCA visualization
    if blocker_embeddings.shape[1] > 2:
        print("Computing PCA for blocker mental types...")
        # Check for NaN or infinite values
        if np.any(np.isnan(blocker_embeddings)) or np.any(np.isinf(blocker_embeddings)):
            print(
                "Warning: Found NaN or infinite values in blocker embeddings. Cleaning..."
            )
            # Replace NaN with 0 and clip infinite values
            blocker_embeddings = np.nan_to_num(
                blocker_embeddings, nan=0.0, posinf=1e10, neginf=-1e10
            )
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

        ax1.set_title(f"PCA by Blocker Mental Type")
        ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
        ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # t-SNE visualization
    print("Computing t-SNE for blocker mental types...")
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

    ax2.set_title("t-SNE by Blocker Mental Type")
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
                output_dir, f"mental_embeddings_blocker_type_exp{experiment_no}.png"
            ),
            dpi=300,
            bbox_inches="tight",
        )
        print(f"Blocker Mental Type embedding plot saved to {output_dir}")

    plt.close()

def _plot_mental_type_based_embeddings_for_achiever(
    embeddings,
    agent_labels,
    goal_labels,
    type_labels,
    config,
    output_dir,
    experiment_no,
):
    """Plot mental embeddings colored by Type, constrained to Achiever agents only"""
    vis_config = config.get_visualization_config()
    embedding_plots = vis_config["embedding_plots"]

    print("\nCreating Type-based mental embedding plots for Achiever agents...")

    # Filter for achiever agents only
    achiever_mask = agent_labels == "achiever"
    achiever_embeddings = embeddings[achiever_mask]
    achiever_types = type_labels[achiever_mask]

    if len(achiever_embeddings) == 0:
        print("No achiever samples found for mental Type visualization")
        return

    print(
        f"Found {len(achiever_embeddings)} achiever samples for mental Type visualization"
    )
    print(f"Type distribution: {np.unique(achiever_types, return_counts=True)}")

    # Create figure with PCA and t-SNE subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=embedding_plots["pca_figsize"])
    fig.suptitle(
        f"Mental Embeddings by Achiever Type (Experiment {experiment_no})",
        fontsize=16,
    )

    # Type colors and names for achievers
    type_colors = ["lightblue", "darkblue"]  # 0=random/lv0va, 1=strategic/lv1va
    type_names = ["Random Achiever", "Strategic Achiever"]

    # PCA visualization
    if achiever_embeddings.shape[1] > 2:
        print("Computing PCA for achiever mental types...")
        # Check for NaN or infinite values
        if np.any(np.isnan(achiever_embeddings)) or np.any(
            np.isinf(achiever_embeddings)
        ):
            print(
                "Warning: Found NaN or infinite values in achiever embeddings. Cleaning..."
            )
            # Replace NaN with 0 and clip infinite values
            achiever_embeddings = np.nan_to_num(
                achiever_embeddings, nan=0.0, posinf=1e10, neginf=-1e10
            )
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

        ax1.set_title(f"PCA by Achiever Mental Type")
        ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
        ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # t-SNE visualization
    print("Computing t-SNE for achiever mental types...")
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

    ax2.set_title("t-SNE by Achiever Mental Type")
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
                output_dir, f"mental_embeddings_achiever_type_exp{experiment_no}.png"
            ),
            dpi=300,
            bbox_inches="tight",
        )
        print(f"Achiever Mental Type embedding plot saved to {output_dir}")

    plt.close()

def plot_second_belief_embeddings_by_agent(
    embeddings, agent_labels, goal_labels, config, output_dir, experiment_no
):
    """Plot second belief embeddings colored by agent type"""
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))

    # PCA
    # Check for NaN or infinite values
    if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
        print("Warning: Found NaN or infinite values in embeddings. Cleaning...")
        # Replace NaN with 0 and clip infinite values
        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=1e10, neginf=-1e10)

    # Check variance before PCA
    if np.var(embeddings) < 1e-10:
        print("Warning: Embeddings have near-zero variance. PCA may not be meaningful.")
        # Add small amount of noise to prevent PCA issues
        embeddings = embeddings + np.random.normal(0, 1e-6, embeddings.shape)

    pca = PCA(n_components=2, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings)

    agent_colors = ["blue", "orange"]
    agent_names = ["Achiever", "Blocker"]

    legend_handles = []
    for i, (agent_name, color) in enumerate(zip(agent_names, agent_colors)):
        mask = [label == agent_name for label in agent_labels]
        if any(mask):
            scatter = axes[0].scatter(
                embeddings_pca[mask, 0],
                embeddings_pca[mask, 1],
                c=color,
                alpha=0.6,
                s=50,
                label=agent_name,
            )
            legend_handles.append(scatter)

    # Safely handle explained variance (might be NaN)
    var1 = pca.explained_variance_ratio_[0]
    var2 = pca.explained_variance_ratio_[1]
    var1_str = f"{var1:.2%}" if not np.isnan(var1) else "N/A"
    var2_str = f"{var2:.2%}" if not np.isnan(var2) else "N/A"

    axes[0].set_xlabel(f"PC1 ({var1_str})")
    axes[0].set_ylabel(f"PC2 ({var2_str})")
    axes[0].set_title("Second Belief Embeddings - PCA")

    # Only add legend if we have legend handles
    if legend_handles:
        axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # t-SNE
    tsne = TSNE(
        n_components=2, random_state=42, perplexity=min(30, len(embeddings) // 4)
    )
    embeddings_tsne = tsne.fit_transform(embeddings)

    tsne_legend_handles = []
    for i, (agent_name, color) in enumerate(zip(agent_names, agent_colors)):
        mask = [label == agent_name for label in agent_labels]
        if any(mask):
            scatter = axes[1].scatter(
                embeddings_tsne[mask, 0],
                embeddings_tsne[mask, 1],
                c=color,
                alpha=0.6,
                s=50,
                label=agent_name,
            )
            tsne_legend_handles.append(scatter)

    axes[1].set_xlabel("t-SNE Component 1")
    axes[1].set_ylabel("t-SNE Component 2")
    axes[1].set_title("Second Belief Embeddings - t-SNE")

    # Only add legend if we have legend handles
    if tsne_legend_handles:
        axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(
        f"Second Belief Embeddings by Agent Type (Experiment {experiment_no})",
        fontsize=16,
    )
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir, f"second_belief_embeddings_by_agent_exp{experiment_no}.png"
        ),
        dpi=300,
        bbox_inches="tight",
    )
    print(f"Second belief agent embedding plot saved to {output_dir}")
    plt.close()

def plot_second_belief_embeddings_by_goal(
    embeddings, agent_labels, goal_labels, config, output_dir, experiment_no
):
    """Plot second belief embeddings colored by goal type"""
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))

    # PCA
    # Check for NaN or infinite values
    if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
        print("Warning: Found NaN or infinite values in embeddings. Cleaning...")
        # Replace NaN with 0 and clip infinite values
        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=1e10, neginf=-1e10)

    # Check variance before PCA
    if np.var(embeddings) < 1e-10:
        print("Warning: Embeddings have near-zero variance. PCA may not be meaningful.")
        # Add small amount of noise to prevent PCA issues
        embeddings = embeddings + np.random.normal(0, 1e-6, embeddings.shape)

    pca = PCA(n_components=2, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings)

    goal_colors = ["red", "green", "blue", "yellow"]
    goal_names = ["Red", "Green", "Blue", "Yellow"]

    legend_handles = []
    for i, (goal_name, color) in enumerate(zip(goal_names, goal_colors)):
        mask = [label == goal_name for label in goal_labels]
        if any(mask):
            scatter = axes[0].scatter(
                embeddings_pca[mask, 0],
                embeddings_pca[mask, 1],
                c=color,
                alpha=0.6,
                s=50,
                label=goal_name,
            )
            legend_handles.append(scatter)

    # Safely handle explained variance (might be NaN)
    var1 = pca.explained_variance_ratio_[0]
    var2 = pca.explained_variance_ratio_[1]
    var1_str = f"{var1:.2%}" if not np.isnan(var1) else "N/A"
    var2_str = f"{var2:.2%}" if not np.isnan(var2) else "N/A"

    axes[0].set_xlabel(f"PC1 ({var1_str})")
    axes[0].set_ylabel(f"PC2 ({var2_str})")
    axes[0].set_title("Second Belief Embeddings - PCA")

    # Only add legend if we have legend handles
    if legend_handles:
        axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # t-SNE
    tsne = TSNE(
        n_components=2, random_state=42, perplexity=min(30, len(embeddings) // 4)
    )
    embeddings_tsne = tsne.fit_transform(embeddings)

    tsne_legend_handles = []
    for i, (goal_name, color) in enumerate(zip(goal_names, goal_colors)):
        mask = [label == goal_name for label in goal_labels]
        if any(mask):
            scatter = axes[1].scatter(
                embeddings_tsne[mask, 0],
                embeddings_tsne[mask, 1],
                c=color,
                alpha=0.6,
                s=50,
                label=goal_name,
            )
            tsne_legend_handles.append(scatter)

    axes[1].set_xlabel("t-SNE Component 1")
    axes[1].set_ylabel("t-SNE Component 2")
    axes[1].set_title("Second Belief Embeddings - t-SNE")

    # Only add legend if we have legend handles
    if tsne_legend_handles:
        axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(
        f"Second Belief Embeddings by Goal Type (Experiment {experiment_no})",
        fontsize=16,
    )
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir, f"second_belief_embeddings_by_goal_exp{experiment_no}.png"
        ),
        dpi=300,
        bbox_inches="tight",
    )
    print(f"Second belief goal embedding plot saved to {output_dir}")
    plt.close()


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
        raise ValueError(
            "config is required: the shared plots cannot construct an "
            "experiment Config. Pass the experiment's config object."
        )

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

    title_prefix = (
        "Single-Agent"
        if config and config.is_single_agent_mode()
        else "AchieverBlocker"
    )
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
    title_prefix = (
        "Single-Agent"
        if config and config.is_single_agent_mode()
        else "AchieverBlocker"
    )
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
        raise ValueError(
            "config is required: the shared plots cannot construct an "
            "experiment Config. Pass the experiment's config object."
        )

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

    title_prefix = (
        "Single-Agent"
        if config and config.is_single_agent_mode()
        else "AchieverBlocker"
    )
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
    title_prefix = (
        "Single-Agent"
        if config and config.is_single_agent_mode()
        else "AchieverBlocker"
    )
    print(f"\n{title_prefix} Confusion Matrix Statistics (Experiment {experiment_no}):")
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
        raise ValueError(
            "config is required: the shared plots cannot construct an "
            "experiment Config. Pass the experiment's config object."
        )

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
    title_prefix = (
        "Single-Agent"
        if config and config.is_single_agent_mode()
        else "AchieverBlocker"
    )
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
