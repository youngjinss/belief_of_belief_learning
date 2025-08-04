import os
import json
import pickle
from typing import Optional, Dict, Any, Tuple, List

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

from config import Config
from utils import prepare_data_for_training, generate_past_episodes_from_batch
from data_generation import DataGenerator as DataReader, DataGenerator
from utils import set_seed, load_chunked_data_for_training, load_test_data_all_combinations, combine_all_combinations_data

# Set seed using Config default value
config = Config()
set_seed(config.seed)

# Remove circular import - load_model will be imported locally when needed

"""
Visualization tools for AchieverBlocker ToMnet experiment (exp7)
Supports both single-agent (KeyDoor) and multi-agent (AchieverBlocker) environments
"""

# No caching - keep it simple and reliable


# Data loading functions moved to utils.py


# ============================================================================
# UNIFIED EMBEDDING EXTRACTION SYSTEM
# ============================================================================

class BaseEmbeddingExtractor:
    """
    Unified embedding extractor that uses ToMnet.forward() for all embedding types.
    Eliminates code duplication with configuration-driven approach.
    """
    
    def __init__(self, model, device, config, batch_size=32):
        """
        Initialize the embedding extractor.
        
        Args:
            model: Trained ToMnet model
            device: Computing device
            config: Configuration object
            batch_size: Batch size for processing
        """
        self.model = model
        self.device = device
        self.config = config
        self.batch_size = batch_size
        self.channels_in = config.get_model_config().get("channels_in", 10)
        
        # Validate model capabilities at initialization
        self._validate_model_capabilities()
    
    def _validate_model_capabilities(self):
        """Validate model has required capabilities. Raise error if not."""
        if not hasattr(self.model, 'forward'):
            raise AttributeError("Model must have forward() method")
        if not hasattr(self.model, 'get_character_embedding'):
            raise AttributeError("Model must have get_character_embedding() method")
        
    def extract_embeddings(self, processed_data, embedding_type="character", n_samples=None):
        """
        Extract embeddings of specified type using unified approach.
        
        Args:
            processed_data: Dictionary containing processed tensors
            embedding_type: Type of embedding to extract ("character", "mental", "second_belief")
            n_samples: Number of samples to process (None for all)
            
        Returns:
            tuple: (embeddings_array, agent_labels, goal_labels, type_labels)
        """
        # Validate embedding type
        if embedding_type not in ["character", "mental", "second_belief"]:
            raise ValueError(f"Invalid embedding_type: {embedding_type}. Must be one of: character, mental, second_belief")
            
        # Validate model capabilities for specific embedding types
        if embedding_type == "mental" and not getattr(self.model, 'use_mentalnet', False):
            raise ValueError(f"Model does not support MentalNet. Cannot extract {embedding_type} embeddings.")
            
        if embedding_type == "second_belief" and not getattr(self.model, 'use_second_belief', False):
            raise ValueError(f"Model does not support SecondBeliefNet. Cannot extract {embedding_type} embeddings.")
        
        # Prepare data tensors
        data_tensors = self._prepare_data_tensors(processed_data, n_samples)
            
        # Extract labels
        labels = self._extract_labels(data_tensors)
        
        # Extract embeddings using unified batch processing
        embeddings = self._extract_embeddings_batch(data_tensors, embedding_type)
        
        return embeddings, labels["agent_labels"], labels["goal_labels"], labels["type_labels"]
    
    def _prepare_data_tensors(self, processed_data, n_samples=None):
        """
        Prepare and validate data tensors for processing.
        
        Args:
            processed_data: Dictionary containing processed tensors
            n_samples: Number of samples to limit to
            
        Returns:
            dict: Prepared data tensors
        """
        # Validate required keys
        required_keys = ["trajectories", "goals", "goal_ranks", "agents", "types"]
        for key in required_keys:
            if key not in processed_data:
                raise KeyError(f"Missing required key in processed_data: {key}")
        
        # Handle both chunked and original tensor formats
        if isinstance(processed_data["trajectories"], torch.Tensor):
            # Original tensor format
            trajectories_tensor = processed_data["trajectories"]
            goals_tensor = processed_data["goals"]
            goal_ranks_tensor = processed_data["goal_ranks"]
            agents_tensor = processed_data["agents"]
            types_tensor = processed_data["types"]
            actions_tensor = processed_data.get("actions")  # May not exist for character embeddings
        else:
            # Chunked format (numpy arrays)
            trajectories_tensor = torch.from_numpy(processed_data["trajectories"]).float()
            goals_tensor = torch.from_numpy(processed_data["goals"]).float()
            goal_ranks_tensor = torch.from_numpy(processed_data["goal_ranks"]).long()
            agents_tensor = torch.from_numpy(processed_data["agents"]).long()
            types_tensor = torch.from_numpy(processed_data["types"]).long()
            actions_tensor = torch.from_numpy(processed_data["actions"]).long() if "actions" in processed_data else None
        
        # Validate tensor shapes are consistent
        batch_size = len(agents_tensor)
        if len(trajectories_tensor) != batch_size or len(goals_tensor) != batch_size:
            raise ValueError(f"Inconsistent tensor shapes: trajectories={len(trajectories_tensor)}, goals={len(goals_tensor)}, agents={batch_size}")
        
        # Apply sample limiting if specified
        if n_samples is not None:
            n_total = len(agents_tensor)
            indices = np.random.choice(n_total, min(n_samples, n_total), replace=False)
            trajectories_tensor = trajectories_tensor[indices]
            goals_tensor = goals_tensor[indices]
            goal_ranks_tensor = goal_ranks_tensor[indices]
            agents_tensor = agents_tensor[indices]
            types_tensor = types_tensor[indices]
            if actions_tensor is not None:
                actions_tensor = actions_tensor[indices]
        
        return {
            "trajectories": trajectories_tensor,
            "goals": goals_tensor,
            "goal_ranks": goal_ranks_tensor,
            "agents": agents_tensor,
            "types": types_tensor,
            "actions": actions_tensor,
            "opponent_trajectories": processed_data.get("opponent_trajectories"),
            "opponent_actions": processed_data.get("opponent_actions")
        }
    
    def _extract_labels(self, data_tensors):
        """
        Extract agent, goal, and type labels from data tensors.
        
        Args:
            data_tensors: Dictionary of prepared tensors
            
        Returns:
            dict: Dictionary containing extracted labels
        """
        # Extract agent labels (0=achiever, 1=blocker)
        agent_indices = data_tensors["agents"]
        if hasattr(agent_indices, 'cpu'):
            agent_indices = agent_indices.cpu()
        agent_labels = np.array(
            ["achiever" if idx == 0 else "blocker" for idx in agent_indices]
        )
        
        # Extract goal labels from one-hot encoded goals
        goals_tensor = data_tensors["goals"]
        if hasattr(goals_tensor, 'cpu'):
            goals_tensor = goals_tensor.cpu().numpy()
        else:
            goals_tensor = np.asarray(goals_tensor)
        goal_labels = np.argmax(goals_tensor, axis=1)
        
        # Extract type labels
        type_labels = data_tensors["types"]
        if hasattr(type_labels, 'cpu'):
            type_labels = type_labels.cpu().numpy()
        else:
            type_labels = np.asarray(type_labels)
        type_labels = type_labels.astype(int)
        
        return {
            "agent_labels": agent_labels,
            "goal_labels": goal_labels,
            "type_labels": type_labels
        }
    
    def _extract_embeddings_batch(self, data_tensors, embedding_type):
        """
        Extract embeddings using batch processing with unified ToMnet.forward() approach.
        
        Args:
            data_tensors: Dictionary of prepared tensors
            embedding_type: Type of embedding to extract
            
        Returns:
            numpy.ndarray: Extracted embeddings
        """
        self.model.eval()
        embeddings = []
        
        trajectories = data_tensors["trajectories"]
        goal_ranks = data_tensors["goal_ranks"]
        agents = data_tensors["agents"]
        actions = data_tensors["actions"]
        
        print(f"Extracting {embedding_type} embeddings for {len(trajectories)} samples...")
        
        # Validate actions tensor for mental and second_belief embeddings
        if embedding_type in ["mental", "second_belief"] and actions is None:
            raise ValueError(f"Actions tensor required for {embedding_type} embeddings but not found")
        
        # Get n_past configuration
        n_past_config = self.config.get_n_past_evaluation_config()
        
        with torch.no_grad():
            for start_idx in range(0, len(trajectories), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(trajectories))
                current_batch_size = end_idx - start_idx
                
                if start_idx % (self.batch_size * 10) == 0:
                    print(f"Processing batch {start_idx//self.batch_size + 1}/{(len(trajectories) + self.batch_size - 1)//self.batch_size}")
                
                # Get batch tensors
                batch_trajectories = trajectories[start_idx:end_idx].to(self.device)
                batch_goal_ranks = goal_ranks[start_idx:end_idx].to(self.device)
                batch_agents = agents[start_idx:end_idx].to(self.device)
                
                # Handle character embeddings (no actions needed)
                if embedding_type == "character":
                    batch_embeddings = self._extract_character_embeddings_batch(
                        batch_trajectories, batch_goal_ranks, batch_agents, 
                        current_batch_size, n_past_config
                    )
                else:
                    # Mental and second belief embeddings use unified ToMnet.forward()
                    batch_actions = actions[start_idx:end_idx].to(self.device)
                    
                    # Handle opponent data for second belief
                    batch_opponent_trajectories = None
                    batch_opponent_actions = None
                    if embedding_type == "second_belief":
                        opponent_trajectories = data_tensors.get("opponent_trajectories")
                        opponent_actions = data_tensors.get("opponent_actions")
                        if opponent_trajectories is not None and start_idx < len(opponent_trajectories):
                            batch_opp_traj = opponent_trajectories[start_idx:end_idx]
                            if isinstance(batch_opp_traj, np.ndarray):
                                batch_opp_traj = torch.from_numpy(batch_opp_traj).float()
                            batch_opponent_trajectories = batch_opp_traj.to(self.device)
                            if opponent_actions is not None and start_idx < len(opponent_actions):
                                batch_opp_act = opponent_actions[start_idx:end_idx]
                                if isinstance(batch_opp_act, np.ndarray):
                                    batch_opp_act = torch.from_numpy(batch_opp_act).long()
                                batch_opponent_actions = batch_opp_act.to(self.device)
                    
                    batch_embeddings = self._extract_forward_based_embeddings_batch(
                        batch_trajectories, batch_actions, batch_goal_ranks, batch_agents,
                        batch_opponent_trajectories, batch_opponent_actions,
                        current_batch_size, n_past_config, embedding_type
                    )
                
                if batch_embeddings is None:
                    raise RuntimeError(f"Failed to extract embeddings for batch starting at {start_idx}")
                    
                for emb in batch_embeddings:
                    embeddings.append(emb.flatten())
        
        embeddings = np.array(embeddings)
        print(f"Extracted {embedding_type} embeddings shape: {embeddings.shape}")
        return embeddings
    
    def _extract_character_embeddings_batch(self, batch_trajectories, batch_goal_ranks, 
                                          batch_agents, current_batch_size, n_past_config):
        """Extract character embeddings using model.get_character_embedding()"""
        # Generate past episodes
        past_episodes = generate_past_episodes_from_batch(
            self_states=batch_trajectories,
            goal_ranks=batch_goal_ranks,
            agents=batch_agents,
            batch_size=current_batch_size,
            n_past_min=n_past_config["n_past_min"],
            n_past_max=n_past_config["n_past_max"],
            max_n_past=n_past_config["n_past_max"],
            rank_threshold=self.config.get_data_config().get("rank_threshold", 4),
        )
        
        # Extract character embeddings
        char_embeddings = self.model.get_character_embedding(past_episodes)
        return char_embeddings.cpu().numpy()
    
    def _extract_forward_based_embeddings_batch(self, batch_trajectories, batch_actions,
                                              batch_goal_ranks, batch_agents,
                                              batch_opponent_trajectories, batch_opponent_actions,
                                              current_batch_size, n_past_config, embedding_type):
        """Extract mental or second belief embeddings using unified ToMNet.forward() approach."""
        # Generate past episodes
        past_episodes = generate_past_episodes_from_batch(
            self_states=batch_trajectories,
            goal_ranks=batch_goal_ranks,
            agents=batch_agents,
            batch_size=current_batch_size,
            n_past_min=n_past_config["n_past_min"],
            n_past_max=n_past_config["n_past_max"],
            max_n_past=n_past_config["n_past_max"],
            rank_threshold=self.config.get_data_config().get("rank_threshold", 4),
        )
        
        # Prepare current state (last timestep) - use full channels
        current_state = batch_trajectories[:, -1, :]
        
        # Apply temporal masking to actions (mask target action at last position)
        masked_actions = batch_actions.clone()
        masked_actions[:, -1] = -1  # Mask the target action so model can't see it
        
        # Call ToMnet.forward() to get all embeddings
        forward_kwargs = {
            "past_trajectories": past_episodes,
            "self_states": batch_trajectories,
            "self_actions": masked_actions,
            "current_state": current_state
        }
        
        # Add opponent data for second belief
        if embedding_type == "second_belief" and batch_opponent_trajectories is not None:
            forward_kwargs["oppo_states"] = batch_opponent_trajectories
            # Apply temporal masking to opponent actions as well
            if batch_opponent_actions is not None:
                masked_opponent_actions = batch_opponent_actions.clone()
                masked_opponent_actions[:, -1] = -1  # Mask opponent target action
                forward_kwargs["oppo_actions"] = masked_opponent_actions
        
        outputs = self.model.forward(**forward_kwargs)
        
        # Extract the requested embedding type using configuration-driven key mapping
        embedding_key_map = {
            "mental": "mental_state",
            "second_belief": "second_belief"
        }
        
        embedding_key = embedding_key_map[embedding_type]
        if embedding_key not in outputs:
            raise KeyError(f"Expected key '{embedding_key}' not found in model outputs. Available keys: {list(outputs.keys())}")
            
        embedding_tensor = outputs[embedding_key]
        if embedding_tensor is None:
            raise ValueError(f"Model output '{embedding_key}' is None")
        
        # Process embeddings based on type
        if embedding_type == "mental":
            # Convert spatial embeddings to vectors using global average pooling
            if embedding_tensor.dim() == 4:  # Spatial mental state
                embedding_vectors = torch.mean(embedding_tensor, dim=(2, 3))  # Global average pooling
            else:  # Already vectorized
                embedding_vectors = embedding_tensor
            return embedding_vectors.cpu().numpy()
            
        elif embedding_type == "second_belief":
            return embedding_tensor.cpu().numpy()


class EmbeddingExtractor(BaseEmbeddingExtractor):
    """
    Main embedding extractor class - inherits all functionality from BaseEmbeddingExtractor.
    This class maintains compatibility with existing code while providing the unified approach.
    """
    pass


def prepare_common_test_data(config, n_samples=None):
    """
    Common helper function to load and prepare test data from all combinations.
    
    Args:
        config: Configuration object
        n_samples: Number of samples to limit to (None for all)
        
    Returns:
        tuple: (processed_data, agent_labels, goal_labels, type_labels)
    """
    print("Loading processed test data from all combinations...")
    
    # Get base data directory from config
    test_data_dir = os.path.join(config.save_dir, config.get_env_name())
    
    # Load test data for all combinations efficiently
    all_test_data = load_test_data_all_combinations(config, test_data_dir_base=test_data_dir)
    
    # Combine data from all combinations
    processed_data = combine_all_combinations_data(all_test_data)
    
    print(f"Loaded combined test data with {processed_data['trajectories'].shape[0]} samples")
    print(f"Achiever types: {list(config.achiever_types.keys())}")
    if config.is_single_agent_mode():
        print(f"Single-agent mode: No blockers")
    else:
        print(f"Blocker types: {list(config.blocker_types.keys())}")
    
    return processed_data


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

            # Handle missing detailed predictions - using overall accuracy distributed across actions
            if n_past == n_past_values[0]:
                print("N_past results don't contain detailed predictions/targets.")
                print("Using overall accuracy distributed across all actions for heatmap.")

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

    # Use unified embedding extraction system
    processed_data = prepare_common_test_data(config, n_samples)
    
    # Create unified embedding extractor
    extractor = EmbeddingExtractor(model, device, config, batch_size=32)
    
    # Extract character embeddings using unified approach
    embeddings, agent_labels, goal_labels, type_labels = extractor.extract_embeddings(
        processed_data, embedding_type="character", n_samples=n_samples
    )
    
    print(f"Agent distribution: {np.unique(agent_labels, return_counts=True)}")
    print(f"Goal distribution: {np.unique(goal_labels, return_counts=True)}")
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
    
    # Ensure labels are numpy arrays
    if hasattr(agent_labels, 'cpu'):
        agent_labels = agent_labels.cpu().numpy()
    if hasattr(goal_labels, 'cpu'):
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
    if hasattr(goal_labels, 'cpu'):
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
    if hasattr(agent_labels, 'cpu'):
        agent_labels = agent_labels.cpu().numpy()
    if hasattr(goal_labels, 'cpu'):
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
    
    # Ensure labels are numpy arrays
    if hasattr(agent_labels, 'cpu'):
        agent_labels = agent_labels.cpu().numpy()
    if hasattr(goal_labels, 'cpu'):
        goal_labels = goal_labels.cpu().numpy()
    if hasattr(type_labels, 'cpu'):
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


# Old extract_mental_embeddings_from_batch function removed - now handled by unified EmbeddingExtractor


def plot_mental_embeddings(
    model,
    test_loader,
    device,
    output_dir,
    config=None,
    experiment_no=None,
    n_samples=None,
):
    """
    Plot mental embeddings using PCA and t-SNE with separate agent analysis
    Creates visualizations colored by:
    1. Agent-based coloring (achiever vs blocker) 
    2. Goal-based coloring (red, green, blue, yellow)
    3. Type-based coloring for achievers and blockers separately
    
    Args:
        model: Trained ToMnet model with MentalNet
        test_loader: Test data loader
        device: Computing device
        output_dir: Directory to save plots
        config: Configuration object containing experiment settings
        experiment_no: Experiment number (defaults to config.experiment_no)
        n_samples: Number of samples to visualize (None for all samples)
    """
    if not model.use_mentalnet:
        print("Warning: Model does not use MentalNet. Skipping mental embedding visualization.")
        return
        
    if config is None:
        config = Config()
        
    if experiment_no is None:
        experiment_no = config.experiment_no
        
    print("Creating mental embedding visualizations...")
    print(f"Model has MentalNet: {model.use_mentalnet}")
    
    # Use the unified data preparation helper
    processed_data = prepare_common_test_data(config, n_samples)
    
    # Create unified embedding extractor
    extractor = EmbeddingExtractor(model, device, config, batch_size=32)
    
    # Extract mental embeddings using unified approach
    embeddings, agent_labels, goal_labels, type_labels = extractor.extract_embeddings(
        processed_data, embedding_type="mental", n_samples=n_samples
    )
    
    if embeddings is None or len(embeddings) == 0:
        print("No mental embeddings to visualize!")
        return
        
    print(f"Mental embeddings shape: {embeddings.shape}")
    
    # Create plots based on mode
    if config.is_single_agent_mode():
        # Single-agent mode: only create goal-based and achiever type embeddings
        _plot_mental_goal_based_embeddings(
            embeddings, agent_labels, goal_labels, config, output_dir, experiment_no
        )
        _plot_mental_type_based_embeddings_for_achiever(
            embeddings, agent_labels, goal_labels, type_labels, config, output_dir, experiment_no
        )
    else:
        # Multi-agent mode: create all types of plots
        _plot_mental_agent_based_embeddings(
            embeddings, agent_labels, goal_labels, config, output_dir, experiment_no
        )
        _plot_mental_goal_based_embeddings(
            embeddings, agent_labels, goal_labels, config, output_dir, experiment_no
        )
        _plot_mental_separate_agent_goal_embeddings(
            embeddings, agent_labels, goal_labels, config, output_dir, experiment_no
        )
        _plot_mental_type_based_embeddings_for_blockers(
            embeddings, agent_labels, goal_labels, type_labels, config, output_dir, experiment_no
        )
        _plot_mental_type_based_embeddings_for_achiever(
            embeddings, agent_labels, goal_labels, type_labels, config, output_dir, experiment_no
        )


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
        os.path.join(
            output_dir, f"mental_embeddings_by_agent_exp{experiment_no}.png"
        ),
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
        os.path.join(
            output_dir, f"mental_embeddings_by_goal_exp{experiment_no}.png"
        ),
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
            output_dir, f"mental_embeddings_separate_agents_exp{experiment_no}.png"
        ),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"All mental embedding plots saved to {output_dir}")


def _plot_mental_type_based_embeddings_for_blockers(
    embeddings, agent_labels, goal_labels, type_labels, config, output_dir, experiment_no
):
    """Plot mental embeddings colored by Type, constrained to Blocker agents only"""
    
    # Ensure labels are numpy arrays
    if hasattr(agent_labels, 'cpu'):
        agent_labels = agent_labels.cpu().numpy()
    if hasattr(goal_labels, 'cpu'):
        goal_labels = goal_labels.cpu().numpy()
    if hasattr(type_labels, 'cpu'):
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

    print(f"Found {len(blocker_embeddings)} blocker samples for mental Type visualization")
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
    embeddings, agent_labels, goal_labels, type_labels, config, output_dir, experiment_no
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

    print(f"Found {len(achiever_embeddings)} achiever samples for mental Type visualization")
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


# Old extract_second_belief_embeddings_from_batch function removed - now handled by unified EmbeddingExtractor


def plot_second_belief_embeddings(
    model,
    test_loader,
    device,
    output_dir: str,
    config=None,
    experiment_no: int = None,
    n_samples: int = None,
) -> bool:
    """
    Plot second belief embeddings (e_opp2) using PCA and t-SNE visualization.
    
    This function extracts second-order belief embeddings from a trained ToMnet model
    and creates visualization plots showing how these embeddings cluster by agent type
    and goal preferences. The visualizations help understand how the model represents
    what agents believe about others' beliefs.
    
    Args:
        model: Trained ToMnet model with SecondBeliefNet capability
        test_loader: DataLoader for test data containing agent trajectories
        device: Computing device (CPU or CUDA)
        output_dir: Directory to save the generated plots
        config: Configuration object containing model and visualization parameters
        experiment_no: Experiment number for plot titles and filenames
        n_samples: Maximum number of samples to use (None for all available)
        
    Returns:
        bool: True if visualization was successful, False otherwise
    """
    # Use the unified data preparation helper
    processed_data = prepare_common_test_data(config, n_samples)
    
    # Create unified embedding extractor
    extractor = EmbeddingExtractor(model, device, config, batch_size=32)
    
    # Extract second belief embeddings using unified approach
    embeddings, agent_labels, goal_labels, type_labels = extractor.extract_embeddings(
        processed_data, embedding_type="second_belief", n_samples=n_samples
    )
    
    if embeddings is None:
        print("Failed to extract second belief embeddings")
        return
    
    print(f"Second belief embeddings shape: {embeddings.shape}")
    
    # Create visualizations
    plot_second_belief_embeddings_by_agent(embeddings, agent_labels, goal_labels, config, output_dir, experiment_no)
    plot_second_belief_embeddings_by_goal(embeddings, agent_labels, goal_labels, config, output_dir, experiment_no)


def plot_second_belief_embeddings_by_agent(embeddings, agent_labels, goal_labels, config, output_dir, experiment_no):
    """Plot second belief embeddings colored by agent type"""
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    
    # PCA
    pca = PCA(n_components=2, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings)
    
    agent_colors = ["blue", "orange"]
    agent_names = ["Achiever", "Blocker"]
    
    for i, (agent_name, color) in enumerate(zip(agent_names, agent_colors)):
        mask = [label == agent_name for label in agent_labels]
        if any(mask):
            axes[0].scatter(
                embeddings_pca[mask, 0], embeddings_pca[mask, 1],
                c=color, alpha=0.6, s=50, label=agent_name
            )
    
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    axes[0].set_title('Second Belief Embeddings - PCA')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)//4))
    embeddings_tsne = tsne.fit_transform(embeddings)
    
    for i, (agent_name, color) in enumerate(zip(agent_names, agent_colors)):
        mask = [label == agent_name for label in agent_labels]
        if any(mask):
            axes[1].scatter(
                embeddings_tsne[mask, 0], embeddings_tsne[mask, 1],
                c=color, alpha=0.6, s=50, label=agent_name
            )
    
    axes[1].set_xlabel('t-SNE Component 1')
    axes[1].set_ylabel('t-SNE Component 2')
    axes[1].set_title('Second Belief Embeddings - t-SNE')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(f"Second Belief Embeddings by Agent Type (Experiment {experiment_no})", fontsize=16)
    plt.tight_layout()
    
    plt.savefig(
        os.path.join(output_dir, f"second_belief_embeddings_by_agent_exp{experiment_no}.png"),
        dpi=300, bbox_inches="tight"
    )
    print(f"Second belief agent embedding plot saved to {output_dir}")
    plt.close()


def plot_second_belief_embeddings_by_goal(embeddings, agent_labels, goal_labels, config, output_dir, experiment_no):
    """Plot second belief embeddings colored by goal type"""
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    
    # PCA
    pca = PCA(n_components=2, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings)
    
    goal_colors = ["red", "green", "blue", "yellow"]
    goal_names = ["Red", "Green", "Blue", "Yellow"]
    
    for i, (goal_name, color) in enumerate(zip(goal_names, goal_colors)):
        mask = [label == goal_name for label in goal_labels]
        if any(mask):
            axes[0].scatter(
                embeddings_pca[mask, 0], embeddings_pca[mask, 1],
                c=color, alpha=0.6, s=50, label=goal_name
            )
    
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    axes[0].set_title('Second Belief Embeddings - PCA')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)//4))
    embeddings_tsne = tsne.fit_transform(embeddings)
    
    for i, (goal_name, color) in enumerate(zip(goal_names, goal_colors)):
        mask = [label == goal_name for label in goal_labels]
        if any(mask):
            axes[1].scatter(
                embeddings_tsne[mask, 0], embeddings_tsne[mask, 1],
                c=color, alpha=0.6, s=50, label=goal_name
            )
    
    axes[1].set_xlabel('t-SNE Component 1')
    axes[1].set_ylabel('t-SNE Component 2')
    axes[1].set_title('Second Belief Embeddings - t-SNE')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(f"Second Belief Embeddings by Goal Type (Experiment {experiment_no})", fontsize=16)
    plt.tight_layout()
    
    plt.savefig(
        os.path.join(output_dir, f"second_belief_embeddings_by_goal_exp{experiment_no}.png"),
        dpi=300, bbox_inches="tight"
    )
    print(f"Second belief goal embedding plot saved to {output_dir}")
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
        "--experiment_no", type=int, default=7, help="Experiment number"
    )
    parser.add_argument(
        "--plot_type",
        type=str,
        choices=["training", "confusion", "likelihood", "embeddings", "mental_embeddings", "second_belief_embeddings", "n_past", "all"],
        default="all",
        help="Type of plot to create",
    )

    args = parser.parse_args()

    config = Config()

    # Override config with command line arguments if specified
    if args.config_override:
        config.update_from_args(args)

    results_dir = args.result_dir or getattr(config, "result_dir", "results/exp7")
    plot_dir = args.plot_dir or getattr(config, "plot_dir", "results/exp7/plots")
    experiment_no = args.experiment_no or getattr(config, "experiment_no", 7)

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
            
            # Get base data directory from config
            test_data_dir_base = os.path.join(config.save_dir, config.get_env_name())
            
            # Load test data for all combinations efficiently (same as evaluate.py)
            all_test_data = load_test_data_all_combinations(config, test_data_dir_base=test_data_dir_base)
            # Combine data from all combinations
            test_data = combine_all_combinations_data(all_test_data)
            print(f"Successfully loaded test data from all combinations: {test_data['trajectories'].shape[0]} samples")
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
            print(f"Model file not found: {model_path}")

    # Plot mental embeddings
    if args.plot_type in ["mental_embeddings", "embeddings", "all"]:
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
            
            # Get base data directory from config
            test_data_dir_base = os.path.join(config.save_dir, config.get_env_name())
            
            # Load test data for all combinations efficiently (same as evaluate.py)
            all_test_data = load_test_data_all_combinations(config, test_data_dir_base=test_data_dir_base)
            # Combine data from all combinations
            test_data = combine_all_combinations_data(all_test_data)
            print(f"Successfully loaded test data from all combinations: {test_data['trajectories'].shape[0]} samples")
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

            # Create mental embeddings plot
            plot_mental_embeddings(
                model,
                test_loader,
                device,
                plot_dir,
                config,
                experiment_no,
                n_samples=None,
            )
            print("Mental embedding visualization completed!")
        else:
            print(f"Model file not found: {model_path}")

    # Plot second belief embeddings
    if args.plot_type in ["second_belief_embeddings", "embeddings", "all"]:
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
            
            # Get base data directory from config
            test_data_dir_base = os.path.join(config.save_dir, config.get_env_name())
            
            # Load test data for all combinations efficiently (same as evaluate.py)
            all_test_data = load_test_data_all_combinations(config, test_data_dir_base=test_data_dir_base)
            # Combine data from all combinations
            test_data = combine_all_combinations_data(all_test_data)
            print(f"Successfully loaded test data from all combinations: {test_data['trajectories'].shape[0]} samples")
            # Convert numpy arrays to tensors for TensorDataset (same as evaluate.py fix)
            test_tensors = {
                key: torch.from_numpy(data) if isinstance(data, np.ndarray) else torch.tensor(data)
                for key, data in test_data.items()
            }
            
            # Create test dataset with opponent trajectory data if available
            dataset_items = [
                test_tensors["trajectories"],
                test_tensors["actions"],
                test_tensors["goals"],
                test_tensors["goal_ranks"],
                test_tensors["agents"],
                test_tensors["types"],
                test_tensors["consumption_labels"],
                test_tensors["sr_labels"],
            ]
            
            # Add opponent trajectory data if available
            if "opponent_recent_trajectory" in test_tensors:
                dataset_items.extend([
                    test_tensors["opponent_recent_trajectory"],
                    test_tensors["opponent_actions"]
                ])
            
            test_dataset = TensorDataset(*dataset_items)
            
            # Create custom data loader that provides dictionary format
            class SecondBeliefDataLoader:
                def __init__(self, dataset, batch_size=32):
                    self.dataset = dataset
                    self.batch_size = batch_size
                    self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
                
                def __iter__(self):
                    for batch in self.dataloader:
                        batch_dict = {
                            'trajectory': batch[0],
                            'actions': batch[1],
                            'goal': batch[2],
                            'goal_ranks': batch[3],
                            'agent': batch[4],
                            'type': batch[5],
                            'consumption_labels': batch[6],
                            'sr_labels': batch[7],
                        }
                        
                        # Add opponent data if available
                        if len(batch) > 8:
                            batch_dict['opponent_recent_trajectory'] = batch[8]
                            batch_dict['opponent_actions'] = batch[9]
                        else:
                            batch_dict['opponent_recent_trajectory'] = None
                            batch_dict['opponent_actions'] = None
                        
                        yield batch_dict

            test_loader = SecondBeliefDataLoader(test_dataset, batch_size=32)

            # Create second belief embeddings plot
            plot_second_belief_embeddings(
                model,
                test_loader,
                device,
                plot_dir,
                config,
                experiment_no,
                n_samples=None,
            )
            print("Second belief embedding visualization completed!")
        else:
            print(f"Model file not found: {model_path}")

    print("Visualization completed!")


