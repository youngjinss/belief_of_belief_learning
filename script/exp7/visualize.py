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


# Moved into the shared core (AST-identical in exp7 and exp8); re-exported so
# this module's own callers keep working unchanged.
from beliefrl.viz.plots import (  # noqa: F401
    _plot_agent_based_embeddings,
    _plot_goal_based_embeddings,
    _plot_mental_agent_based_embeddings,
    _plot_mental_goal_based_embeddings,
    _plot_mental_separate_agent_goal_embeddings,
    _plot_mental_type_based_embeddings_for_achiever,
    _plot_mental_type_based_embeddings_for_blockers,
    _plot_separate_agent_goal_embeddings,
    _plot_type_based_embeddings_for_achiever,
    _plot_type_based_embeddings_for_blockers,
    plot_accuracy_by_n_past,
    plot_accuracy_heatmap_by_n_past,
    plot_second_belief_embeddings_by_agent,
    plot_second_belief_embeddings_by_goal,
)



from beliefrl.viz.plots import (  # noqa: F401
    plot_action_likelihood,
    plot_confusion_matrix,
    plot_training_curves,
)


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
        required_keys = ["self_states", "goals", "goal_ranks", "agents", "types"]
        for key in required_keys:
            if key not in processed_data:
                raise KeyError(f"Missing required key in processed_data: {key}")
        
        # Handle both chunked and original tensor formats
        if isinstance(processed_data["self_states"], torch.Tensor):
            # Original tensor format
            trajectories_tensor = processed_data["self_states"]
            goals_tensor = processed_data["goals"]
            goal_ranks_tensor = processed_data["goal_ranks"]
            agents_tensor = processed_data["agents"]
            types_tensor = processed_data["types"]
            actions_tensor = processed_data.get("self_actions")  # May not exist for character embeddings
        else:
            # Chunked format (numpy arrays)
            trajectories_tensor = torch.from_numpy(processed_data["self_states"]).float()
            goals_tensor = torch.from_numpy(processed_data["goals"]).float()
            goal_ranks_tensor = torch.from_numpy(processed_data["goal_ranks"]).long()
            agents_tensor = torch.from_numpy(processed_data["agents"]).long()
            types_tensor = torch.from_numpy(processed_data["types"]).long()
            actions_tensor = torch.from_numpy(processed_data["self_actions"]).long() if "self_actions" in processed_data else None
        
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
            "self_states": trajectories_tensor,
            "goals": goals_tensor,
            "goal_ranks": goal_ranks_tensor,
            "agents": agents_tensor,
            "types": types_tensor,
            "self_actions": actions_tensor,
            "oppo_states": processed_data.get("oppo_states"),
            "oppo_actions": processed_data.get("oppo_actions")
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
        
        trajectories = data_tensors["self_states"]
        goal_ranks = data_tensors["goal_ranks"]
        agents = data_tensors["agents"]
        actions = data_tensors["self_actions"]
        
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
                        opponent_trajectories = data_tensors.get("oppo_states")
                        opponent_actions = data_tensors.get("oppo_actions")
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
    
    print(f"Loaded combined test data with {processed_data['self_states'].shape[0]} samples")
    print(f"Achiever types: {list(config.achiever_types.keys())}")
    if config.is_single_agent_mode():
        print(f"Single-agent mode: No blockers")
    else:
        print(f"Blocker types: {list(config.blocker_types.keys())}")
    
    return processed_data


def load_model_for_visualization(results_dir, config):
    """
    Load a trained model for visualization purposes.
    
    Args:
        results_dir: Directory containing the model
        config: Configuration object
        
    Returns:
        tuple: (model, device) or (None, None) if model not found
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = os.path.join(results_dir, "best_model.pth")
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return None, None
    
    print(f"Loading model from {model_path}...")
    from evaluate import load_model
    model_kwargs = config.get_model_kwargs()
    model = load_model(model_path, device, model_kwargs)
    
    return model, device


def load_test_data_for_visualization(config):
    """
    Load and prepare test data for visualization.
    
    Args:
        config: Configuration object
        
    Returns:
        dict: Test data tensors ready for use
    """
    from utils import load_test_data_all_combinations, combine_all_combinations_data
    
    # Get base data directory from config
    test_data_dir_base = os.path.join(config.save_dir, config.get_env_name())
    
    print("Loading test data from all combinations...")
    all_test_data = load_test_data_all_combinations(config, test_data_dir_base=test_data_dir_base)
    test_data = combine_all_combinations_data(all_test_data)
    print(f"Successfully loaded test data: {test_data['self_states'].shape[0]} samples")
    
    # Convert numpy arrays to tensors
    test_tensors = {
        key: torch.from_numpy(data) if isinstance(data, np.ndarray) else torch.tensor(data)
        for key, data in test_data.items()
    }
    
    return test_tensors


def create_standard_dataloader(test_tensors, batch_size=32):
    """
    Create a standard DataLoader for character and mental embeddings.
    
    Args:
        test_tensors: Dictionary of test data tensors
        batch_size: Batch size for DataLoader
        
    Returns:
        DataLoader: Standard test data loader
    """
    test_dataset = TensorDataset(
        test_tensors["self_states"],
        test_tensors["self_actions"],
        test_tensors["goals"],
        test_tensors["goal_ranks"],
        test_tensors["agents"],
        test_tensors["types"],
        test_tensors["consumption_labels"],
        test_tensors["sr_labels"],
    )
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class SecondBeliefDataLoader:
    """
    Custom DataLoader for second belief embeddings that provides dictionary format.
    """
    def __init__(self, test_tensors, batch_size=32):
        """
        Initialize the SecondBeliefDataLoader.
        
        Args:
            test_tensors: Dictionary of test data tensors
            batch_size: Batch size for DataLoader
        """
        dataset_items = [
            test_tensors["self_states"],
            test_tensors["self_actions"],
            test_tensors["goals"],
            test_tensors["goal_ranks"],
            test_tensors["agents"],
            test_tensors["types"],
            test_tensors["consumption_labels"],
            test_tensors["sr_labels"],
        ]
        
        # Add opponent data if available
        self.has_opponent_data = "opponent_recent_trajectory" in test_tensors
        if self.has_opponent_data:
            dataset_items.extend([
                test_tensors["opponent_recent_trajectory"],
                test_tensors["oppo_actions"]
            ])
        
        self.dataset = TensorDataset(*dataset_items)
        self.batch_size = batch_size
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)
    
    def __iter__(self):
        """Iterate through batches and convert to dictionary format."""
        for batch in self.dataloader:
            batch_dict = {
                'self_states': batch[0],
                'self_actions': batch[1],
                'goal': batch[2],
                'goal_ranks': batch[3],
                'agent': batch[4],
                'type': batch[5],
                'consumption_labels': batch[6],
                'sr_labels': batch[7],
            }
            
            # Add opponent data if available
            if self.has_opponent_data and len(batch) > 8:
                batch_dict['opponent_recent_trajectory'] = batch[8]
                batch_dict['oppo_actions'] = batch[9]
            else:
                batch_dict['opponent_recent_trajectory'] = None
                batch_dict['oppo_actions'] = None
            
            yield batch_dict


def create_second_belief_dataloader(test_tensors, batch_size=32):
    """
    Create a DataLoader for second belief embeddings.
    
    Args:
        test_tensors: Dictionary of test data tensors
        batch_size: Batch size for DataLoader
        
    Returns:
        SecondBeliefDataLoader: Custom dataloader for second belief
    """
    return SecondBeliefDataLoader(test_tensors, batch_size=batch_size)


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
    
    # Debug embeddings quality
    print(f"Embeddings stats:")
    print(f"  Mean: {np.mean(embeddings):.6f}")
    print(f"  Std: {np.std(embeddings):.6f}")
    print(f"  Min: {np.min(embeddings):.6f}")
    print(f"  Max: {np.max(embeddings):.6f}")
    print(f"  NaN count: {np.sum(np.isnan(embeddings))}")
    print(f"  Inf count: {np.sum(np.isinf(embeddings))}")
    
    # Check if embeddings have sufficient variance for visualization
    if np.std(embeddings) < 1e-8:
        print("Warning: Second belief embeddings have very low variance. Visualization may not be meaningful.")
        print("This might indicate that the model's SecondBeliefNet is not properly trained or activated.")
        return False
    
    # Create visualizations
    plot_second_belief_embeddings_by_agent(embeddings, agent_labels, goal_labels, config, output_dir, experiment_no)
    plot_second_belief_embeddings_by_goal(embeddings, agent_labels, goal_labels, config, output_dir, experiment_no)


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

    # Determine which embedding visualizations are requested
    embedding_types_requested = []
    if args.plot_type in ["embeddings", "all"]:
        embedding_types_requested.append("character")
    if args.plot_type in ["mental_embeddings", "all"]:
        embedding_types_requested.append("mental")
    if args.plot_type in ["second_belief_embeddings", "all"]:
        embedding_types_requested.append("second_belief")
    
    if embedding_types_requested:
        # Load model once
        model, device = load_model_for_visualization(results_dir, config)
        
        if model is not None:
            # Load test data once
            test_tensors = load_test_data_for_visualization(config)
            
            # Create appropriate dataloaders based on what's needed
            test_loader = None
            test_loader_second_belief = None
            
            # Create standard dataloader if needed for character or mental embeddings
            if "character" in embedding_types_requested or "mental" in embedding_types_requested:
                test_loader = create_standard_dataloader(test_tensors)
            
            # Create second belief dataloader if needed
            if "second_belief" in embedding_types_requested:
                test_loader_second_belief = create_second_belief_dataloader(test_tensors)
            
            # Plot character embeddings
            if "character" in embedding_types_requested:
                print("Creating character embedding visualizations...")
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
            
            # Plot mental embeddings  
            if "mental" in embedding_types_requested:
                print("Creating mental embedding visualizations...")
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
            
            # Plot second belief embeddings
            if "second_belief" in embedding_types_requested:
                print("Creating second belief embedding visualizations...")
                plot_second_belief_embeddings(
                    model,
                    test_loader_second_belief,
                    device,
                    plot_dir,
                    config,
                    experiment_no,
                    n_samples=None,
                )
                print("Second belief embedding visualization completed!")
            
            # Clean up memory safely
            # Move model to CPU before deletion to avoid CUDA memory issues
            if hasattr(model, 'cpu'):
                model = model.cpu()
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Delete objects in safe order
            if test_loader_second_belief is not None:
                del test_loader_second_belief
            if test_loader is not None:
                del test_loader
            del test_tensors
            
            # Delete model last after moving to CPU
            del model
            
            # Garbage collection without forcing
            import gc
            gc.collect()
            
            # Final CUDA cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            print("Cannot create embedding visualizations: model loading failed.")

    print("Visualization completed!")


