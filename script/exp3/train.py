import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os
import json
import sys
import numpy as np
from datetime import datetime
import time

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from tomnet import ToMnet, ToMnetLoss, create_model, count_parameters
from data_generation import DataReader
from config import Config

"""
Training system for KeyDoor ToMnet implementation
Adapted from ToMnetF experiment5 for KeyDoor environment
@author: Based on ToMnetF implementation, adapted for KeyDoor
"""


def convert_sparse_sr_to_dense(
    sr_data_timestep, height, width, gammas=[0.5, 0.9, 0.99]
):
    """
    Convert sparse SR data to dense format

    Args:
        sr_data_timestep: Dictionary with gamma values as keys and sparse data as values
        height: Grid height
        width: Grid width
        gammas: List of discount factors

    Returns:
        Dense SR array of shape (3, height, width)
    """
    dense_sr = np.zeros((len(gammas), height, width))

    for gamma_idx, gamma in enumerate(gammas):
        # Convert gamma to string to match data keys
        gamma_key = str(gamma)
        if gamma_key in sr_data_timestep:
            sparse_entries = sr_data_timestep[gamma_key]
            for pos, value in sparse_entries:
                x, y = pos
                if 0 <= x < width and 0 <= y < height:
                    dense_sr[gamma_idx, y, x] = value

    return dense_sr


def calculate_sr_loss_kl_divergence(sr_pred, sr_target):
    """
    Calculate SR loss using KL divergence for probability distributions
    Vectorized version for efficiency (adapted from experiment 5)

    Args:
        sr_pred: Predicted SR maps (batch_size, 3, height, width) - already normalized by softmax
        sr_target: Target SR maps (batch_size, 3, height, width) - raw values, need normalization

    Returns:
        sr_loss: KL divergence loss averaged over discount factors
    """
    batch_size, n_gammas, height, width = sr_pred.shape

    # Vectorized reshape: (batch_size, 3, height*width)
    sr_pred_flat = sr_pred.view(batch_size, n_gammas, -1)
    sr_target_flat = sr_target.view(batch_size, n_gammas, -1)

    # SR predictions are already normalized by softmax in the model
    # Normalize SR targets to probability distributions (sum=1 across spatial locations)
    sr_target_flat = sr_target_flat / (sr_target_flat.sum(dim=2, keepdim=True) + 1e-8)

    # Add small epsilon to avoid log(0)
    sr_pred_flat_safe = sr_pred_flat + 1e-8
    sr_target_flat_safe = sr_target_flat + 1e-8

    # Vectorized KL divergence computation for all gammas at once
    # KL(target || pred) = sum(target * log(target/pred))
    kl_loss = torch.nn.functional.kl_div(
        sr_pred_flat_safe.log(),
        sr_target_flat_safe,
        reduction="none",  # Keep batch and gamma dimensions
    )

    # Sum over spatial dimension, then average over batch and gamma
    kl_loss = kl_loss.sum(dim=2)  # (batch_size, n_gammas)
    kl_loss = kl_loss.mean()  # Average over batch and gamma dimensions

    return kl_loss


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""

    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change in validation loss to qualify as improvement
            restore_best_weights: Whether to restore model weights from the best epoch
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float("inf")
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        """
        Call this method after each epoch

        Args:
            val_loss: Current validation loss
            model: Model to potentially store weights from

        Returns:
            True if training should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = {
                    k: v.clone() for k, v in model.state_dict().items()
                }
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False


def generate_past_episodes_from_batch(
    trajectories,
    goals,
    batch_size,
    n_past_min=1,
    n_past_max=5,
    max_n_past=5,
    rank_threshold=1,
):
    """
    Generate past episodes by randomly sampling from other trajectories in the batch
    with the same goal, using fully vectorized operations for efficiency
    (Adapted from experiment 5 for exp3 tensor format)

    Args:
        trajectories: Batch of trajectories [batch_size, seq_len, channels, height, width]
        goals: Batch of goal labels [batch_size] or goal ranks [batch_size, 4]
        batch_size: Size of current batch
        n_past_min: Minimum number of past episodes to sample
        n_past_max: Maximum number of past episodes to sample
        max_n_past: Maximum number of past episodes for consistent tensor shape
        rank_threshold: How many top ranks to consider for matching (1=only highest, 2=top 2, etc.)

    Returns:
        past_episodes_batch: [batch_size, max_n_past, seq_len, channels, height, width]
    """
    device = trajectories.device
    seq_len, channels, height, width = trajectories.shape[1:]

    # Initialize past episodes tensor
    past_episodes_batch = torch.zeros(
        (batch_size, max_n_past, seq_len, channels, height, width),
        dtype=trajectories.dtype,
        device=device,
    )

    # Generate random n_past values for all samples at once
    n_past_values = torch.randint(
        n_past_min, n_past_max + 1, (batch_size,), device=device
    )

    # Create goal similarity matrix (batch_size x batch_size)
    # same_goal_mask[i, j] = True if sample i and j have the same goal/rank (within threshold)
    if goals.dim() > 1:  # Handle goal_ranks (multi-dimensional)
        # For goal ranks, use rank_threshold to limit comparison
        # goals shape: [batch_size, 4] for goal ranks

        if rank_threshold < 4:
            # Create vectorized comparison using only top ranks for efficiency
            # Find which goals have ranks <= rank_threshold for all samples
            top_goals_mask = goals <= rank_threshold  # [batch_size, 4]

            # Vectorized comparison: check if same goals are in top ranks
            # Expand dimensions for broadcasting: [batch_size, 1, 4] and [1, batch_size, 4]
            top_goals_i = top_goals_mask.unsqueeze(1)  # [batch_size, 1, 4]
            top_goals_j = top_goals_mask.unsqueeze(0)  # [1, batch_size, 4]

            # Check if all 4 positions match (same top goals)
            same_goal_mask = torch.all(
                top_goals_i == top_goals_j, dim=2
            )  # [batch_size, batch_size]
        else:
            # Use full rank comparison (rank_threshold >= 4)
            # Vectorized comparison of full rank vectors
            goals_i = goals.unsqueeze(1)  # [batch_size, 1, 4]
            goals_j = goals.unsqueeze(0)  # [1, batch_size, 4]
            same_goal_mask = torch.all(
                goals_i == goals_j, dim=2
            )  # [batch_size, batch_size]
    else:  # Handle single goal values
        goals_expanded = goals.unsqueeze(1)  # [batch_size, 1]
        same_goal_mask = goals_expanded == goals.unsqueeze(
            0
        )  # [batch_size, batch_size]

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


def prepare_data_for_training(games, min_timestep=6, max_trajectory_length=100):
    """
    Prepare game data for training using trajectory slicing (like experiment 5)

    Args:
        games: List of game data from DataReader
        min_timestep: Minimum timestep to start slicing from
        max_trajectory_length: Maximum length of trajectory to use

    Returns:
        Dictionary containing prepared training data
    """
    trajectories = []
    actions = []
    goals = []
    goal_ranks = []
    goal_rewards = []
    consumption_labels = []
    sr_labels = []

    print(f"Preparing data from {len(games)} games using trajectory slicing...")

    for game in games:
        trajectory = game["trajectory_tensor"]  # [seq_len, channels, height, width]
        action_list = game["actions"]
        goal_tensor = game["goal_tensor"]  # [4] one-hot encoded
        goal_rank = game["goal_rank"]  # [rank1, rank2, rank3, rank4]

        # Extract SR and consumption data
        game_consumption = game.get(
            "consumption_labels", np.zeros(8)
        )  # 8 = 4 keys + 4 doors
        game_sr_data = game.get("sr_data_per_timestep", {})

        # Truncate trajectory to max length
        seq_len = min(trajectory.shape[0], max_trajectory_length)
        trajectory = trajectory[:seq_len]
        action_list = action_list[:seq_len]

        # Ensure we have exactly 9 channels (8 original + 1 heading direction)
        if trajectory.shape[1] >= 9:
            # Data already has 9+ channels, use first 9
            trajectory = trajectory[:, :9, :, :]
        else:
            # Add heading direction channel (9th channel)
            height, width = trajectory.shape[2], trajectory.shape[3]
            heading_channel = np.zeros((seq_len, 1, height, width))

            # Simple heading direction: 0=north, 1=east, 2=south, 3=west (encoded as 0.0, 0.25, 0.5, 0.75)
            # For now, use a placeholder value of 0 (north) for all timesteps
            # TODO: Extract actual heading from action sequence

            # Concatenate heading channel to trajectory
            trajectory = np.concatenate(
                [trajectory, heading_channel], axis=1
            )  # Now 9 channels

        # Get intended goal from goal_rank
        intended_goal_idx = goal_rank.index(1) if 1 in goal_rank else 0

        # Get height and width from trajectory
        height, width = trajectory.shape[2], trajectory.shape[3]

        # TRAJECTORY SLICING: Create multiple samples per game (like experiment 5)
        for i in range(min_timestep, seq_len):
            # Slice trajectory up to timestep i
            trajectory_slice = trajectory[:i]  # [i, channels, height, width]

            # Pad trajectory slice to consistent length for batching
            if i < max_trajectory_length:
                padding = np.zeros((max_trajectory_length - i, *trajectory.shape[1:]))
                trajectory_padded = np.concatenate([trajectory_slice, padding], axis=0)
            else:
                trajectory_padded = trajectory_slice

            # Current state at timestep i-1 (what agent sees before taking action)
            current_timestep = i - 1

            # Action at timestep i (what we want to predict)
            if i < len(action_list):
                action_target = action_list[i]
            else:
                continue  # Skip if no action available

            # Process SR data for this timestep
            if current_timestep in game_sr_data:
                sr_data_timestep = game_sr_data[current_timestep]
                sr_dense = convert_sparse_sr_to_dense(sr_data_timestep, height, width)
            else:
                sr_dense = np.zeros((3, height, width))

            # Add this training sample
            trajectories.append(trajectory_padded)
            actions.append(
                [action_target] + [0] * (max_trajectory_length - 1)
            )  # Pad actions too
            goals.append(intended_goal_idx)
            goal_ranks.append(goal_rank)
            goal_rewards.append(game["goal_rewards"])
            consumption_labels.append(game_consumption)
            sr_labels.append(sr_dense)

    # Convert to tensors
    trajectories = torch.tensor(np.array(trajectories), dtype=torch.float32)
    actions = torch.tensor(np.array(actions), dtype=torch.long)
    goals = torch.tensor(np.array(goals), dtype=torch.long)
    goal_ranks = torch.tensor(np.array(goal_ranks), dtype=torch.long)
    goal_rewards = torch.tensor(np.array(goal_rewards), dtype=torch.float32)
    consumption_labels = torch.tensor(np.array(consumption_labels), dtype=torch.float32)
    sr_labels = torch.tensor(np.array(sr_labels), dtype=torch.float32)

    print(f"Data shapes:")
    print(f"  Trajectories: {trajectories.shape}")
    print(f"  Actions: {actions.shape}")
    print(f"  Goals: {goals.shape}")
    print(f"  Goal ranks: {goal_ranks.shape}")
    print(f"  Goal rewards: {goal_rewards.shape}")
    print(f"  Consumption labels: {consumption_labels.shape}")
    print(f"  SR labels: {sr_labels.shape}")

    return {
        "trajectories": trajectories,
        "actions": actions,
        "goals": goals,
        "goal_ranks": goal_ranks,
        "goal_rewards": goal_rewards,
        "consumption_labels": consumption_labels,
        "sr_labels": sr_labels,
    }


def train_epoch(
    model,
    train_loader,
    optimizer,
    loss_fn,
    device,
    max_n_past=5,
    data_config=None,
    training_process_config=None,
    model_config=None,
):
    """
    Train for one epoch

    Args:
        model: ToMnet model
        train_loader: Training data loader
        optimizer: Optimizer
        loss_fn: Loss function
        device: Device to run on
        max_n_past: Maximum number of past episodes

    Returns:
        Dictionary containing training metrics
    """
    model.train()
    total_loss = 0
    total_action_loss = 0
    total_goal_loss = 0
    total_consumption_loss = 0
    total_sr_loss = 0
    correct_actions = 0
    correct_goals = 0
    total_samples = 0

    for batch_idx, batch in enumerate(train_loader):
        # Unpack all data including new SR and consumption labels
        (
            trajectories,
            actions,
            goals,
            goal_ranks,
            goal_rewards,
            consumption_labels,
            sr_labels,
        ) = batch

        trajectories = trajectories.to(device)
        actions = actions.to(device)
        goals = goals.to(device)
        goal_ranks = goal_ranks.to(device)
        consumption_labels = consumption_labels.to(device)
        sr_labels = sr_labels.to(device)

        batch_size = trajectories.size(0)

        # Generate past episodes from batch
        past_episodes = generate_past_episodes_from_batch(
            trajectories,
            goal_ranks,
            batch_size,
            n_past_min=data_config.get("n_past_min", 1),
            n_past_max=data_config.get("n_past_max", 5),
            max_n_past=max_n_past,
            rank_threshold=data_config.get("rank_threshold", 1),
        )

        # With trajectory slicing, we use dynamic timesteps
        # Each sample has a different effective length, stored in actions[:,0]
        batch_size = trajectories.size(0)

        # For trajectory slicing, use the action at index 0 (the target action for this slice)
        action_targets = actions[:, 0]  # Target action for each sliced trajectory

        # Vectorized: Find the effective length for each sample (remove padding)
        # Sum over spatial dimensions to check if timestep is non-zero
        traj_sum = trajectories.sum(dim=(2, 3, 4))  # [batch_size, seq_len]
        non_zero_mask = traj_sum > 0  # [batch_size, seq_len]

        # Find last non-zero timestep for each sample
        # Create timestep indices on the same device as trajectories
        timestep_indices = (
            torch.arange(trajectories.size(1), device=trajectories.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        # Set padded timesteps to -1
        timestep_indices = timestep_indices * non_zero_mask - (1 - non_zero_mask.long())
        # Get last non-zero timestep index
        effective_lengths = timestep_indices.max(dim=1)[0]  # [batch_size]
        # Convert to 0-based indexing for current state (predict next action)
        effective_lengths = torch.clamp(effective_lengths, min=0)

        # Use trajectory without heading direction for MentalNet (first 8 channels only)
        current_state_channels = model_config.get("current_state_channels", 8)
        recent_trajectory = trajectories[:, :, :current_state_channels]  # [batch_size, seq_len, 8, height, width]

        # Vectorized: Extract current state for PredNet (last non-padded timestep)

        # Create batch indices on the same device as trajectories
        batch_indices = torch.arange(batch_size, device=trajectories.device)
        # Extract current state using advanced indexing
        current_state = trajectories[
            batch_indices, effective_lengths, :current_state_channels
        ]

        goal_targets = goals
        consumption_targets = consumption_labels
        sr_targets = sr_labels

        optimizer.zero_grad()

        # Forward pass - CharNet gets past_episodes, MentalNet gets recent_trajectory, PredNet gets current_state
        action_logits, goal_logits, consumption_logits, sr_pred, _, _ = model(
            past_episodes, recent_trajectory, current_state
        )

        # Compute loss with all components
        (
            total_loss_batch,
            action_loss_batch,
            goal_loss_batch,
            consumption_loss_batch,
            sr_loss_batch,
        ) = loss_fn(
            action_logits,
            goal_logits,
            consumption_logits,
            sr_pred,
            action_targets,
            goal_targets,
            consumption_targets,
            sr_targets,
        )

        # Backward pass
        total_loss_batch.backward()
        max_grad_norm = (
            training_process_config["max_grad_norm"] if training_process_config else 1.0
        )
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        # Update metrics
        total_loss += total_loss_batch.item()
        total_action_loss += action_loss_batch.item()
        total_goal_loss += goal_loss_batch.item()
        total_consumption_loss += consumption_loss_batch.item()
        total_sr_loss += sr_loss_batch.item()

        # Calculate accuracy
        _, predicted_actions = torch.max(action_logits, 1)
        _, predicted_goals = torch.max(goal_logits, 1)

        correct_actions += (predicted_actions == action_targets).sum().item()
        correct_goals += (predicted_goals == goal_targets).sum().item()
        total_samples += batch_size

    num_batches = len(train_loader)
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_action_loss = total_action_loss / num_batches if num_batches > 0 else 0
    avg_goal_loss = total_goal_loss / num_batches if num_batches > 0 else 0
    avg_consumption_loss = (
        total_consumption_loss / num_batches if num_batches > 0 else 0
    )
    avg_sr_loss = total_sr_loss / num_batches if num_batches > 0 else 0
    action_accuracy = correct_actions / total_samples
    goal_accuracy = correct_goals / total_samples

    return {
        "loss": avg_loss,
        "action_loss": avg_action_loss,
        "goal_loss": avg_goal_loss,
        "consumption_loss": avg_consumption_loss,
        "sr_loss": avg_sr_loss,
        "action_accuracy": action_accuracy,
        "goal_accuracy": goal_accuracy,
    }


def validate_epoch(
    model,
    val_loader,
    loss_fn,
    device,
    max_n_past=5,
    data_config=None,
    model_config=None,
):
    """
    Validate for one epoch

    Args:
        model: ToMnet model
        val_loader: Validation data loader
        loss_fn: Loss function
        device: Device to run on
        max_n_past: Maximum number of past episodes

    Returns:
        Dictionary containing validation metrics
    """
    model.eval()
    total_loss = 0
    total_action_loss = 0
    total_goal_loss = 0
    total_consumption_loss = 0
    total_sr_loss = 0
    correct_actions = 0
    correct_goals = 0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            # Unpack all data including new SR and consumption labels
            (
                trajectories,
                actions,
                goals,
                goal_ranks,
                goal_rewards,
                consumption_labels,
                sr_labels,
            ) = batch

            trajectories = trajectories.to(device)
            actions = actions.to(device)
            goals = goals.to(device)
            goal_ranks = goal_ranks.to(device)
            consumption_labels = consumption_labels.to(device)
            sr_labels = sr_labels.to(device)

            batch_size = trajectories.size(0)

            # Generate past episodes from batch
            past_episodes = generate_past_episodes_from_batch(
                trajectories,
                goal_ranks,
                batch_size,
                n_past_min=data_config.get("n_past_min", 1),
                n_past_max=data_config.get("n_past_max", 5),
                max_n_past=max_n_past,
                rank_threshold=data_config.get("rank_threshold", 1),
            )

            # With trajectory slicing, we use dynamic timesteps
            # Each sample has a different effective length, stored in actions[:,0]
            batch_size = trajectories.size(0)

            # Fully vectorized: Find the effective length for each sample (remove padding)
            # Sum over spatial dimensions for each timestep: [batch_size, seq_len]
            traj_sums = trajectories.sum(
                dim=(2, 3, 4)
            )  # Sum over channels, height, width
            # Find last non-zero timestep for each batch sample
            non_zero_mask = traj_sums > 0  # [batch_size, seq_len]
            # Get the last True index for each batch sample using vectorized operation
            # Create sequence indices and mask them on the same device
            seq_indices = (
                torch.arange(trajectories.size(1), device=trajectories.device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )
            masked_indices = torch.where(
                non_zero_mask, seq_indices, torch.tensor(-1, device=trajectories.device)
            )
            # Find the maximum index for each batch (last non-zero timestep)
            effective_lengths = masked_indices.max(dim=1)[0].clamp(min=0).tolist()
            # Apply max(1, length) constraint
            effective_lengths = [max(1, length) for length in effective_lengths]

            # Use trajectory without heading direction for MentalNet (first 8 channels only)
            current_state_channels = model_config.get("current_state_channels", 8)
            recent_trajectory = trajectories[:, :, :current_state_channels]  # [batch_size, seq_len, 8, height, width]

            # Vectorized: Extract current state for PredNet (last non-padded timestep)
            current_state = torch.zeros(
                batch_size,
                current_state_channels,
                trajectories.size(3),
                trajectories.size(4),
            )

            # Create batch indices and timestep indices for advanced indexing on the same device
            batch_indices = torch.arange(batch_size, device=trajectories.device)
            last_timesteps = torch.tensor(
                [max(0, length - 1) for length in effective_lengths],
                device=trajectories.device,
            )

            # Extract current state using advanced indexing
            current_state = trajectories[
                batch_indices, last_timesteps, :current_state_channels
            ]

            # Action target: action at index 0 (target action for this slice)
            action_targets = actions[:, 0]  # Target action for each sliced trajectory

            goal_targets = goals
            consumption_targets = consumption_labels
            sr_targets = sr_labels

            # Forward pass - CharNet gets past_episodes, MentalNet gets recent_trajectory, PredNet gets current_state
            action_logits, goal_logits, consumption_logits, sr_pred, _, _ = model(
                past_episodes, recent_trajectory, current_state
            )

            # Compute loss with all components
            (
                total_loss_batch,
                action_loss_batch,
                goal_loss_batch,
                consumption_loss_batch,
                sr_loss_batch,
            ) = loss_fn(
                action_logits,
                goal_logits,
                consumption_logits,
                sr_pred,
                action_targets,
                goal_targets,
                consumption_targets,
                sr_targets,
            )

            # Update metrics
            total_loss += total_loss_batch.item()
            total_action_loss += action_loss_batch.item()
            total_goal_loss += goal_loss_batch.item()
            total_consumption_loss += consumption_loss_batch.item()
            total_sr_loss += sr_loss_batch.item()

            # Calculate accuracy
            _, predicted_actions = torch.max(action_logits, 1)
            _, predicted_goals = torch.max(goal_logits, 1)

            correct_actions += (predicted_actions == action_targets).sum().item()
            correct_goals += (predicted_goals == goal_targets).sum().item()
            total_samples += batch_size

    num_batches = len(val_loader)
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_action_loss = total_action_loss / num_batches if num_batches > 0 else 0
    avg_goal_loss = total_goal_loss / num_batches if num_batches > 0 else 0
    avg_consumption_loss = (
        total_consumption_loss / num_batches if num_batches > 0 else 0
    )
    avg_sr_loss = total_sr_loss / num_batches if num_batches > 0 else 0
    action_accuracy = correct_actions / total_samples
    goal_accuracy = correct_goals / total_samples

    return {
        "loss": avg_loss,
        "action_loss": avg_action_loss,
        "goal_loss": avg_goal_loss,
        "consumption_loss": avg_consumption_loss,
        "sr_loss": avg_sr_loss,
        "action_accuracy": action_accuracy,
        "goal_accuracy": goal_accuracy,
    }


def save_training_plots(history, save_dir):
    """
    Save training history plots

    Args:
        history: Training history dictionary
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss plot
    axes[0, 0].plot(
        history["epoch"], history["train_loss"], label="Train Loss", marker="o"
    )
    axes[0, 0].plot(history["epoch"], history["val_loss"], label="Val Loss", marker="s")
    axes[0, 0].set_title("Total Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Action accuracy plot
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
    axes[0, 1].grid(True)

    # Goal accuracy plot
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
    axes[1, 0].grid(True)

    # Combined loss components
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
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "training_history.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()

    print(f"Training plots saved to {save_dir}/training_history.png")


def train_tomnet(
    data_dir="./data/exp3",
    save_dir="./results/exp3",
    config=None,
):
    """
    Main training function for KeyDoor ToMnet

    Args:
        data_dir: Directory containing game data
        save_dir: Directory to save results
        config: Configuration object (Config instance)
    """
    # Use provided config or create default
    if config is None:
        config = Config()

    # Extract parameters from config
    training_kwargs = config.get_training_kwargs()
    model_kwargs = config.get_model_kwargs()
    model_config = config.get_model_config()
    data_config = config.get_data_config()
    training_process_config = config.get_training_process_config()

    batch_size = training_kwargs["batch_size"]
    epochs = training_kwargs["epochs"]
    lr = training_kwargs["lr"]
    training_proportion = training_kwargs["training_proportion"]
    time_step = training_kwargs["time_step"]
    max_n_past = training_kwargs["max_n_past"]
    device = training_kwargs["device"]
    patience = training_kwargs["patience"]
    min_delta = training_kwargs["min_delta"]
    # Setup
    experiment_save_dir = save_dir
    os.makedirs(experiment_save_dir, exist_ok=True)

    # Device setup
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    print(f"Using device: {device}")
    print(f"Results will be saved to: {experiment_save_dir}")

    # Load data
    print("Loading data...")
    data_reader = DataReader()
    games = data_reader.ReadAllGames(data_dir)

    if len(games) == 0:
        raise ValueError(f"No games found in {data_dir}")

    # Prepare data using trajectory slicing
    data = prepare_data_for_training(
        games,
        min_timestep=6,  # Start slicing from timestep 6
        max_trajectory_length=time_step,
    )

    # Create datasets with all data including SR and consumption labels
    dataset = TensorDataset(
        data["trajectories"],
        data["actions"],
        data["goals"],
        data["goal_ranks"],
        data["goal_rewards"],
        data["consumption_labels"],
        data["sr_labels"],
    )

    # Train/validation split
    total_size = len(dataset)
    train_size = int(total_size * training_proportion)
    val_size = total_size - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Create data loaders
    # Adjust batch size if dataset is too small
    effective_batch_size = min(batch_size, len(train_dataset))
    effective_val_batch_size = min(batch_size, len(val_dataset))

    train_loader = DataLoader(
        train_dataset, batch_size=effective_batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=effective_val_batch_size, shuffle=False, drop_last=False
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(
        f"Effective batch sizes: train={effective_batch_size}, val={effective_val_batch_size}"
    )

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Dataset too small for training. Need at least 2 samples.")

    # Create model using config
    # Ensure the model gets current_state_channels for MentalNet
    model_kwargs_updated = model_kwargs.copy()
    model_kwargs_updated["current_state_channels"] = model_config.get(
        "current_state_channels", 8
    )
    model = create_model(model_kwargs_updated)
    model = model.to(device)

    print(f"Model created with {count_parameters(model):,} parameters")

    # Loss function and optimizer
    loss_fn = ToMnetLoss(
        action_weight=training_process_config["action_weight"],
        goal_weight=training_process_config["goal_weight"],
        consumption_weight=training_process_config.get("consumption_weight", 1.0),
        sr_weight=training_process_config.get("sr_weight", 1.0),
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=config.training_config["weight_decay"]
    )

    # Early stopping
    early_stopping = EarlyStopping(
        patience=patience, min_delta=min_delta, restore_best_weights=True
    )

    # Training history
    history = {
        "epoch": [],
        "train_loss": [],
        "train_action_loss": [],
        "train_goal_loss": [],
        "train_consumption_loss": [],
        "train_sr_loss": [],
        "train_action_accuracy": [],
        "train_goal_accuracy": [],
        "val_loss": [],
        "val_action_loss": [],
        "val_goal_loss": [],
        "val_consumption_loss": [],
        "val_sr_loss": [],
        "val_action_accuracy": [],
        "val_goal_accuracy": [],
        "epoch_time": [],
    }

    # Training loop
    print("Starting training...")
    best_val_loss = float("inf")

    for epoch in range(epochs):
        epoch_start_time = time.time()

        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 50)

        # Training
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            max_n_past,
            data_config,
            training_process_config,
            model_config,
        )

        # Validation
        val_metrics = validate_epoch(
            model, val_loader, loss_fn, device, max_n_past, data_config, model_config
        )

        epoch_time = time.time() - epoch_start_time

        # Update history
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_metrics["loss"])
        history["train_action_loss"].append(train_metrics["action_loss"])
        history["train_goal_loss"].append(train_metrics["goal_loss"])
        history["train_consumption_loss"].append(train_metrics["consumption_loss"])
        history["train_sr_loss"].append(train_metrics["sr_loss"])
        history["train_action_accuracy"].append(train_metrics["action_accuracy"])
        history["train_goal_accuracy"].append(train_metrics["goal_accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_action_loss"].append(val_metrics["action_loss"])
        history["val_goal_loss"].append(val_metrics["goal_loss"])
        history["val_consumption_loss"].append(val_metrics["consumption_loss"])
        history["val_sr_loss"].append(val_metrics["sr_loss"])
        history["val_action_accuracy"].append(val_metrics["action_accuracy"])
        history["val_goal_accuracy"].append(val_metrics["goal_accuracy"])
        history["epoch_time"].append(epoch_time)

        # Print metrics
        train_loss = train_metrics["loss"]
        train_acc = train_metrics["action_accuracy"] * 100
        val_acc = val_metrics["action_accuracy"] * 100
        train_goal_acc = train_metrics["goal_accuracy"] * 100
        val_goal_acc = val_metrics["goal_accuracy"] * 100
        train_action_loss = train_metrics["action_loss"]
        train_consumption_loss = train_metrics["consumption_loss"]
        train_sr_loss = train_metrics["sr_loss"]
        val_action_loss = val_metrics["action_loss"]
        val_consumption_loss = val_metrics["consumption_loss"]
        val_sr_loss = val_metrics["sr_loss"]

        print(
            f"Epoch: {epoch + 1:3d} | Train Loss: {train_loss:.4f} | Train Action Acc: {train_acc:.4f}% | Val Action Acc: {val_acc:.4f}% | Time: {epoch_time:.2f}s"
        )
        print(f"  Goal Acc - Train: {train_goal_acc:.4f}% | Val: {val_goal_acc:.4f}%")
        print(
            f"  Train - Action: {train_action_loss:.4f} | Consumption: {train_consumption_loss:.4f} | SR: {train_sr_loss:.4f}"
        )
        print(
            f"  Val   - Action: {val_action_loss:.4f} | Consumption: {val_consumption_loss:.4f} | SR: {val_sr_loss:.4f}"
        )
        print("-" * 80)

        # Force flush to ensure real-time logging
        sys.stdout.flush()
        # Also flush any file handlers if redirected
        if hasattr(sys.stdout, "buffer"):
            sys.stdout.buffer.flush()

        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(
                model.state_dict(), os.path.join(experiment_save_dir, "best_model.pth")
            )
            print(f"New best model saved (val_loss: {best_val_loss:.4f})")

        # Early stopping
        if early_stopping(val_metrics["loss"], model):
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    # Save final results
    print("\nSaving results...")

    # Save training history
    with open(os.path.join(experiment_save_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Save model configuration
    with open(os.path.join(experiment_save_dir, "model_config.json"), "w") as f:
        json.dump(model_kwargs, f, indent=2)

    # Save full configuration
    config_dict = {
        "training_config": config.get_training_config(),
        "model_config": config.get_model_config(),
        "data_config": config.get_data_config(),
        "training_process_config": config.get_training_process_config(),
    }
    with open(os.path.join(experiment_save_dir, "full_config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    # Save training plots
    save_training_plots(history, experiment_save_dir)

    # Save final model
    torch.save(model.state_dict(), os.path.join(experiment_save_dir, "final_model.pth"))

    # Save data statistics
    stats = data_reader.get_data_statistics(games)
    with open(os.path.join(experiment_save_dir, "data_statistics.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nTraining completed!")
    print(f"Results saved to: {experiment_save_dir}")
    print(f"Best validation loss: {best_val_loss:.4f}")

    return {
        "model": model,
        "history": history,
        "save_dir": experiment_save_dir,
        "best_val_loss": best_val_loss,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train KeyDoor ToMnet")

    # Basic parameters
    parser.add_argument(
        "--config_override",
        action="store_true",
        help="Enable command line parameter overrides",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/exp3",
        help="Directory containing game data",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./results/exp3",
        help="Directory to save results",
    )

    # Training configuration
    parser.add_argument("--batch_size", type=int, help="Batch size for training")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, help="Weight decay for optimizer")
    parser.add_argument(
        "--training_proportion",
        type=float,
        help="Proportion of data to use for training",
    )
    parser.add_argument("--device", type=str, help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--optimizer", type=str, help="Optimizer type (adam)")

    # Model architecture
    parser.add_argument("--residual_blocks", type=int, help="Number of residual blocks")
    parser.add_argument("--n_echar", type=int, help="Character embedding dimension")
    parser.add_argument("--n_ement", type=int, help="Mental state embedding dimension")
    parser.add_argument("--out_channels", type=int, help="CNN output channels")
    parser.add_argument("--channels_in", type=int, help="CNN input channels")
    parser.add_argument("--action_space", type=int, help="Action space size")
    parser.add_argument("--goal_space", type=int, help="Goal space size")
    parser.add_argument("--hidden_size_lstm", type=int, help="LSTM hidden size")

    # Data processing
    parser.add_argument("--max_moves", type=int, help="Maximum moves per trajectory")
    parser.add_argument("--time_step", type=int, help="Time step for model processing")
    parser.add_argument(
        "--max_n_past", type=int, help="Maximum number of past episodes"
    )
    parser.add_argument(
        "--n_past_min", type=int, help="Minimum number of past episodes"
    )
    parser.add_argument(
        "--n_past_max", type=int, help="Maximum number of past episodes for sampling"
    )
    parser.add_argument(
        "--rank_threshold",
        type=int,
        help="How many top ranks to consider for matching (1=only highest, 2=top 2, etc.)",
    )

    # Training process
    parser.add_argument(
        "--early_stopping_patience", type=int, help="Early stopping patience"
    )
    parser.add_argument(
        "--early_stopping_min_delta", type=float, help="Early stopping minimum delta"
    )
    parser.add_argument(
        "--max_grad_norm", type=float, help="Maximum gradient norm for clipping"
    )
    parser.add_argument("--action_weight", type=float, help="Action loss weight")
    parser.add_argument("--goal_weight", type=float, help="Goal loss weight")

    args = parser.parse_args()

    # Create config and update from args if override is enabled
    config = Config()
    if args.config_override:
        config.update_from_args(args)

    # Run training
    results = train_tomnet(
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        config=config,
    )
