import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os
import json
import sys
import numpy as np
from tqdm import tqdm
from datetime import datetime
import time
import pickle
import gc
from torch.amp import autocast, GradScaler

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from tomnet import ToMnet, ToMnetLoss, create_model, count_parameters
from data_generation import DataGenerator
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
                # Handle DataParallel models
                if isinstance(model, torch.nn.DataParallel):
                    self.best_weights = {
                        k: v.clone() for k, v in model.module.state_dict().items()
                    }
                else:
                    self.best_weights = {
                        k: v.clone() for k, v in model.state_dict().items()
                    }
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                # Handle DataParallel models
                if isinstance(model, torch.nn.DataParallel):
                    model.module.load_state_dict(self.best_weights)
                else:
                    model.load_state_dict(self.best_weights)
            return True
        return False


def generate_past_episodes_from_batch(
    trajectories,
    goal_ranks,
    agents,
    batch_size,
    n_past_min=1,
    n_past_max=5,
    max_n_past=5,
    rank_threshold=1,
):
    """
    Generate past episodes by randomly sampling from other trajectories in the batch
    with the same goal rank AND same agent type, using fully vectorized operations for efficiency
    (Adapted from experiment 5 for exp4 multi-agent tensor format)

    Args:
        trajectories: Batch of trajectories [batch_size, seq_len, channels, height, width]
        goal_ranks: Batch of goal ranks [batch_size, 4] (rank format [1,2,2,2] etc.)
        agents: Batch of agent labels [batch_size] (0=achiever, 1=blocker)
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

    # Create goal similarity matrix (batch_size x batch_size) based on goal ranks
    # same_goal_mask[i, j] = True if sample i and j have similar goal ranks (within threshold)
    # goal_ranks shape: [batch_size, 4] for goal ranks

    if rank_threshold < 4:
        # Create vectorized comparison using only top ranks for efficiency
        # Find which goals have ranks <= rank_threshold for all samples
        top_goals_mask = goal_ranks <= rank_threshold  # [batch_size, 4]

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
        goals_i = goal_ranks.unsqueeze(1)  # [batch_size, 1, 4]
        goals_j = goal_ranks.unsqueeze(0)  # [1, batch_size, 4]
        same_goal_mask = torch.all(
            goals_i == goals_j, dim=2
        )  # [batch_size, batch_size]

    # Create agent similarity matrix (batch_size x batch_size)
    # same_agent_mask[i, j] = True if sample i and j have the same agent type
    agents_expanded = agents.unsqueeze(1)  # [batch_size, 1]
    same_agent_mask = agents_expanded == agents.unsqueeze(0)  # [batch_size, batch_size]

    # Combine goal and agent matching: both must match
    same_goal_and_agent_mask = same_goal_mask & same_agent_mask

    # Exclude self-matches by setting diagonal to False
    same_goal_and_agent_mask.fill_diagonal_(False)

    # Create random sampling matrix for all samples at once
    # For each sample, we create random indices for selecting past episodes
    rand_matrix = torch.rand(batch_size, batch_size, device=device)

    # Mask out invalid sources (different goals/agents or self)
    rand_matrix = rand_matrix * same_goal_and_agent_mask.float()

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


def prepare_data_for_training(
    samples, grid_size=9, min_timestep=6, max_trajectory_length=100
):
    """
    Prepare multi-agent sample data for training from processed samples with trajectory slicing

    Args:
        samples: List of processed samples from DataGenerator (containing both achiever and blocker samples)
        shuffle_data: Whether to shuffle the samples
        grid_size: Size of the grid (default 9 for 9x9)
        min_timestep: Minimum timestep to start slicing from
        max_trajectory_length: Maximum length of trajectory to use

    Returns:
        Dictionary containing prepared training data
    """

    trajectories = []
    actions = []
    goals = []
    goal_ranks = []
    agents = []
    consumption_labels = []
    sr_labels = []

    print(f"Preparing data from {len(samples)} samples with trajectory slicing...")

    for sample in tqdm(samples, desc="Dataset processing"):
        # Extract data from sample
        trajectory = sample["trajectory"]  # [seq_len, channels, height, width]
        goal_tensor = sample["goal"]  # [4] one-hot encoded
        agent_type = sample["agent"]  # 'achiever' or 'blocker'
        consumption = sample["consumption_labels"]  # [8] consumption labels
        sr_data_per_timestep = sample.get("sr_data_per_timestep", {})

        # Convert agent type to numerical (0=achiever, 1=blocker)
        agent_label = 0 if agent_type == "achiever" else 1

        # Extract goal rank from goal tensor and agent type
        if agent_type == "achiever":
            # For achiever: convert one-hot goal to rank format
            # goal_tensor is one-hot [0,0,1,0] -> goal_rank should be [2,2,1,2]
            goal_idx = torch.argmax(torch.tensor(goal_tensor)).item()
            goal_rank = [2, 2, 2, 2]  # Default rank 2 for all
            goal_rank[goal_idx] = 1  # Set the achieved goal to rank 1
        else:
            # For blocker: convert inferred goal to rank format
            # If blocker inferred goal C (index 2), then rank should be [2,2,1,2]
            goal_idx = torch.argmax(torch.tensor(goal_tensor)).item()
            goal_rank = [2, 2, 2, 2]  # Default rank 2 for all
            goal_rank[goal_idx] = 1  # Set the inferred goal to rank 1

        # Get actions from sample data (already extracted from trajectory_steps)
        action_list = sample.get("actions", [])

        # Truncate trajectory to max length
        seq_len = min(trajectory.shape[0], max_trajectory_length)
        trajectory = trajectory[:seq_len]
        action_list = action_list[:seq_len]

        # TRAJECTORY SLICING: Create multiple samples per trajectory
        for i in range(min_timestep, seq_len):
            # Slice trajectory up to timestep i
            trajectory_slice = trajectory[:i]  # [i, channels, height, width]

            # Pad trajectory slice to consistent length for batching
            if i < max_trajectory_length:
                padding_shape = (max_trajectory_length - i, *trajectory.shape[1:])
                padding = np.zeros(padding_shape)
                trajectory_padded = np.concatenate([trajectory_slice, padding], axis=0)
            else:
                trajectory_padded = trajectory_slice

            # Current timestep for action prediction
            current_timestep = i - 1

            # Action at timestep i (what we want to predict)
            if i < len(action_list):
                action_target = action_list[i]
            else:
                continue  # Skip if no action available

            # Process SR data for this timestep
            if current_timestep in sr_data_per_timestep:
                sr_data_timestep = sr_data_per_timestep[current_timestep]
                sr_dense = convert_sparse_sr_to_dense(
                    sr_data_timestep, grid_size, grid_size
                )
            else:
                sr_dense = np.zeros((3, grid_size, grid_size))

            # Add this training sample
            trajectories.append(trajectory_padded)
            actions.append(
                [action_target] + [0] * (max_trajectory_length - 1)
            )  # Pad actions
            goals.append(goal_tensor)
            goal_ranks.append(goal_rank)
            agents.append(agent_label)
            consumption_labels.append(consumption)
            sr_labels.append(sr_dense)

    # Convert to tensors
    trajectories = torch.tensor(np.array(trajectories), dtype=torch.float32)
    actions = torch.tensor(np.array(actions), dtype=torch.long)
    goals = torch.tensor(np.array(goals), dtype=torch.float32)
    goal_ranks = torch.tensor(np.array(goal_ranks), dtype=torch.long)
    agents = torch.tensor(np.array(agents), dtype=torch.long)
    consumption_labels = torch.tensor(np.array(consumption_labels), dtype=torch.float32)
    sr_labels = torch.tensor(np.array(sr_labels), dtype=torch.float32)

    print(f"Data shapes:")
    print(f"  Trajectories: {trajectories.shape}")
    print(f"  Actions: {actions.shape}")
    print(f"  Goals: {goals.shape}")
    print(f"  Goal ranks: {goal_ranks.shape}")
    print(f"  Agents: {agents.shape}")
    print(f"  Consumption labels: {consumption_labels.shape}")
    print(f"  SR labels: {sr_labels.shape}")

    return {
        "trajectories": trajectories,
        "actions": actions,
        "goals": goals,
        "goal_ranks": goal_ranks,
        "agents": agents,
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
    scaler=None,
    gradient_accumulation_steps=1,
    device_type='cuda',
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
    total_agent_loss = 0
    total_consumption_loss = 0
    total_sr_loss = 0
    correct_actions = 0
    correct_goals = 0
    correct_agents = 0
    total_samples = 0
    accumulation_loss = 0

    for batch_idx, batch in enumerate(train_loader):
        # Unpack multi-agent data
        (
            trajectories,
            actions,
            goals,
            goal_ranks,
            agents,
            consumption_labels,
            sr_labels,
        ) = batch

        trajectories = trajectories.to(device)
        actions = actions.to(device)
        goals = goals.to(device)
        goal_ranks = goal_ranks.to(device)
        agents = agents.to(device)
        consumption_labels = consumption_labels.to(device)
        sr_labels = sr_labels.to(device)

        batch_size = trajectories.size(0)

        # Generate past episodes from batch
        past_episodes = generate_past_episodes_from_batch(
            trajectories,
            goal_ranks,
            agents,
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

        # Find the effective length for each sample (remove padding)
        traj_sum = trajectories.sum(dim=(2, 3, 4))  # [batch_size, seq_len]
        non_zero_mask = traj_sum > 0  # [batch_size, seq_len]

        # Find last non-zero timestep for each sample
        timestep_indices = (
            torch.arange(trajectories.size(1), device=trajectories.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        # Set padded timesteps to -1
        timestep_indices = timestep_indices * non_zero_mask - (1 - non_zero_mask.long())
        # Get last non-zero timestep index
        effective_lengths = timestep_indices.max(dim=1)[0]  # [batch_size]
        # Convert to 0-based indexing for current state
        effective_lengths = torch.clamp(effective_lengths, min=0)

        # Use trajectory without heading direction for MentalNet (first 8 channels only)
        current_state_channels = model_config.get("current_state_channels", 8)
        recent_trajectory = trajectories[
            :, :, :current_state_channels
        ]  # [batch_size, seq_len, 8, height, width]

        # Extract current state for PredNet (last non-padded timestep)
        batch_indices = torch.arange(batch_size, device=trajectories.device)
        current_state = trajectories[
            batch_indices, effective_lengths, :current_state_channels
        ]

        # Convert one-hot goals to class indices for loss computation
        if goals.dim() > 1:  # One-hot encoded goals
            goal_targets = torch.argmax(goals, dim=1)
        else:
            goal_targets = goals

        agent_targets = agents
        consumption_targets = consumption_labels

        # Zero gradients only at start of accumulation
        if batch_idx % gradient_accumulation_steps == 0:
            optimizer.zero_grad()

        # Forward pass with AMP if enabled
        if scaler is not None:
            with autocast(device_type):
                action_logits, goal_logits, agent_logits, consumption_logits, sr_pred, _, _ = (
                    model(past_episodes, recent_trajectory, current_state)
                )
                
                sr_targets = sr_labels
                
                # Compute loss with all components including agent prediction
                (
                    total_loss_batch,
                    action_loss_batch,
                    goal_loss_batch,
                    agent_loss_batch,
                    consumption_loss_batch,
                    sr_loss_batch,
                ) = loss_fn(
                    action_logits,
                    goal_logits,
                    agent_logits,
                    consumption_logits,
                    sr_pred,
                    action_targets,
                    goal_targets,
                    agent_targets,
                    consumption_targets,
                    sr_targets,
                )
                
                # Scale loss for gradient accumulation
                total_loss_batch = total_loss_batch / gradient_accumulation_steps
            
            # Backward pass with AMP
            scaler.scale(total_loss_batch).backward()
        else:
            # Regular forward pass
            action_logits, goal_logits, agent_logits, consumption_logits, sr_pred, _, _ = (
                model(past_episodes, recent_trajectory, current_state)
            )
            
            sr_targets = sr_labels
            
            # Compute loss with all components including agent prediction
            (
                total_loss_batch,
                action_loss_batch,
                goal_loss_batch,
                agent_loss_batch,
                consumption_loss_batch,
                sr_loss_batch,
            ) = loss_fn(
                action_logits,
                goal_logits,
                agent_logits,
                consumption_logits,
                sr_pred,
                action_targets,
                goal_targets,
                agent_targets,
                consumption_targets,
                sr_targets,
            )
            
            # Scale loss for gradient accumulation
            total_loss_batch = total_loss_batch / gradient_accumulation_steps
            
            # Regular backward pass
            total_loss_batch.backward()

        # Accumulate loss for reporting (unscaled)
        accumulation_loss += total_loss_batch.item() * gradient_accumulation_steps

        # Optimizer step with gradient accumulation
        if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
            max_grad_norm = (
                training_process_config["max_grad_norm"] if training_process_config else 1.0
            )
            
            if scaler is not None:
                # AMP gradient clipping and step
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Regular gradient clipping and step
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                optimizer.step()
            
            # Clear gradients for next accumulation
            optimizer.zero_grad()

        # Update metrics (use unscaled values for reporting)
        total_loss += total_loss_batch.item() * gradient_accumulation_steps
        total_action_loss += action_loss_batch.item()
        total_goal_loss += goal_loss_batch.item()
        total_agent_loss += agent_loss_batch.item()  # Track agent loss
        total_consumption_loss += consumption_loss_batch.item()
        total_sr_loss += sr_loss_batch.item()

        # Calculate accuracy
        _, predicted_actions = torch.max(action_logits, 1)
        _, predicted_goals = torch.max(goal_logits, 1)
        _, predicted_agents = torch.max(agent_logits, 1)

        correct_actions += (predicted_actions == action_targets).sum().item()
        correct_goals += (predicted_goals == goal_targets).sum().item()
        correct_agents += (
            (predicted_agents == agent_targets).sum().item()
        )  # Track agent accuracy
        total_samples += batch_size
        
        # Memory cleanup for large batches
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    num_batches = len(train_loader)
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_action_loss = total_action_loss / num_batches if num_batches > 0 else 0
    avg_goal_loss = total_goal_loss / num_batches if num_batches > 0 else 0
    avg_agent_loss = total_agent_loss / num_batches if num_batches > 0 else 0
    avg_consumption_loss = (
        total_consumption_loss / num_batches if num_batches > 0 else 0
    )
    avg_sr_loss = total_sr_loss / num_batches if num_batches > 0 else 0
    action_accuracy = correct_actions / total_samples
    goal_accuracy = correct_goals / total_samples
    agent_accuracy = correct_agents / total_samples

    return {
        "loss": avg_loss,
        "action_loss": avg_action_loss,
        "goal_loss": avg_goal_loss,
        "agent_loss": avg_agent_loss,
        "consumption_loss": avg_consumption_loss,
        "sr_loss": avg_sr_loss,
        "action_accuracy": action_accuracy,
        "goal_accuracy": goal_accuracy,
        "agent_accuracy": agent_accuracy,
    }


def validate_epoch(
    model,
    val_loader,
    loss_fn,
    device,
    max_n_past=5,
    data_config=None,
    model_config=None,
    scaler=None,
    device_type='cuda',
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
    total_agent_loss = 0
    total_consumption_loss = 0
    total_sr_loss = 0
    correct_actions = 0
    correct_goals = 0
    correct_agents = 0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            # Unpack multi-agent data
            (
                trajectories,
                actions,
                goals,
                goal_ranks,
                agents,
                consumption_labels,
                sr_labels,
            ) = batch

            trajectories = trajectories.to(device)
            actions = actions.to(device)
            goals = goals.to(device)
            goal_ranks = goal_ranks.to(device)
            agents = agents.to(device)
            consumption_labels = consumption_labels.to(device)
            sr_labels = sr_labels.to(device)

            batch_size = trajectories.size(0)

            # Generate past episodes from batch
            past_episodes = generate_past_episodes_from_batch(
                trajectories,
                goal_ranks,
                agents,
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
            recent_trajectory = trajectories[
                :, :, :current_state_channels
            ]  # [batch_size, seq_len, 8, height, width]

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

            # For trajectory slicing, use the action at index 0 (the target action for this slice)
            action_targets = actions[:, 0]  # Target action for each sliced trajectory

            # Convert one-hot goals to class indices for loss computation
            if goals.dim() > 1:  # One-hot encoded goals
                goal_targets = torch.argmax(goals, dim=1)
            else:
                goal_targets = goals

            agent_targets = agents
            consumption_targets = consumption_labels
            sr_targets = sr_labels

            # Forward pass with AMP if enabled
            if scaler is not None:
                with autocast(device_type):
                    (
                        action_logits,
                        goal_logits,
                        agent_logits,
                        consumption_logits,
                        sr_pred,
                        _,
                        _,
                    ) = model(past_episodes, recent_trajectory, current_state)
            else:
                # Regular forward pass
                (
                    action_logits,
                    goal_logits,
                    agent_logits,
                    consumption_logits,
                    sr_pred,
                    _,
                    _,
                ) = model(past_episodes, recent_trajectory, current_state)

            # Compute loss with all components including agent prediction
            (
                total_loss_batch,
                action_loss_batch,
                goal_loss_batch,
                agent_loss_batch,
                consumption_loss_batch,
                sr_loss_batch,
            ) = loss_fn(
                action_logits,
                goal_logits,
                agent_logits,
                consumption_logits,
                sr_pred,
                action_targets,
                goal_targets,
                agent_targets,
                consumption_targets,
                sr_targets,
            )

            # Update metrics
            total_loss += total_loss_batch.item()
            total_action_loss += action_loss_batch.item()
            total_goal_loss += goal_loss_batch.item()
            total_agent_loss += agent_loss_batch.item()
            total_consumption_loss += consumption_loss_batch.item()
            total_sr_loss += sr_loss_batch.item()

            # Calculate accuracy
            _, predicted_actions = torch.max(action_logits, 1)
            _, predicted_goals = torch.max(goal_logits, 1)
            _, predicted_agents = torch.max(agent_logits, 1)

            correct_actions += (predicted_actions == action_targets).sum().item()
            correct_goals += (predicted_goals == goal_targets).sum().item()
            correct_agents += (predicted_agents == agent_targets).sum().item()
            total_samples += batch_size

    num_batches = len(val_loader)
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_action_loss = total_action_loss / num_batches if num_batches > 0 else 0
    avg_goal_loss = total_goal_loss / num_batches if num_batches > 0 else 0
    avg_agent_loss = total_agent_loss / num_batches if num_batches > 0 else 0
    avg_consumption_loss = (
        total_consumption_loss / num_batches if num_batches > 0 else 0
    )
    avg_sr_loss = total_sr_loss / num_batches if num_batches > 0 else 0
    action_accuracy = correct_actions / total_samples
    goal_accuracy = correct_goals / total_samples
    agent_accuracy = correct_agents / total_samples

    return {
        "loss": avg_loss,
        "action_loss": avg_action_loss,
        "goal_loss": avg_goal_loss,
        "agent_loss": avg_agent_loss,
        "consumption_loss": avg_consumption_loss,
        "sr_loss": avg_sr_loss,
        "action_accuracy": action_accuracy,
        "goal_accuracy": goal_accuracy,
        "agent_accuracy": agent_accuracy,
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
    data_dir=None,
    save_dir="./results/exp4",
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

    # Set data_dir based on config if not provided
    if data_dir is None:
        env_name = config.get_env_name()
        agent_type = f"{config.achiever_type}_{config.blocker_type}"
        data_dir = f"./data/{env_name}/{agent_type}/"

    # Extract parameters from config
    training_kwargs = config.get_training_kwargs()
    model_kwargs = config.get_model_kwargs()
    model_config = config.get_model_config()
    data_config = config.get_data_config()
    training_process_config = config.get_training_process_config()
    training_config = config.get_training_config()

    batch_size = training_kwargs["batch_size"]
    epochs = training_kwargs["epochs"]
    lr = training_kwargs["lr"]
    training_proportion = training_kwargs["training_proportion"]
    time_step = training_kwargs["time_step"]
    max_n_past = training_kwargs["max_n_past"]
    device = training_kwargs["device"]
    patience = training_kwargs["patience"]
    min_delta = training_kwargs["min_delta"]
    
    # Get parallel training configuration
    use_parallel = training_config.get("use_parallel", False)
    device_ids = training_config.get("device_ids", [2, 3])
    
    # Memory and computation optimization settings
    use_amp = training_config.get("use_amp", True)  # Automatic Mixed Precision
    gradient_accumulation_steps = training_config.get("gradient_accumulation_steps", 1)
    pin_memory = training_config.get("pin_memory", True)
    num_workers = training_config.get("num_workers", 4)
    # Setup
    experiment_save_dir = save_dir
    os.makedirs(experiment_save_dir, exist_ok=True)

    # Device setup
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        primary_device_id = 0 if torch.cuda.is_available() else None
    else:
        device = torch.device(device)
        # Extract device ID from device string (e.g., "cuda:3" -> 3)
        if "cuda:" in str(device):
            primary_device_id = int(str(device).split(":")[1])
        else:
            primary_device_id = 0

    # Setup for parallel training
    if use_parallel and torch.cuda.is_available() and len(device_ids) > 1:
        print(f"Using parallel training on GPUs: {device_ids}")
        print(f"Primary device: cuda:{device_ids[0]}")
        # Set primary device for model initialization
        primary_device = torch.device(f"cuda:{device_ids[0]}")
        device = primary_device
    else:
        print(f"Using single device: {device}")
        use_parallel = False
    
    # Memory optimization setup
    if torch.cuda.is_available():
        # Clear GPU cache
        torch.cuda.empty_cache()
        print(f"GPU memory allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB")
        
    print(f"Using AMP (Automatic Mixed Precision): {use_amp}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"Results will be saved to: {experiment_save_dir}")

    # Check if processed data exists, if not generate it
    processed_data_path = os.path.join(
        data_dir, f"processed_data_exp{config.experiment_no}.pkl"
    )

    if not os.path.exists(processed_data_path):
        print("Processed data not found. Generating...")
        # Load and process data
        print("Loading raw data...")
        # Import the new DataReader for multi-agent environment
        from data_generation import DataGenerator as MultiAgentDataReader

        # Create DataReader for multi-agent data
        data_reader = MultiAgentDataReader(
            time_step=time_step,
            w=config.width,
            h=config.height,
            d=data_config.get("maze_depth", 9),
            config=config,
        )

        # Process directory to get samples
        samples = data_reader.process_directory(data_dir)

        if len(samples) == 0:
            raise ValueError(f"No samples found in {data_dir}")

        # Prepare data from multi-agent samples with shuffling
        data = prepare_data_for_training(
            samples, grid_size=config.width, max_trajectory_length=time_step
        )

        # Save processed training data for future use
        with open(processed_data_path, "wb") as f:
            pickle.dump(data, f)
        print(f"  Successfully saved to {processed_data_path}")
    else:
        print("Loading existing processed data...")
        # Load pre-processed training data directly
        with open(processed_data_path, "rb") as f:
            data = pickle.load(f)
        print(f"  Successfully loaded from {processed_data_path}")

    # Log data shapes for verification
    print(f"Data shapes:")
    print(f"Trajectories: {data['trajectories'].shape}")
    print(f"Actions: {data['actions'].shape}")
    print(f"Goals: {data['goals'].shape}")
    print(f"Goal ranks: {data['goal_ranks'].shape}")
    print(f"Agents: {data['agents'].shape}")
    print(f"Consumption labels: {data['consumption_labels'].shape}")
    print(f"SR labels: {data['sr_labels'].shape}")

    # Create datasets with multi-agent data
    dataset = TensorDataset(
        data["trajectories"],
        data["actions"],
        data["goals"],
        data["goal_ranks"],
        data["agents"],
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
    # Adjust batch size if dataset is too small and for parallel training
    if use_parallel and len(device_ids) > 1:
        # Increase batch size for parallel training (distribute across GPUs)
        parallel_batch_size = batch_size * len(device_ids)
        effective_batch_size = min(parallel_batch_size, len(train_dataset))
        effective_val_batch_size = min(parallel_batch_size, len(val_dataset))
        print(f"Parallel training: increasing batch size from {batch_size} to {parallel_batch_size}")
    else:
        effective_batch_size = min(batch_size, len(train_dataset))
        effective_val_batch_size = min(batch_size, len(val_dataset))

    train_loader = DataLoader(
        train_dataset, 
        batch_size=effective_batch_size, 
        shuffle=True, 
        drop_last=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=effective_val_batch_size, 
        shuffle=False, 
        drop_last=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False
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

    # Setup parallel training if enabled
    if use_parallel and torch.cuda.is_available() and len(device_ids) > 1:
        print(f"Wrapping model with DataParallel for GPUs: {device_ids}")
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # Count parameters correctly for DataParallel
    if isinstance(model, torch.nn.DataParallel):
        param_count = count_parameters(model.module)
    else:
        param_count = count_parameters(model)
    print(f"Model created with {param_count:,} parameters")

    # Loss function and optimizer
    loss_fn = ToMnetLoss(
        action_weight=training_process_config["action_weight"],
        goal_weight=training_process_config["goal_weight"],
        agent_weight=training_process_config.get("agent_weight", 1.0),
        consumption_weight=training_process_config.get("consumption_weight", 1.0),
        sr_weight=training_process_config.get("sr_weight", 1.0),
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=config.training_config["weight_decay"]
    )
    
    # Initialize AMP scaler for mixed precision training
    device_type = 'cuda' if torch.cuda.is_available() and 'cuda' in str(device) else 'cpu'
    scaler = GradScaler(device_type) if use_amp and device_type == 'cuda' else None

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
        "train_agent_loss": [],
        "train_consumption_loss": [],
        "train_sr_loss": [],
        "train_action_accuracy": [],
        "train_goal_accuracy": [],
        "train_agent_accuracy": [],
        "val_loss": [],
        "val_action_loss": [],
        "val_goal_loss": [],
        "val_agent_loss": [],
        "val_consumption_loss": [],
        "val_sr_loss": [],
        "val_action_accuracy": [],
        "val_goal_accuracy": [],
        "val_agent_accuracy": [],
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
            scaler,
            gradient_accumulation_steps,
            device_type,
        )

        # Validation
        val_metrics = validate_epoch(
            model, val_loader, loss_fn, device, max_n_past, data_config, model_config, scaler, device_type
        )

        epoch_time = time.time() - epoch_start_time

        # Update history
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_metrics["loss"])
        history["train_action_loss"].append(train_metrics["action_loss"])
        history["train_goal_loss"].append(train_metrics["goal_loss"])
        history["train_agent_loss"].append(train_metrics["agent_loss"])
        history["train_consumption_loss"].append(train_metrics["consumption_loss"])
        history["train_sr_loss"].append(train_metrics["sr_loss"])
        history["train_action_accuracy"].append(train_metrics["action_accuracy"])
        history["train_goal_accuracy"].append(train_metrics["goal_accuracy"])
        history["train_agent_accuracy"].append(train_metrics["agent_accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_action_loss"].append(val_metrics["action_loss"])
        history["val_goal_loss"].append(val_metrics["goal_loss"])
        history["val_agent_loss"].append(val_metrics["agent_loss"])
        history["val_consumption_loss"].append(val_metrics["consumption_loss"])
        history["val_sr_loss"].append(val_metrics["sr_loss"])
        history["val_action_accuracy"].append(val_metrics["action_accuracy"])
        history["val_goal_accuracy"].append(val_metrics["goal_accuracy"])
        history["val_agent_accuracy"].append(val_metrics["agent_accuracy"])
        history["epoch_time"].append(epoch_time)

        # Print metrics
        train_loss = train_metrics["loss"]
        train_acc = train_metrics["action_accuracy"] * 100
        val_acc = val_metrics["action_accuracy"] * 100
        train_goal_acc = train_metrics["goal_accuracy"] * 100
        val_goal_acc = val_metrics["goal_accuracy"] * 100
        train_agent_acc = train_metrics["agent_accuracy"] * 100
        val_agent_acc = val_metrics["agent_accuracy"] * 100
        train_action_loss = train_metrics["action_loss"]
        train_agent_loss = train_metrics["agent_loss"]
        train_consumption_loss = train_metrics["consumption_loss"]
        train_sr_loss = train_metrics["sr_loss"]
        val_action_loss = val_metrics["action_loss"]
        val_agent_loss = val_metrics["agent_loss"]
        val_consumption_loss = val_metrics["consumption_loss"]
        val_sr_loss = val_metrics["sr_loss"]

        print(
            f"Epoch: {epoch + 1:3d} | Train Loss: {train_loss:.4f} | Train Action Acc: {train_acc:.4f}% | Val Action Acc: {val_acc:.4f}% | Time: {epoch_time:.2f}s"
        )
        print(f"  Goal Acc - Train: {train_goal_acc:.4f}% | Val: {val_goal_acc:.4f}%")
        print(
            f"  Agent Acc - Train: {train_agent_acc:.4f}% | Val: {val_agent_acc:.4f}%"
        )
        print(
            f"  Train - Action: {train_action_loss:.4f} | Agent: {train_agent_loss:.4f} | Consumption: {train_consumption_loss:.4f} | SR: {train_sr_loss:.4f}"
        )
        print(
            f"  Val   - Action: {val_action_loss:.4f} | Agent: {val_agent_loss:.4f} | Consumption: {val_consumption_loss:.4f} | SR: {val_sr_loss:.4f}"
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
            # Save model correctly for DataParallel
            if isinstance(model, torch.nn.DataParallel):
                torch.save(
                    model.module.state_dict(), os.path.join(experiment_save_dir, "best_model.pth")
                )
            else:
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
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), os.path.join(experiment_save_dir, "final_model.pth"))
    else:
        torch.save(model.state_dict(), os.path.join(experiment_save_dir, "final_model.pth"))

    # Save data statistics if data_reader is available
    try:
        if 'data_reader' in locals() and 'samples' in locals():
            stats = data_reader.get_statistics(samples)
            with open(os.path.join(experiment_save_dir, "data_statistics.json"), "w") as f:
                json.dump(stats, f, indent=2)
        else:
            print("Skipping data statistics - using pre-processed data")
    except Exception as e:
        print(f"Warning: Could not save data statistics: {e}")

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
        default=None,
        help="Directory containing game data (auto-generated from config if not provided)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./results/exp4",
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
    parser.add_argument("--use_parallel", action="store_true", help="Enable parallel GPU training")
    parser.add_argument("--device_ids", nargs="+", type=int, help="GPU device IDs for parallel training (e.g., 2 3)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with small-scale settings")

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
    if args.debug:
        config.enable_debug_mode()
    if args.config_override:
        config.update_from_args(args)

    # Run training
    results = train_tomnet(
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        config=config,
    )
