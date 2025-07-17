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
from torch.cuda.amp import autocast, GradScaler
import mmap
import multiprocessing as mp
from functools import partial

# Add current directory to path
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from tomnet import ToMnet, ToMnetLoss, create_model, count_parameters
from data_generation import DataGenerator
from config import Config
from utils import (
    set_seed,
    load_data_efficient,
    save_data_for_mmap,
    load_training_data_all_combinations,
    get_data_for_combination,
)

# Set seed using Config default value
config = Config()
set_seed(config.seed)

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
    (Adapted from experiment 5 for exp5 multi-agent tensor format)

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


def process_sample_batch(samples, grid_size, min_timestep, max_trajectory_length):
    """
    Process a batch of samples for better multiprocessing efficiency
    """
    if not isinstance(samples, list):
        samples = [samples]

    # Batch results
    batch_results = {
        "trajectories": [],
        "actions": [],
        "goals": [],
        "goal_ranks": [],
        "agents": [],
        "types": [],
        "consumption_labels": [],
        "sr_labels": [],
    }

    for sample in samples:
        sample_result = process_single_sample(
            sample, grid_size, min_timestep, max_trajectory_length
        )

        # Combine results
        for key in batch_results:
            batch_results[key].extend(sample_result[key])

    return batch_results


def process_single_sample(sample, grid_size, min_timestep, max_trajectory_length):
    """
    Process a single sample for multiprocessing
    """
    # Extract data from sample
    trajectory = sample["trajectory"]  # [seq_len, channels, height, width]
    goal_tensor = sample["goal"]  # [4] one-hot encoded
    agent_type = sample["agent"]  # 'achiever' or 'blocker'
    type_label = sample[
        "type"
    ]  # 0 for randomly select / achiever, 1 for rule-based blocker
    consumption = sample["consumption_labels"]  # [8] consumption labels
    sr_data_per_timestep = sample.get("sr_data_per_timestep", {})

    # Convert agent type to numerical (0=achiever, 1=blocker)
    agent_label = 0 if agent_type == "achiever" else 1

    # Extract goal rank from goal tensor and agent type
    if agent_type == "achiever":
        goal_idx = torch.argmax(torch.tensor(goal_tensor)).item()
        goal_rank = [2, 2, 2, 2]  # Default rank 2 for all
        goal_rank[goal_idx] = 1  # Set the achieved goal to rank 1
    else:
        goal_idx = torch.argmax(torch.tensor(goal_tensor)).item()
        goal_rank = [2, 2, 2, 2]  # Default rank 2 for all
        goal_rank[goal_idx] = 1  # Set the inferred goal to rank 1

    # Get actions from sample data
    action_list = sample.get("actions", [])

    # Truncate trajectory to max length
    seq_len = min(trajectory.shape[0], max_trajectory_length)
    trajectory = trajectory[:seq_len]
    action_list = action_list[:seq_len]

    # Local lists for this sample
    sample_trajectories = []
    sample_actions = []
    sample_goals = []
    sample_goal_ranks = []
    sample_agents = []
    sample_types = []
    sample_consumption_labels = []
    sample_sr_labels = []

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
        sample_trajectories.append(trajectory_padded)
        sample_actions.append(
            [action_target] + [0] * (max_trajectory_length - 1)
        )  # Pad actions
        sample_goals.append(goal_tensor)
        sample_goal_ranks.append(goal_rank)
        sample_agents.append(agent_label)
        sample_types.append(type_label)
        sample_consumption_labels.append(consumption)
        sample_sr_labels.append(sr_dense)

    return {
        "trajectories": sample_trajectories,
        "actions": sample_actions,
        "goals": sample_goals,
        "goal_ranks": sample_goal_ranks,
        "agents": sample_agents,
        "types": sample_types,
        "consumption_labels": sample_consumption_labels,
        "sr_labels": sample_sr_labels,
    }


def prepare_data_for_training(
    samples,
    grid_size=9,
    min_timestep=3,
    max_trajectory_length=100,
    n_processes=None,
    use_batch_processing=True,
):
    """
    Prepare multi-agent sample data for training from processed samples with trajectory slicing
    Now supports multiprocessing for faster processing

    Args:
        samples: List of processed samples from DataGenerator (containing both achiever and blocker samples)
        grid_size: Size of the grid (default 9 for 9x9)
        min_timestep: Minimum timestep to start slicing from
        max_trajectory_length: Maximum length of trajectory to use
        n_processes: Number of processes to use (default: CPU count)
        use_batch_processing: Whether to use batch processing for better efficiency (default: True)

    Returns:
        Dictionary containing prepared training data
    """

    if n_processes is None:
        n_processes = mp.cpu_count()

    print(
        f"Preparing data from {len(samples)} samples with trajectory slicing using {n_processes} processes..."
    )

    if use_batch_processing:
        # Create batches of samples for better CPU utilization
        batch_size = max(
            1, len(samples) // (n_processes * 5)
        )  # Larger batches for better efficiency
        sample_batches = [
            samples[i : i + batch_size] for i in range(0, len(samples), batch_size)
        ]

        # Create partial function for batch processing
        batch_worker_func = partial(
            process_sample_batch,
            grid_size=grid_size,
            min_timestep=min_timestep,
            max_trajectory_length=max_trajectory_length,
        )

        # Process batches in parallel
        with mp.Pool(processes=n_processes, maxtasksperchild=100) as pool:
            # No need for additional chunking when using batch processing
            results = list(
                tqdm(
                    pool.imap(batch_worker_func, sample_batches),
                    total=len(sample_batches),
                    desc="Dataset processing (batch multiprocessing)",
                )
            )
    else:
        # Original single-sample processing with chunking
        worker_func = partial(
            process_single_sample,
            grid_size=grid_size,
            min_timestep=min_timestep,
            max_trajectory_length=max_trajectory_length,
        )

        # Process samples in parallel with chunking for better CPU utilization
        with mp.Pool(processes=n_processes, maxtasksperchild=100) as pool:
            # Calculate optimal chunk size (similar to generate.py)
            chunk_size = max(1, len(samples) // (n_processes * 10))

            # Use imap with chunking for better performance
            results = list(
                tqdm(
                    pool.imap(worker_func, samples, chunksize=chunk_size),
                    total=len(samples),
                    desc="Dataset processing (multiprocessing)",
                )
            )

    # Combine results from all processes
    trajectories = []
    actions = []
    goals = []
    goal_ranks = []
    agents = []
    types = []
    consumption_labels = []
    sr_labels = []

    for result in results:
        trajectories.extend(result["trajectories"])
        actions.extend(result["actions"])
        goals.extend(result["goals"])
        goal_ranks.extend(result["goal_ranks"])
        agents.extend(result["agents"])
        types.extend(result["types"])
        consumption_labels.extend(result["consumption_labels"])
        sr_labels.extend(result["sr_labels"])

    # Convert to tensors
    trajectories = torch.tensor(np.array(trajectories), dtype=torch.float32)
    actions = torch.tensor(np.array(actions), dtype=torch.long)
    goals = torch.tensor(np.array(goals), dtype=torch.float32)
    goal_ranks = torch.tensor(np.array(goal_ranks), dtype=torch.long)
    agents = torch.tensor(np.array(agents), dtype=torch.long)
    types = torch.tensor(np.array(types), dtype=torch.long)
    consumption_labels = torch.tensor(np.array(consumption_labels), dtype=torch.float32)
    sr_labels = torch.tensor(np.array(sr_labels), dtype=torch.float32)

    print(f"Data shapes:")
    print(f"  Trajectories: {trajectories.shape}")
    print(f"  Actions: {actions.shape}")
    print(f"  Goals: {goals.shape}")
    print(f"  Goal ranks: {goal_ranks.shape}")
    print(f"  Agents: {agents.shape}")
    print(f"  Types: {types.shape}")
    print(f"  Consumption labels: {consumption_labels.shape}")
    print(f"  SR labels: {sr_labels.shape}")

    return {
        "trajectories": trajectories,
        "actions": actions,
        "goals": goals,
        "goal_ranks": goal_ranks,
        "agents": agents,
        "types": types,
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
    total_type_loss = 0
    total_consumption_loss = 0
    total_sr_loss = 0
    correct_actions = 0
    correct_goals = 0
    correct_agents = 0
    correct_types = 0
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
            types,
            consumption_labels,
            sr_labels,
        ) = batch

        trajectories = trajectories.to(device)
        actions = actions.to(device)
        goals = goals.to(device)
        goal_ranks = goal_ranks.to(device)
        agents = agents.to(device)
        types = types.to(device)
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

        # Vectorized effective length calculation (optimized)
        with torch.no_grad():
            # Sum over spatial dimensions for each timestep: [batch_size, seq_len]
            traj_sums = trajectories.sum(dim=(2, 3, 4))
            # Find last non-zero timestep for each batch sample (vectorized)
            non_zero_mask = traj_sums > 0

            # Use flip and argmax trick for efficient last non-zero index finding
            flipped_mask = torch.flip(non_zero_mask, dims=[1])
            last_nonzero_positions = (
                non_zero_mask.size(1) - 1 - torch.argmax(flipped_mask.float(), dim=1)
            )

            # Handle edge case where all timesteps are zero
            all_zero_mask = ~non_zero_mask.any(dim=1)
            effective_lengths = torch.where(
                all_zero_mask,
                torch.zeros_like(last_nonzero_positions),
                last_nonzero_positions,
            )
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
        type_targets = types
        consumption_targets = consumption_labels

        # Zero gradients only at start of accumulation
        if batch_idx % gradient_accumulation_steps == 0:
            optimizer.zero_grad()

        # Forward pass with AMP if enabled
        if scaler is not None:
            with autocast():
                (
                    action_logits,
                    goal_logits,
                    agent_logits,
                    type_logits,
                    consumption_logits,
                    sr_pred,
                    _,
                    _,
                ) = model(past_episodes, recent_trajectory, current_state)

                sr_targets = sr_labels

                # Compute loss with all components including agent and type prediction
                (
                    total_loss_batch,
                    action_loss_batch,
                    goal_loss_batch,
                    agent_loss_batch,
                    type_loss_batch,
                    consumption_loss_batch,
                    sr_loss_batch,
                ) = loss_fn(
                    action_logits,
                    goal_logits,
                    agent_logits,
                    type_logits,
                    consumption_logits,
                    sr_pred,
                    action_targets,
                    goal_targets,
                    agent_targets,
                    type_targets,
                    consumption_targets,
                    sr_targets,
                )

                # Scale loss for gradient accumulation
                total_loss_batch = total_loss_batch / gradient_accumulation_steps

            # Backward pass with AMP
            scaler.scale(total_loss_batch).backward()
        else:
            # Regular forward pass
            (
                action_logits,
                goal_logits,
                agent_logits,
                type_logits,
                consumption_logits,
                sr_pred,
                _,
                _,
            ) = model(past_episodes, recent_trajectory, current_state)

            sr_targets = sr_labels

            # Compute loss with all components including agent and type prediction
            (
                total_loss_batch,
                action_loss_batch,
                goal_loss_batch,
                agent_loss_batch,
                type_loss_batch,
                consumption_loss_batch,
                sr_loss_batch,
            ) = loss_fn(
                action_logits,
                goal_logits,
                agent_logits,
                type_logits,
                consumption_logits,
                sr_pred,
                action_targets,
                goal_targets,
                agent_targets,
                type_targets,
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
        if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(
            train_loader
        ):
            max_grad_norm = (
                training_process_config["max_grad_norm"]
                if training_process_config
                else 1.0
            )

            if scaler is not None:
                # AMP gradient clipping and step
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                # Regular gradient clipping and step
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=max_grad_norm
                )
                optimizer.step()

            # Clear gradients for next accumulation
            optimizer.zero_grad()

        # Update metrics (use unscaled values for reporting)
        total_loss += total_loss_batch.item() * gradient_accumulation_steps
        total_action_loss += action_loss_batch.item()
        total_goal_loss += goal_loss_batch.item()
        total_agent_loss += agent_loss_batch.item()  # Track agent loss
        total_type_loss += type_loss_batch.item()  # Track type loss
        total_consumption_loss += consumption_loss_batch.item()
        total_sr_loss += sr_loss_batch.item()

        # Calculate accuracy
        _, predicted_actions = torch.max(action_logits, 1)
        _, predicted_goals = torch.max(goal_logits, 1)
        _, predicted_agents = torch.max(agent_logits, 1)
        _, predicted_types = torch.max(type_logits, 1)

        correct_actions += (predicted_actions == action_targets).sum().item()
        correct_goals += (predicted_goals == goal_targets).sum().item()
        correct_agents += (
            (predicted_agents == agent_targets).sum().item()
        )  # Track agent accuracy
        correct_types += (
            (predicted_types == type_targets).sum().item()
        )  # Track type accuracy
        total_samples += batch_size

        # Memory cleanup for large batches (optimized)
        if batch_idx % 10 == 0:
            # Clear intermediate variables
            del past_episodes, recent_trajectory, current_state
            del (
                action_logits,
                goal_logits,
                agent_logits,
                type_logits,
                consumption_logits,
                sr_pred,
            )
            del (
                action_targets,
                goal_targets,
                agent_targets,
                type_targets,
                consumption_targets,
                sr_targets,
            )

            # GPU memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    num_batches = len(train_loader)
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_action_loss = total_action_loss / num_batches if num_batches > 0 else 0
    avg_goal_loss = total_goal_loss / num_batches if num_batches > 0 else 0
    avg_agent_loss = total_agent_loss / num_batches if num_batches > 0 else 0
    avg_type_loss = total_type_loss / num_batches if num_batches > 0 else 0
    avg_consumption_loss = (
        total_consumption_loss / num_batches if num_batches > 0 else 0
    )
    avg_sr_loss = total_sr_loss / num_batches if num_batches > 0 else 0
    action_accuracy = correct_actions / total_samples
    goal_accuracy = correct_goals / total_samples
    agent_accuracy = correct_agents / total_samples
    type_accuracy = correct_types / total_samples

    return {
        "loss": avg_loss,
        "action_loss": avg_action_loss,
        "goal_loss": avg_goal_loss,
        "agent_loss": avg_agent_loss,
        "type_loss": avg_type_loss,
        "consumption_loss": avg_consumption_loss,
        "sr_loss": avg_sr_loss,
        "action_accuracy": action_accuracy,
        "goal_accuracy": goal_accuracy,
        "agent_accuracy": agent_accuracy,
        "type_accuracy": type_accuracy,
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
    total_type_loss = 0
    total_consumption_loss = 0
    total_sr_loss = 0
    correct_actions = 0
    correct_goals = 0
    correct_agents = 0
    correct_types = 0
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
                types,
                consumption_labels,
                sr_labels,
            ) = batch

            trajectories = trajectories.to(device)
            actions = actions.to(device)
            goals = goals.to(device)
            goal_ranks = goal_ranks.to(device)
            agents = agents.to(device)
            types = types.to(device)
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

            # Optimized vectorized effective length calculation
            with torch.no_grad():
                # Sum over spatial dimensions for each timestep: [batch_size, seq_len]
                traj_sums = trajectories.sum(dim=(2, 3, 4))
                # Find last non-zero timestep for each batch sample (vectorized)
                non_zero_mask = traj_sums > 0

                # Use flip and argmax trick for efficient last non-zero index finding
                flipped_mask = torch.flip(non_zero_mask, dims=[1])
                last_nonzero_positions = (
                    non_zero_mask.size(1)
                    - 1
                    - torch.argmax(flipped_mask.float(), dim=1)
                )

                # Handle edge case where all timesteps are zero
                all_zero_mask = ~non_zero_mask.any(dim=1)
                effective_lengths = torch.where(
                    all_zero_mask,
                    torch.zeros_like(last_nonzero_positions),
                    last_nonzero_positions,
                )
                effective_lengths = torch.clamp(effective_lengths, min=0)

            # Use trajectory without heading direction for MentalNet (first 8 channels only)
            current_state_channels = model_config.get("current_state_channels", 8)
            recent_trajectory = trajectories[
                :, :, :current_state_channels
            ]  # [batch_size, seq_len, 8, height, width]

            # Optimized current state extraction using advanced indexing
            batch_indices = torch.arange(batch_size, device=trajectories.device)
            last_timesteps = torch.clamp(effective_lengths, min=0)

            # Extract current state using advanced indexing (vectorized)
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
            type_targets = types
            consumption_targets = consumption_labels
            sr_targets = sr_labels

            # Forward pass with AMP if enabled
            if scaler is not None:
                with autocast():
                    (
                        action_logits,
                        goal_logits,
                        agent_logits,
                        type_logits,
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
                    type_logits,
                    consumption_logits,
                    sr_pred,
                    _,
                    _,
                ) = model(past_episodes, recent_trajectory, current_state)

            # Compute loss with all components including agent and type prediction
            (
                total_loss_batch,
                action_loss_batch,
                goal_loss_batch,
                agent_loss_batch,
                type_loss_batch,
                consumption_loss_batch,
                sr_loss_batch,
            ) = loss_fn(
                action_logits,
                goal_logits,
                agent_logits,
                type_logits,
                consumption_logits,
                sr_pred,
                action_targets,
                goal_targets,
                agent_targets,
                type_targets,
                consumption_targets,
                sr_targets,
            )

            # Update metrics
            total_loss += total_loss_batch.item()
            total_action_loss += action_loss_batch.item()
            total_goal_loss += goal_loss_batch.item()
            total_agent_loss += agent_loss_batch.item()
            total_type_loss += type_loss_batch.item()
            total_consumption_loss += consumption_loss_batch.item()
            total_sr_loss += sr_loss_batch.item()

            # Calculate accuracy
            _, predicted_actions = torch.max(action_logits, 1)
            _, predicted_goals = torch.max(goal_logits, 1)
            _, predicted_agents = torch.max(agent_logits, 1)
            _, predicted_types = torch.max(type_logits, 1)

            correct_actions += (predicted_actions == action_targets).sum().item()
            correct_goals += (predicted_goals == goal_targets).sum().item()
            correct_agents += (predicted_agents == agent_targets).sum().item()
            correct_types += (predicted_types == type_targets).sum().item()
            total_samples += batch_size

    num_batches = len(val_loader)
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_action_loss = total_action_loss / num_batches if num_batches > 0 else 0
    avg_goal_loss = total_goal_loss / num_batches if num_batches > 0 else 0
    avg_agent_loss = total_agent_loss / num_batches if num_batches > 0 else 0
    avg_type_loss = total_type_loss / num_batches if num_batches > 0 else 0
    avg_consumption_loss = (
        total_consumption_loss / num_batches if num_batches > 0 else 0
    )
    avg_sr_loss = total_sr_loss / num_batches if num_batches > 0 else 0
    action_accuracy = correct_actions / total_samples
    goal_accuracy = correct_goals / total_samples
    agent_accuracy = correct_agents / total_samples
    type_accuracy = correct_types / total_samples

    return {
        "loss": avg_loss,
        "action_loss": avg_action_loss,
        "goal_loss": avg_goal_loss,
        "agent_loss": avg_agent_loss,
        "type_loss": avg_type_loss,
        "consumption_loss": avg_consumption_loss,
        "sr_loss": avg_sr_loss,
        "action_accuracy": action_accuracy,
        "goal_accuracy": goal_accuracy,
        "agent_accuracy": agent_accuracy,
        "type_accuracy": type_accuracy,
    }


def save_training_plots(history, save_dir):
    """
    Save training history plots as 3 separate graphs: Total, Achiever, Blocker

    Args:
        history: Training history dictionary
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)

    # Create 3 separate figures for Total, Achiever, and Blocker

    # 1. TOTAL metrics plot
    fig_total, axes_total = plt.subplots(2, 2, figsize=(15, 10))
    fig_total.suptitle("TOTAL Metrics", fontsize=16, fontweight="bold")

    # Total Loss plot
    axes_total[0, 0].plot(
        history["epoch"], history["train_loss"], label="Train Loss", marker="o"
    )
    axes_total[0, 0].plot(
        history["epoch"], history["val_loss"], label="Val Loss", marker="s"
    )
    axes_total[0, 0].set_title("Total Loss")
    axes_total[0, 0].set_xlabel("Epoch")
    axes_total[0, 0].set_ylabel("Loss")
    axes_total[0, 0].legend()
    axes_total[0, 0].grid(True)

    # Total Action accuracy plot
    axes_total[0, 1].plot(
        history["epoch"],
        history["train_action_accuracy"],
        label="Train Action Acc",
        marker="o",
    )
    axes_total[0, 1].plot(
        history["epoch"],
        history["val_action_accuracy"],
        label="Val Action Acc",
        marker="s",
    )
    axes_total[0, 1].set_title("Action Accuracy")
    axes_total[0, 1].set_xlabel("Epoch")
    axes_total[0, 1].set_ylabel("Accuracy")
    axes_total[0, 1].legend()
    axes_total[0, 1].grid(True)

    # Total Goal accuracy plot
    axes_total[1, 0].plot(
        history["epoch"],
        history["train_goal_accuracy"],
        label="Train Goal Acc",
        marker="o",
    )
    axes_total[1, 0].plot(
        history["epoch"], history["val_goal_accuracy"], label="Val Goal Acc", marker="s"
    )
    axes_total[1, 0].set_title("Goal Accuracy")
    axes_total[1, 0].set_xlabel("Epoch")
    axes_total[1, 0].set_ylabel("Accuracy")
    axes_total[1, 0].legend()
    axes_total[1, 0].grid(True)

    # Total Combined loss components
    axes_total[1, 1].plot(
        history["epoch"],
        history["train_action_loss"],
        label="Train Action Loss",
        marker="o",
    )
    axes_total[1, 1].plot(
        history["epoch"],
        history["train_goal_loss"],
        label="Train Goal Loss",
        marker="s",
    )
    axes_total[1, 1].plot(
        history["epoch"],
        history["val_action_loss"],
        label="Val Action Loss",
        marker="^",
    )
    axes_total[1, 1].plot(
        history["epoch"], history["val_goal_loss"], label="Val Goal Loss", marker="v"
    )
    axes_total[1, 1].set_title("Loss Components")
    axes_total[1, 1].set_xlabel("Epoch")
    axes_total[1, 1].set_ylabel("Loss")
    axes_total[1, 1].legend()
    axes_total[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "training_history_total.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig_total)

    # 2. ACHIEVER metrics plot (Note: Currently using total metrics - placeholder for future achiever-specific metrics)
    fig_achiever, axes_achiever = plt.subplots(2, 2, figsize=(15, 10))
    fig_achiever.suptitle("ACHIEVER Metrics", fontsize=16, fontweight="bold")

    # Achiever Loss plot (using total metrics as placeholder)
    axes_achiever[0, 0].plot(
        history["epoch"],
        history["train_loss"],
        label="Train Loss",
        marker="o",
        color="green",
    )
    axes_achiever[0, 0].plot(
        history["epoch"],
        history["val_loss"],
        label="Val Loss",
        marker="s",
        color="lightgreen",
    )
    axes_achiever[0, 0].set_title("Achiever Loss")
    axes_achiever[0, 0].set_xlabel("Epoch")
    axes_achiever[0, 0].set_ylabel("Loss")
    axes_achiever[0, 0].legend()
    axes_achiever[0, 0].grid(True)

    # Achiever Action accuracy
    axes_achiever[0, 1].plot(
        history["epoch"],
        history["train_action_accuracy"],
        label="Train Action Acc",
        marker="o",
        color="green",
    )
    axes_achiever[0, 1].plot(
        history["epoch"],
        history["val_action_accuracy"],
        label="Val Action Acc",
        marker="s",
        color="lightgreen",
    )
    axes_achiever[0, 1].set_title("Achiever Action Accuracy")
    axes_achiever[0, 1].set_xlabel("Epoch")
    axes_achiever[0, 1].set_ylabel("Accuracy")
    axes_achiever[0, 1].legend()
    axes_achiever[0, 1].grid(True)

    # Achiever Goal accuracy
    axes_achiever[1, 0].plot(
        history["epoch"],
        history["train_goal_accuracy"],
        label="Train Goal Acc",
        marker="o",
        color="green",
    )
    axes_achiever[1, 0].plot(
        history["epoch"],
        history["val_goal_accuracy"],
        label="Val Goal Acc",
        marker="s",
        color="lightgreen",
    )
    axes_achiever[1, 0].set_title("Achiever Goal Accuracy")
    axes_achiever[1, 0].set_xlabel("Epoch")
    axes_achiever[1, 0].set_ylabel("Accuracy")
    axes_achiever[1, 0].legend()
    axes_achiever[1, 0].grid(True)

    # Achiever Loss components
    axes_achiever[1, 1].plot(
        history["epoch"],
        history["train_action_loss"],
        label="Train Action Loss",
        marker="o",
        color="green",
    )
    axes_achiever[1, 1].plot(
        history["epoch"],
        history["train_goal_loss"],
        label="Train Goal Loss",
        marker="s",
        color="lightgreen",
    )
    axes_achiever[1, 1].plot(
        history["epoch"],
        history["train_consumption_loss"],
        label="Train Consumption Loss",
        marker="^",
        color="darkgreen",
    )
    axes_achiever[1, 1].plot(
        history["epoch"],
        history["train_sr_loss"],
        label="Train SR Loss",
        marker="v",
        color="olive",
    )
    axes_achiever[1, 1].set_title("Achiever Loss Components")
    axes_achiever[1, 1].set_xlabel("Epoch")
    axes_achiever[1, 1].set_ylabel("Loss")
    axes_achiever[1, 1].legend()
    axes_achiever[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "training_history_achiever.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig_achiever)

    # 3. BLOCKER metrics plot (Note: Currently using total metrics - placeholder for future blocker-specific metrics)
    fig_blocker, axes_blocker = plt.subplots(2, 2, figsize=(15, 10))
    fig_blocker.suptitle("BLOCKER Metrics", fontsize=16, fontweight="bold")

    # Blocker Loss plot (using total metrics as placeholder)
    axes_blocker[0, 0].plot(
        history["epoch"],
        history["train_loss"],
        label="Train Loss",
        marker="o",
        color="red",
    )
    axes_blocker[0, 0].plot(
        history["epoch"],
        history["val_loss"],
        label="Val Loss",
        marker="s",
        color="lightcoral",
    )
    axes_blocker[0, 0].set_title("Blocker Loss")
    axes_blocker[0, 0].set_xlabel("Epoch")
    axes_blocker[0, 0].set_ylabel("Loss")
    axes_blocker[0, 0].legend()
    axes_blocker[0, 0].grid(True)

    # Blocker Action accuracy
    axes_blocker[0, 1].plot(
        history["epoch"],
        history["train_action_accuracy"],
        label="Train Action Acc",
        marker="o",
        color="red",
    )
    axes_blocker[0, 1].plot(
        history["epoch"],
        history["val_action_accuracy"],
        label="Val Action Acc",
        marker="s",
        color="lightcoral",
    )
    axes_blocker[0, 1].set_title("Blocker Action Accuracy")
    axes_blocker[0, 1].set_xlabel("Epoch")
    axes_blocker[0, 1].set_ylabel("Accuracy")
    axes_blocker[0, 1].legend()
    axes_blocker[0, 1].grid(True)

    # Blocker Goal accuracy
    axes_blocker[1, 0].plot(
        history["epoch"],
        history["train_goal_accuracy"],
        label="Train Goal Acc",
        marker="o",
        color="red",
    )
    axes_blocker[1, 0].plot(
        history["epoch"],
        history["val_goal_accuracy"],
        label="Val Goal Acc",
        marker="s",
        color="lightcoral",
    )
    axes_blocker[1, 0].set_title("Blocker Goal Accuracy")
    axes_blocker[1, 0].set_xlabel("Epoch")
    axes_blocker[1, 0].set_ylabel("Accuracy")
    axes_blocker[1, 0].legend()
    axes_blocker[1, 0].grid(True)

    # Blocker Loss components
    axes_blocker[1, 1].plot(
        history["epoch"],
        history["train_action_loss"],
        label="Train Action Loss",
        marker="o",
        color="red",
    )
    axes_blocker[1, 1].plot(
        history["epoch"],
        history["train_goal_loss"],
        label="Train Goal Loss",
        marker="s",
        color="lightcoral",
    )
    axes_blocker[1, 1].plot(
        history["epoch"],
        history["train_consumption_loss"],
        label="Train Consumption Loss",
        marker="^",
        color="darkred",
    )
    axes_blocker[1, 1].plot(
        history["epoch"],
        history["train_sr_loss"],
        label="Train SR Loss",
        marker="v",
        color="maroon",
    )
    axes_blocker[1, 1].set_title("Blocker Loss Components")
    axes_blocker[1, 1].set_xlabel("Epoch")
    axes_blocker[1, 1].set_ylabel("Loss")
    axes_blocker[1, 1].legend()
    axes_blocker[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "training_history_blocker.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig_blocker)

    print(f"Training plots saved to:")
    print(f"  - {save_dir}/training_history_total.png")
    print(f"  - {save_dir}/training_history_achiever.png")
    print(f"  - {save_dir}/training_history_blocker.png")


def train_tomnet(
    data_dir=None,
    save_dir="./results/exp5",
    config=None,
    achiever_type=None,
    blocker_type=None,
):
    """
    Main training function for KeyDoor ToMnet

    Args:
        data_dir: Directory containing game data
        save_dir: Directory to save results
        config: Configuration object (Config instance)
        achiever_type: Specific achiever type for this training session
        blocker_type: Specific blocker type for this training session
    """
    # Use provided config or create default
    if config is None:
        config = Config()

    # Set data_dir based on config if not provided
    if data_dir is None:
        env_name = config.get_env_name()
        # Use specific achiever and blocker types if provided
        if achiever_type and blocker_type:
            agent_type = config.get_agent_pair_name(achiever_type, blocker_type)
            data_dir = f"./data/{env_name}/{agent_type}/"
        else:
            # Default to first combination if types not specified
            achiever_type = config.achiever_types[0]
            blocker_type = config.blocker_types[0]
            agent_type = config.get_agent_pair_name(achiever_type, blocker_type)
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

    # Auto-detect CPU count if num_workers is 0
    if num_workers == 0:
        num_workers = mp.cpu_count()
        print(f"Auto-detected {num_workers} CPU cores for data loading")
    # Setup
    experiment_save_dir = save_dir
    os.makedirs(experiment_save_dir, exist_ok=True)

    # Device setup
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # Setup for parallel training
    if use_parallel and torch.cuda.is_available() and len(device_ids) > 1:
        # Check which GPUs are actually available
        available_gpus = []
        for gpu_id in device_ids:
            if gpu_id < torch.cuda.device_count():
                torch.cuda.set_device(gpu_id)
                # Test GPU memory
                test_tensor = torch.zeros(1, device=f"cuda:{gpu_id}")
                available_gpus.append(gpu_id)
                del test_tensor
                torch.cuda.empty_cache()

        if len(available_gpus) > 1:
            print(f"Using parallel training on GPUs: {available_gpus}")
            print(f"Primary device: cuda:{available_gpus[0]}")
            device_ids = available_gpus  # Use only available GPUs
            primary_device = torch.device(f"cuda:{available_gpus[0]}")
            device = primary_device
        else:
            print(
                f"Only {len(available_gpus)} GPU(s) available, using single GPU training"
            )
            if available_gpus:
                device = torch.device(f"cuda:{available_gpus[0]}")
                print(f"Using single device: {device}")
            use_parallel = False
    else:
        print(f"Using single device: {device}")
        use_parallel = False

    # Memory optimization setup
    if torch.cuda.is_available():
        # Clear GPU cache on all available devices
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            torch.cuda.empty_cache()

        # Set back to primary device
        torch.cuda.set_device(device)
        print(
            f"GPU memory allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB"
        )
        print(
            f"GPU memory reserved: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB"
        )

        # Check if we need to reduce batch size due to larger model
        total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
        allocated_memory = torch.cuda.memory_allocated(device) / 1024**3
        available_memory = total_memory - allocated_memory
        print(f"Available GPU memory: {available_memory:.2f} GB")

        if available_memory < 2.0:  # Less than 2GB available
            print(
                "Warning: Low GPU memory detected. Consider reducing batch size or model complexity."
            )

    print(f"Using AMP (Automatic Mixed Precision): {use_amp}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"Training {achiever_type} achiever with {blocker_type} blocker")
    print(f"Results will be saved to: {experiment_save_dir}")

    # Load training data for all combinations efficiently
    all_training_data = load_training_data_all_combinations(
        config, data_dir.replace(f"/{agent_type}", "")
    )

    # Use data for the current combination
    data = get_data_for_combination(
        all_training_data, achiever_type, blocker_type, "training"
    )

    # Sample trajectories randomly for training if specified in config
    if achiever_type and blocker_type:
        achiever_games = config.achiever_types.get(achiever_type, 30000)
        blocker_games = config.blocker_types.get(blocker_type, 30000)
        target_games = achiever_games + blocker_games

        total_samples = data["trajectories"].shape[0]
        if total_samples > target_games:
            print(
                f"Sampling {target_games} trajectories from {total_samples} available samples"
            )

            # Set random seed for reproducible sampling
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)

            # Random sampling indices
            sample_indices = np.random.choice(
                total_samples, target_games, replace=False
            )
            sample_indices = np.sort(sample_indices)

            # Sample all data arrays - create new dictionary to avoid modifying read-only NpzFile
            sampled_data = {}
            for key in data.keys():
                if isinstance(data[key], np.ndarray):
                    sampled_data[key] = data[key][sample_indices]
                elif isinstance(data[key], torch.Tensor):
                    sampled_data[key] = data[key][sample_indices]
                else:
                    sampled_data[key] = data[key]  # Keep non-array data as-is
            data = sampled_data

            print(f"Sampled data to {target_games} trajectories")
        else:
            print(f"Using all {total_samples} trajectories (target: {target_games})")

    # Log data shapes for verification
    print(f"Data shapes:")
    print(f"Trajectories: {data['trajectories'].shape}")
    print(f"Actions: {data['actions'].shape}")
    print(f"Goals: {data['goals'].shape}")
    print(f"Goal ranks: {data['goal_ranks'].shape}")
    print(f"Agents: {data['agents'].shape}")
    print(f"Types: {data['types'].shape}")
    print(f"Consumption labels: {data['consumption_labels'].shape}")
    print(f"SR labels: {data['sr_labels'].shape}")

    # Create datasets with multi-agent data (optimized for memory-mapped data)
    # Force regular tensors if using multiprocessing to avoid serialization issues
    if (
        hasattr(data, "files") and num_workers == 0
    ):  # Memory-mapped data only if no multiprocessing
        print("Using memory-mapped dataset for efficient loading")
        # Create dataset with memory-mapped data
        dataset = MemoryMappedDataset(data)
    else:
        # Use regular tensor dataset for multiprocessing compatibility
        print("Using regular tensor dataset")

        # Convert data to tensors if needed
        trajectories_tensor = (
            torch.from_numpy(data["trajectories"])
            if isinstance(data["trajectories"], np.ndarray)
            else data["trajectories"]
        )
        actions_tensor = (
            torch.from_numpy(data["actions"])
            if isinstance(data["actions"], np.ndarray)
            else data["actions"]
        )
        goals_tensor = (
            torch.from_numpy(data["goals"])
            if isinstance(data["goals"], np.ndarray)
            else data["goals"]
        )
        goal_ranks_tensor = (
            torch.from_numpy(data["goal_ranks"])
            if isinstance(data["goal_ranks"], np.ndarray)
            else data["goal_ranks"]
        )
        agents_tensor = (
            torch.from_numpy(data["agents"])
            if isinstance(data["agents"], np.ndarray)
            else data["agents"]
        )
        types_tensor = (
            torch.from_numpy(data["types"])
            if isinstance(data["types"], np.ndarray)
            else data["types"]
        )
        consumption_labels_tensor = (
            torch.from_numpy(data["consumption_labels"])
            if isinstance(data["consumption_labels"], np.ndarray)
            else data["consumption_labels"]
        )
        sr_labels_tensor = (
            torch.from_numpy(data["sr_labels"])
            if isinstance(data["sr_labels"], np.ndarray)
            else data["sr_labels"]
        )

        dataset = TensorDataset(
            trajectories_tensor,
            actions_tensor,
            goals_tensor,
            goal_ranks_tensor,
            agents_tensor,
            types_tensor,
            consumption_labels_tensor,
            sr_labels_tensor,
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
        print(
            f"Parallel training: increasing batch size from {batch_size} to {parallel_batch_size}"
        )
    else:
        effective_batch_size = min(batch_size, len(train_dataset))
        effective_val_batch_size = min(batch_size, len(val_dataset))

        # Dynamic batch size reduction for memory optimization
        if torch.cuda.is_available():
            available_memory = (
                torch.cuda.get_device_properties(device).total_memory
                - torch.cuda.memory_allocated(device)
            ) / 1024**3
            if available_memory < 3.0 and effective_batch_size > 256:
                new_batch_size = max(256, effective_batch_size // 2)
                print(
                    f"Reducing batch size from {effective_batch_size} to {new_batch_size} due to memory constraints"
                )
                effective_batch_size = new_batch_size
                effective_val_batch_size = min(new_batch_size, len(val_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=effective_val_batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        worker_init_fn=seed_worker,
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

    if torch.cuda.is_available():
        print(
            f"Model loaded to GPU. Memory after model: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB"
        )

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
        type_weight=training_process_config.get("type_weight", 1.0),
        consumption_weight=training_process_config.get("consumption_weight", 1.0),
        sr_weight=training_process_config.get("sr_weight", 1.0),
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=config.training_config["weight_decay"]
    )

    # Initialize AMP scaler for mixed precision training
    device_type = (
        "cuda" if torch.cuda.is_available() and "cuda" in str(device) else "cpu"
    )
    scaler = GradScaler() if use_amp and device_type == "cuda" else None

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
        "train_type_loss": [],
        "train_consumption_loss": [],
        "train_sr_loss": [],
        "train_action_accuracy": [],
        "train_goal_accuracy": [],
        "train_agent_accuracy": [],
        "train_type_accuracy": [],
        "val_loss": [],
        "val_action_loss": [],
        "val_goal_loss": [],
        "val_agent_loss": [],
        "val_type_loss": [],
        "val_consumption_loss": [],
        "val_sr_loss": [],
        "val_action_accuracy": [],
        "val_goal_accuracy": [],
        "val_agent_accuracy": [],
        "val_type_accuracy": [],
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
        )

        # Validation
        val_metrics = validate_epoch(
            model,
            val_loader,
            loss_fn,
            device,
            max_n_past,
            data_config,
            model_config,
            scaler,
        )

        epoch_time = time.time() - epoch_start_time

        # Update history
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_metrics["loss"])
        history["train_action_loss"].append(train_metrics["action_loss"])
        history["train_goal_loss"].append(train_metrics["goal_loss"])
        history["train_agent_loss"].append(train_metrics["agent_loss"])
        history["train_type_loss"].append(train_metrics["type_loss"])
        history["train_consumption_loss"].append(train_metrics["consumption_loss"])
        history["train_sr_loss"].append(train_metrics["sr_loss"])
        history["train_action_accuracy"].append(train_metrics["action_accuracy"])
        history["train_goal_accuracy"].append(train_metrics["goal_accuracy"])
        history["train_agent_accuracy"].append(train_metrics["agent_accuracy"])
        history["train_type_accuracy"].append(train_metrics["type_accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_action_loss"].append(val_metrics["action_loss"])
        history["val_goal_loss"].append(val_metrics["goal_loss"])
        history["val_agent_loss"].append(val_metrics["agent_loss"])
        history["val_type_loss"].append(val_metrics["type_loss"])
        history["val_consumption_loss"].append(val_metrics["consumption_loss"])
        history["val_sr_loss"].append(val_metrics["sr_loss"])
        history["val_action_accuracy"].append(val_metrics["action_accuracy"])
        history["val_goal_accuracy"].append(val_metrics["goal_accuracy"])
        history["val_agent_accuracy"].append(val_metrics["agent_accuracy"])
        history["val_type_accuracy"].append(val_metrics["type_accuracy"])
        history["epoch_time"].append(epoch_time)

        # Print metrics
        train_loss = train_metrics["loss"]
        train_acc = train_metrics["action_accuracy"] * 100
        val_acc = val_metrics["action_accuracy"] * 100
        train_goal_acc = train_metrics["goal_accuracy"] * 100
        val_goal_acc = val_metrics["goal_accuracy"] * 100
        train_agent_acc = train_metrics["agent_accuracy"] * 100
        val_agent_acc = val_metrics["agent_accuracy"] * 100
        train_type_acc = train_metrics["type_accuracy"] * 100
        val_type_acc = val_metrics["type_accuracy"] * 100
        train_action_loss = train_metrics["action_loss"]
        train_agent_loss = train_metrics["agent_loss"]
        train_type_loss = train_metrics["type_loss"]
        train_consumption_loss = train_metrics["consumption_loss"]
        train_sr_loss = train_metrics["sr_loss"]
        val_action_loss = val_metrics["action_loss"]
        val_agent_loss = val_metrics["agent_loss"]
        val_type_loss = val_metrics["type_loss"]
        val_consumption_loss = val_metrics["consumption_loss"]
        val_sr_loss = val_metrics["sr_loss"]

        # Print epoch results in 3 paragraphs: Total, Achiever, Blocker
        print(f"Epoch: {epoch + 1:3d} | Time: {epoch_time:.2f}s")

        # TOTAL Loss and Accuracy
        val_loss = val_metrics["loss"]
        train_goal_loss = train_metrics["goal_loss"]
        val_goal_loss = val_metrics["goal_loss"]
        print(f"  TOTAL    - Loss: Train {train_loss:.4f} | Val {val_loss:.4f}")
        print(
            f"           - Agent Acc: Train {train_agent_acc:.4f}% | Val {val_agent_acc:.4f}%"
        )
        print(
            f"           - Type Acc: Train {train_type_acc:.4f}% | Val {val_type_acc:.4f}%"
        )
        print(
            f"           - Goal Acc: Train {train_goal_acc:.4f}% | Val {val_goal_acc:.4f}%"
        )
        print(f"           - Action Acc: Train {train_acc:.4f}% | Val {val_acc:.4f}%")
        print(
            f"           - Losses: Action {train_action_loss:.4f} | Agent {train_agent_loss:.4f} | Type {train_type_loss:.4f} | Consumption {train_consumption_loss:.4f} | SR {train_sr_loss:.4f}"
        )

        # ACHIEVER-specific metrics (Note: Currently showing total metrics - would need separate calculation for true achiever-only metrics)
        print(
            f"  ACHIEVER - Goal Acc: Train {train_goal_acc:.4f}% | Val {val_goal_acc:.4f}%"
        )
        print(f"           - Action Acc: Train {train_acc:.4f}% | Val {val_acc:.4f}%")
        print(
            f"           - Losses: Action {train_action_loss:.4f} | Goal {train_goal_loss:.4f} | Consumption {train_consumption_loss:.4f} | SR {train_sr_loss:.4f}"
        )

        # BLOCKER-specific metrics (Note: Currently showing total metrics - would need separate calculation for true blocker-only metrics)
        print(
            f"  BLOCKER  - Goal Acc: Train {train_goal_acc:.4f}% | Val {val_goal_acc:.4f}%"
        )
        print(f"           - Action Acc: Train {train_acc:.4f}% | Val {val_acc:.4f}%")
        print(
            f"           - Losses: Action {train_action_loss:.4f} | Goal {train_goal_loss:.4f} | Consumption {train_consumption_loss:.4f} | SR {train_sr_loss:.4f}"
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
                    model.module.state_dict(),
                    os.path.join(experiment_save_dir, "best_model.pth"),
                )
            else:
                torch.save(
                    model.state_dict(),
                    os.path.join(experiment_save_dir, "best_model.pth"),
                )
            print(f"New best model saved (val_loss: {best_val_loss:.4f})")

        # Early stopping
        if early_stopping(val_metrics["loss"], model):
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

        # Memory cleanup after each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

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
        torch.save(
            model.module.state_dict(),
            os.path.join(experiment_save_dir, "final_model.pth"),
        )
    else:
        torch.save(
            model.state_dict(), os.path.join(experiment_save_dir, "final_model.pth")
        )

    print(
        f"\nTraining completed for {achiever_type} achiever with {blocker_type} blocker!"
    )
    print(f"Results saved to: {experiment_save_dir}")
    print(f"Best validation loss: {best_val_loss:.4f}")

    return {
        "model": model,
        "history": history,
        "save_dir": experiment_save_dir,
        "best_val_loss": best_val_loss,
    }


# Data loading functions moved to utils.py


class MemoryMappedDataset(torch.utils.data.Dataset):
    """Dataset that works with memory-mapped data"""

    def __init__(self, mmap_data, indices=None):
        self.mmap_data = mmap_data
        self.indices = (
            indices if indices is not None else range(len(mmap_data["trajectories"]))
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]

        # Convert numpy arrays to torch tensors on access
        return (
            torch.from_numpy(self.mmap_data["trajectories"][real_idx].copy()),
            torch.from_numpy(self.mmap_data["actions"][real_idx].copy()),
            torch.from_numpy(self.mmap_data["goals"][real_idx].copy()),
            torch.from_numpy(self.mmap_data["goal_ranks"][real_idx].copy()),
            torch.from_numpy(self.mmap_data["agents"][real_idx].copy()),
            torch.from_numpy(self.mmap_data["types"][real_idx].copy()),
            torch.from_numpy(self.mmap_data["consumption_labels"][real_idx].copy()),
            torch.from_numpy(self.mmap_data["sr_labels"][real_idx].copy()),
        )


def seed_worker(worker_id):
    """Worker init function for DataLoader to ensure reproducible random seeds"""
    import random
    import numpy as np
    import torch

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


if __name__ == "__main__":
    import argparse
    import os

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
        default="./results/exp5",
        help="Directory to save results",
    )

    # Training configuration
    parser.add_argument("--batch_size", type=int, help="Batch size for training")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument("--weight_decay", type=float, help="Weight decay for optimizer")
    parser.add_argument(
        "--training_proportion",
        type=float,
        help="Proportion of data to use for training",
    )
    parser.add_argument("--device", type=str, help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--optimizer", type=str, help="Optimizer type (adam)")
    parser.add_argument(
        "--use_parallel", action="store_true", help="Enable parallel GPU training"
    )
    parser.add_argument(
        "--device_ids",
        nargs="+",
        type=int,
        help="GPU device IDs for parallel training (e.g., 2 3)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with small-scale settings",
    )

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

    # Set seed for reproducibility
    seed = args.seed if hasattr(args, "seed") else config.seed
    set_seed(seed)
    print(f"Set random seed to {seed} for reproducibility")

    # Run training for all achiever-blocker combinations
    all_results = []

    for achiever_type in config.achiever_types:
        for blocker_type in config.blocker_types:
            print(f"\n{'='*60}")
            print(f"Training for {achiever_type} achiever with {blocker_type} blocker")
            print(f"{'='*60}")

            # Create specific save directory for this combination
            combination_save_dir = os.path.join(
                args.save_dir, f"{achiever_type}_{blocker_type}"
            )

            # Run training for this combination
            results = train_tomnet(
                data_dir=args.data_dir,
                save_dir=combination_save_dir,
                config=config,
                achiever_type=achiever_type,
                blocker_type=blocker_type,
            )

            all_results.append(
                {
                    "achiever_type": achiever_type,
                    "blocker_type": blocker_type,
                    "results": results,
                }
            )

    print(f"\n{'='*60}")
    print(
        f"Training completed for all {len(config.achiever_types)} x {len(config.blocker_types)} combinations"
    )
    print(f"{'='*60}")

    # Print summary
    for result in all_results:
        achiever = result["achiever_type"]
        blocker = result["blocker_type"]
        best_loss = result["results"]["best_val_loss"]
        print(f"{achiever}_{blocker}: Best validation loss = {best_loss:.4f}")
