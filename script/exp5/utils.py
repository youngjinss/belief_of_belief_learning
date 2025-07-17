"""
Common utility functions for exp5 scripts
Consolidates duplicated functions across train.py, evaluate.py, visualize.py
"""

# Additional imports for utility functions moved from train.py
import torch
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import gc
import multiprocessing as mp
from functools import partial
import psutil


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
    chunk_size=10000,  # Number of samples per chunk
    output_dir="./data_chunks",
):
    """
    Prepare multi-agent sample data for training from processed samples with trajectory slicing
    Now supports multiprocessing and memory-efficient chunked processing

    Args:
        samples: List of processed samples from DataGenerator (containing both achiever and blocker samples)
        grid_size: Size of the grid (default 9 for 9x9)
        min_timestep: Minimum timestep to start slicing from
        max_trajectory_length: Maximum length of trajectory to use
        n_processes: Number of processes to use (default: CPU count)
        use_batch_processing: Whether to use batch processing for better efficiency (default: True)
        chunk_size: Number of samples to process per chunk (default: 10000)
        output_dir: Directory to save data chunks (default: ./data_chunks)

    Returns:
        Dictionary containing metadata about the chunked data
    """

    if n_processes is None:
        n_processes = mp.cpu_count()

    print(
        f"Preparing data from {len(samples)} samples with trajectory slicing using {n_processes} processes..."
    )
    print(f"Processing in chunks of {chunk_size} samples, saving to {output_dir}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Split samples into chunks
    sample_chunks = [
        samples[i : i + chunk_size] for i in range(0, len(samples), chunk_size)
    ]

    chunk_metadata = []
    total_samples = 0

    for chunk_idx, chunk_samples in enumerate(tqdm(sample_chunks, desc="Processing chunks")):
        print(f"Processing chunk {chunk_idx + 1}/{len(sample_chunks)} ({len(chunk_samples)} samples)")
        
        # Process current chunk
        chunk_data = prepare_data_memory_efficient(
            chunk_samples,
            grid_size=grid_size,
            min_timestep=min_timestep,
            max_trajectory_length=max_trajectory_length,
            n_processes=n_processes,
            use_batch_processing=use_batch_processing,
        )
        
        # Save chunk to disk
        chunk_file = os.path.join(output_dir, f"chunk_{chunk_idx:04d}.pt")
        torch.save(chunk_data, chunk_file)
        
        # Record metadata
        chunk_info = {
            "chunk_idx": chunk_idx,
            "file_path": chunk_file,
            "num_samples": len(chunk_data["trajectories"]),
            "data_shapes": {
                "trajectories": chunk_data["trajectories"].shape,
                "actions": chunk_data["actions"].shape,
                "goals": chunk_data["goals"].shape,
                "goal_ranks": chunk_data["goal_ranks"].shape,
                "agents": chunk_data["agents"].shape,
                "types": chunk_data["types"].shape,
                "consumption_labels": chunk_data["consumption_labels"].shape,
                "sr_labels": chunk_data["sr_labels"].shape,
            }
        }
        chunk_metadata.append(chunk_info)
        total_samples += len(chunk_data["trajectories"])
        
        print(f"  Saved chunk {chunk_idx} with {len(chunk_data['trajectories'])} samples to {chunk_file}")
        
        # Free memory
        del chunk_data
        gc.collect()

    print(f"Total processed samples: {total_samples}")
    print(f"Data saved in {len(chunk_metadata)} chunks in {output_dir}")

    return {
        "chunk_metadata": chunk_metadata,
        "total_samples": total_samples,
        "num_chunks": len(chunk_metadata),
        "output_dir": output_dir,
    }


def prepare_data_memory_efficient(
    samples,
    grid_size=9,
    min_timestep=3,
    max_trajectory_length=100,
    n_processes=None,
    use_batch_processing=True,
):
    """
    Memory-efficient version of prepare_data_for_training that processes a single chunk
    """

    if n_processes is None:
        n_processes = mp.cpu_count()

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

    # Convert to tensors (this is now done on smaller chunks)
    trajectories = torch.tensor(np.array(trajectories), dtype=torch.float32)
    actions = torch.tensor(np.array(actions), dtype=torch.long)
    goals = torch.tensor(np.array(goals), dtype=torch.float32)
    goal_ranks = torch.tensor(np.array(goal_ranks), dtype=torch.long)
    agents = torch.tensor(np.array(agents), dtype=torch.long)
    types = torch.tensor(np.array(types), dtype=torch.long)
    consumption_labels = torch.tensor(np.array(consumption_labels), dtype=torch.float32)
    sr_labels = torch.tensor(np.array(sr_labels), dtype=torch.float32)

    print(f"Chunk data shapes:")
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
        history["train_achiever_action_accuracy"],
        label="Train Action Acc",
        marker="o",
        color="green",
    )
    axes_achiever[0, 1].plot(
        history["epoch"],
        history["val_achiever_action_accuracy"],
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
        history["train_achiever_goal_accuracy"],
        label="Train Goal Acc",
        marker="o",
        color="green",
    )
    axes_achiever[1, 0].plot(
        history["epoch"],
        history["val_achiever_goal_accuracy"],
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
        history["train_blocker_action_accuracy"],
        label="Train Action Acc",
        marker="o",
        color="red",
    )
    axes_blocker[0, 1].plot(
        history["epoch"],
        history["val_blocker_action_accuracy"],
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
        history["train_blocker_goal_accuracy"],
        label="Train Goal Acc",
        marker="o",
        color="red",
    )
    axes_blocker[1, 0].plot(
        history["epoch"],
        history["val_blocker_goal_accuracy"],
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


def print_epoch_metrics(epoch, epoch_time, train_metrics, val_metrics):
    """
    Print epoch metrics in an organized format with agent-specific losses

    Args:
        epoch: Current epoch number (0-indexed)
        epoch_time: Time taken for this epoch
        train_metrics: Dictionary of training metrics
        val_metrics: Dictionary of validation metrics
    """
    # Extract metrics
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

    # Achiever/Blocker specific metrics
    train_achiever_acc = train_metrics["achiever_action_accuracy"] * 100
    val_achiever_acc = val_metrics["achiever_action_accuracy"] * 100
    train_achiever_goal_acc = train_metrics["achiever_goal_accuracy"] * 100
    val_achiever_goal_acc = val_metrics["achiever_goal_accuracy"] * 100
    train_blocker_acc = train_metrics["blocker_action_accuracy"] * 100
    val_blocker_acc = val_metrics["blocker_action_accuracy"] * 100
    train_blocker_goal_acc = train_metrics["blocker_goal_accuracy"] * 100
    val_blocker_goal_acc = val_metrics["blocker_goal_accuracy"] * 100

    # Agent-specific losses
    train_achiever_action_loss = train_metrics.get(
        "achiever_action_loss", train_action_loss
    )
    train_achiever_goal_loss = train_metrics.get(
        "achiever_goal_loss", train_metrics["goal_loss"]
    )
    train_achiever_consumption_loss = train_metrics.get(
        "achiever_consumption_loss", train_consumption_loss
    )
    train_achiever_sr_loss = train_metrics.get("achiever_sr_loss", train_sr_loss)

    train_blocker_action_loss = train_metrics.get(
        "blocker_action_loss", train_action_loss
    )
    train_blocker_goal_loss = train_metrics.get(
        "blocker_goal_loss", train_metrics["goal_loss"]
    )
    train_blocker_consumption_loss = train_metrics.get(
        "blocker_consumption_loss", train_consumption_loss
    )
    train_blocker_sr_loss = train_metrics.get("blocker_sr_loss", train_sr_loss)

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

    # ACHIEVER-specific metrics
    print(
        f"  ACHIEVER - Goal Acc: Train {train_achiever_goal_acc:.4f}% | Val {val_achiever_goal_acc:.4f}%"
    )
    print(
        f"           - Action Acc: Train {train_achiever_acc:.4f}% | Val {val_achiever_acc:.4f}%"
    )
    print(
        f"           - Losses: Action {train_achiever_action_loss:.4f} | Goal {train_achiever_goal_loss:.4f} | Consumption {train_achiever_consumption_loss:.4f} | SR {train_achiever_sr_loss:.4f}"
    )

    # BLOCKER-specific metrics
    print(
        f"  BLOCKER  - Goal Acc: Train {train_blocker_goal_acc:.4f}% | Val {val_blocker_goal_acc:.4f}%"
    )
    print(
        f"           - Action Acc: Train {train_blocker_acc:.4f}% | Val {val_blocker_acc:.4f}%"
    )
    print(
        f"           - Losses: Action {train_blocker_action_loss:.4f} | Goal {train_blocker_goal_loss:.4f} | Consumption {train_blocker_consumption_loss:.4f} | SR {train_blocker_sr_loss:.4f}"
    )

    print("-" * 80)


def setup_training_environment(
    config, training_kwargs, training_config, device_setting
):
    """
    Setup training environment including device, parallel training, and memory optimization

    Args:
        config: Configuration object
        training_kwargs: Training keyword arguments
        training_config: Training configuration dictionary
        device_setting: Device string or "auto"

    Returns:
        tuple: (device, use_parallel, device_ids, use_amp, gradient_accumulation_steps, other_configs)
    """
    # Get configuration values
    use_parallel = training_config.get("use_parallel", False)
    device_ids = training_config.get("device_ids", [2, 3])
    use_amp = training_config.get("use_amp", True)
    gradient_accumulation_steps = training_config.get("gradient_accumulation_steps", 1)
    pin_memory = training_config.get("pin_memory", True)
    num_workers = training_config.get("num_workers", 4)

    # Auto-detect CPU count if num_workers is 0
    if num_workers == 0:
        num_workers = mp.cpu_count()
        print(f"Auto-detected {num_workers} CPU cores for data loading")

    # Device setup
    if device_setting == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_setting)

    # Setup for parallel training
    if use_parallel and torch.cuda.is_available() and len(device_ids) > 1:
        available_gpus = []
        for gpu_id in device_ids:
            if gpu_id < torch.cuda.device_count():
                torch.cuda.set_device(gpu_id)
                test_tensor = torch.zeros(1, device=f"cuda:{gpu_id}")
                available_gpus.append(gpu_id)
                del test_tensor
                torch.cuda.empty_cache()

        if len(available_gpus) > 1:
            print(f"Using parallel training on GPUs: {available_gpus}")
            print(f"Primary device: cuda:{available_gpus[0]}")
            device_ids = available_gpus
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
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            torch.cuda.empty_cache()

        torch.cuda.set_device(device)
        print(
            f"GPU memory allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB"
        )
        print(
            f"GPU memory reserved: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB"
        )

        total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
        allocated_memory = torch.cuda.memory_allocated(device) / 1024**3
        available_memory = total_memory - allocated_memory
        print(f"Available GPU memory: {available_memory:.2f} GB")

        if available_memory < 2.0:
            print(
                "Warning: Low GPU memory detected. Consider reducing batch size or model complexity."
            )

    print(f"Using AMP (Automatic Mixed Precision): {use_amp}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")

    other_configs = {"pin_memory": pin_memory, "num_workers": num_workers}

    return (
        device,
        use_parallel,
        device_ids,
        use_amp,
        gradient_accumulation_steps,
        other_configs,
    )


def _load_chunks_memory_efficient(chunk_metadata):
    """
    Memory-efficient chunk loading
    """
    if "chunk_metadata" not in chunk_metadata:
        # Old format
        return chunk_metadata
    
    import os
    
    num_chunks = chunk_metadata['num_chunks']
    chunk_dir = chunk_metadata['output_dir']
    
    print(f"Loading {num_chunks} chunks with memory-efficient strategy...")
    
    # Check available memory
    process = psutil.Process()
    available_memory = psutil.virtual_memory().available / (1024**3)
    
    # For very low memory, process one chunk at a time and accumulate
    if available_memory < 8.0:
        print(f"Very low memory ({available_memory:.1f} GB). Using single-chunk processing.")
        
        # First pass: get shapes
        first_chunk = torch.load(os.path.join(chunk_dir, "chunk_0000.pt"), map_location='cpu')
        data_keys = list(first_chunk.keys())
        
        # Accumulate data arrays
        accumulated_data = {}
        
        for chunk_idx in range(num_chunks):
            chunk_path = os.path.join(chunk_dir, f"chunk_{chunk_idx:04d}.pt")
            
            try:
                chunk_data = torch.load(chunk_path, map_location='cpu')
            
                for key in data_keys:
                    if key in chunk_data:
                        if key not in accumulated_data:
                            accumulated_data[key] = []
                        # Convert to numpy immediately to save memory
                        accumulated_data[key].append(chunk_data[key].numpy())
                
                del chunk_data
                gc.collect()
                
                if chunk_idx % 5 == 0:
                    print(f"  Loaded {chunk_idx + 1}/{num_chunks} chunks...")
            
            except:
                print(f"{chunk_path} end")
                break
        
        # Final combination
        print("Combining all data...")
        combined_data = {}
        for key in data_keys:
            if key in accumulated_data and accumulated_data[key]:
                combined_data[key] = np.concatenate(accumulated_data[key], axis=0)
                accumulated_data[key].clear()
        
        del accumulated_data
        gc.collect()
        
        return combined_data
    
    # Standard loading for sufficient memory
    return _standard_chunk_loading(chunk_metadata)


def _standard_chunk_loading(chunk_metadata):
    """Standard chunk loading when memory is sufficient"""
    if "chunk_metadata" not in chunk_metadata:
        return chunk_metadata
        
    print(f"Loading {chunk_metadata['num_chunks']} chunks for training...")
    
    # Initialize lists to collect data
    all_trajectories = []
    all_actions = []
    all_goals = []
    all_goal_ranks = []
    all_agents = []
    all_types = []
    all_consumption_labels = []
    all_sr_labels = []
    
    # Load each chunk
    for chunk_idx in range(chunk_metadata['num_chunks']):
        chunk_path = os.path.join(chunk_metadata['output_dir'], f"chunk_{chunk_idx:04d}.pt")
        print(f"Loading chunk {chunk_idx} from {chunk_path}")
        
        try:
            chunk_data = torch.load(chunk_path, map_location='cpu')
            
            # Append to lists
            all_trajectories.append(chunk_data['trajectories'])
            all_actions.append(chunk_data['actions'])
            all_goals.append(chunk_data['goals'])
            all_goal_ranks.append(chunk_data['goal_ranks'])
            all_agents.append(chunk_data['agents'])
            all_types.append(chunk_data['types'])
            all_consumption_labels.append(chunk_data['consumption_labels'])
            all_sr_labels.append(chunk_data['sr_labels'])
            
            # Free memory
            del chunk_data
        except:
            print(f"{chunk_path} end")
    
    # Combine all data efficiently
    print("Combining all chunks...")
    
    combined_data = {}
    
    # Process each data type separately
    for key, data_list in [
        ('trajectories', all_trajectories),
        ('actions', all_actions),
        ('goals', all_goals),
        ('goal_ranks', all_goal_ranks),
        ('agents', all_agents),
        ('types', all_types),
        ('consumption_labels', all_consumption_labels),
        ('sr_labels', all_sr_labels),
    ]:
        print(f"  Combining {key}...")
        combined_data[key] = torch.cat(data_list, dim=0).numpy()
        data_list.clear()
        gc.collect()
    
    return combined_data


def setup_model_and_data(
    config,
    model_kwargs,
    data_dir,
    agent_type,
    achiever_type,
    blocker_type,
    training_proportion,
    device,
    use_parallel,
    device_ids,
    pin_memory,
    num_workers,
    training_process_config,
    batch_size,
    lr,
    weight_decay,
    patience,
    min_delta,
):
    """
    Setup model, data loaders, optimizer, and other training components

    Returns:
        tuple: (model, train_loader, val_loader, optimizer, loss_fn, scaler, early_stopping)
    """
    from torch.utils.data import DataLoader
    from torch.cuda.amp import GradScaler
    import torch.utils.data
    
    # Import these here to avoid circular imports
    from tomnet import ToMnetLoss, create_model

    # Load training data
    # Extract base directory from full data path
    data_dir_base = os.path.dirname(data_dir)
    all_training_data = load_training_data_all_combinations(
        config, data_dir_base
    )

    chunk_metadata = get_data_for_combination(
        all_training_data, achiever_type, blocker_type, "training"
    )

    # Load chunked data and combine for training
    data = _load_chunks_memory_efficient(chunk_metadata)

    # Convert numpy arrays to torch tensors
    trajectories = torch.from_numpy(data["trajectories"]).float()
    actions = torch.from_numpy(data["actions"]).long()
    goals = torch.from_numpy(data["goals"]).float()
    goal_ranks = torch.from_numpy(data["goal_ranks"]).long()
    agents = torch.from_numpy(data["agents"]).long()
    types = torch.from_numpy(data["types"]).long()
    consumption_labels = torch.from_numpy(data["consumption_labels"]).float()
    sr_labels = torch.from_numpy(data["sr_labels"]).float()

    # Create TensorDataset from the tensors
    dataset = TensorDataset(
        trajectories,
        actions,
        goals,
        goal_ranks,
        agents,
        types,
        consumption_labels,
        sr_labels,
    )

    # Split data
    total_samples = len(dataset)
    val_size = int(config.n_games_per_type * (1 - training_proportion))
    split_idx = total_samples - val_size

    train_data = torch.utils.data.Subset(dataset, range(split_idx))
    val_data = torch.utils.data.Subset(dataset, range(split_idx, total_samples))

    # Calculate samples per epoch based on agent types
    total_achiever_samples = sum(config.achiever_types.values())
    total_blocker_samples = sum(config.blocker_types.values())
    samples_per_epoch = total_achiever_samples + total_blocker_samples

    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)} (using n_games_per_type * (1 - training_proportion) = {val_size})")
    print(f"Total samples: {total_samples}")
    print(f"Samples per epoch: {samples_per_epoch} (achiever: {total_achiever_samples}, blocker: {total_blocker_samples})")

    # Training loader will be created dynamically each epoch
    train_loader = None  # Will be created in training loop

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
    )

    # Create model
    model = create_model(model_kwargs)
    model = model.to(device)

    # Setup parallel training
    if use_parallel and len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        print(f"Model wrapped with DataParallel using devices: {device_ids}")

    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # Create loss function
    loss_fn = ToMnetLoss(
        action_weight=training_process_config["action_weight"],
        goal_weight=training_process_config["goal_weight"],
        agent_weight=training_process_config.get("agent_weight", 1.0),
        type_weight=training_process_config.get("type_weight", 1.0),
        consumption_weight=training_process_config.get("consumption_weight", 1.0),
        sr_weight=training_process_config.get("sr_weight", 1.0),
    )

    # Create scaler for AMP
    use_amp = training_process_config.get("use_amp", True)
    scaler = GradScaler() if use_amp else None

    # Create early stopping
    early_stopping = EarlyStopping(
        patience=patience, min_delta=min_delta, restore_best_weights=True
    )

    return model, train_data, val_loader, optimizer, loss_fn, scaler, early_stopping, samples_per_epoch, batch_size, pin_memory, num_workers

import os
import pickle
import random
import numpy as np
import torch
import sys

# Add path for imports
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# All imports will be done dynamically to avoid circular imports


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility across all major libraries.

    Args:
        seed (int): Random seed value
    """
    # Python random module
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    # CUDA convolution determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Environment variables for additional reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)

    # For DataLoader workers
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return seed_worker


def load_data_mmap(filepath):
    """Load data using memory mapping for faster access"""
    data = np.load(filepath, mmap_mode="r")
    print(f"Loaded memory-mapped data from {filepath}")
    return data


def load_data_efficient(filepath):
    """Load data efficiently from pickle or chunk metadata"""
    # Load pickle file (now contains chunk metadata)
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        return data

    return None



def load_training_data_all_combinations(config, data_dir_base=None):
    """
    Load training data for all combinations following train.py pattern

    Args:
        config: Config object
        data_dir_base: Base directory for training data

    Returns:
        dict: Dictionary with (achiever_type, blocker_type) -> training_data mapping
    """
    if data_dir_base is None:
        env_name = config.get_env_name()
        data_dir_base = f"./data/{env_name}"

    # Get all combinations
    all_combinations = []
    for achiever_type in config.achiever_types.keys():
        for blocker_type in config.blocker_types.keys():
            all_combinations.append((achiever_type, blocker_type))

    existing_data = {}
    missing_combinations = []

    for combo_achiever, combo_blocker in all_combinations:
        # Construct training data directory path
        agent_pair = config.get_agent_pair_name(combo_achiever, combo_blocker)
        train_data_dir = os.path.join(data_dir_base, agent_pair)

        processed_data_path = os.path.join(
            train_data_dir,
            f"processed_data_exp{config.experiment_no}_{combo_achiever}_{combo_blocker}.pkl",
        )

        # Try efficient data loading
        data = load_data_efficient(processed_data_path)
        if data is not None:
            print(
                f"Loading existing processed training data for {combo_achiever}_{combo_blocker}..."
            )
            existing_data[(combo_achiever, combo_blocker)] = data
            print(f"  Successfully loaded from {processed_data_path}")
        else:
            missing_combinations.append((combo_achiever, combo_blocker))

    # Generate missing data if needed
    if missing_combinations:
        print(
            f"Processed training data not found for combinations: {missing_combinations}"
        )
        print("Generating training data for missing combinations...")

        for combo_achiever, combo_blocker in missing_combinations:
            # Construct training data directory path
            agent_pair = config.get_agent_pair_name(combo_achiever, combo_blocker)
            train_data_dir = os.path.join(data_dir_base, agent_pair)

            processed_data_path = os.path.join(
                train_data_dir,
                f"processed_data_exp{config.experiment_no}_{combo_achiever}_{combo_blocker}.pkl",
            )


            if not os.path.exists(train_data_dir):
                print(f"Training data directory not found: {train_data_dir}")
                print(f"Skipping combination {combo_achiever}_{combo_blocker}")
                continue

            # Create chunk directory for this combination
            chunk_dir = os.path.join(train_data_dir, f"chunks_{combo_achiever}_{combo_blocker}")
            
            # Check if processed data already exists
            if os.path.exists(processed_data_path):
                print(f"Loading existing processed data from: {processed_data_path}")
                train_data = load_data_efficient(processed_data_path)
                if train_data is not None:
                    # Check if this is chunk metadata (new format) or old format
                    if isinstance(train_data, dict) and "chunk_metadata" in train_data:
                        print(f"  Found chunk metadata with {train_data['num_chunks']} chunks")
                        # Verify chunks still exist
                        chunks_exist = all(os.path.exists(chunk['file_path']) for chunk in train_data['chunk_metadata'])
                        if chunks_exist:
                            print(f"  All chunks verified in {chunk_dir}")
                            existing_data[(combo_achiever, combo_blocker)] = train_data
                            continue
                        else:
                            print(f"  Some chunks are missing, regenerating...")
                    else:
                        print(f"  Found old format data, converting to chunked format...")
                        # Remove old pkl file to force regeneration
                        os.remove(processed_data_path)

            # Load and process raw training data
            from data_generation import DataGenerator as DataReader

            data_config = config.get_data_config()
            data_reader = DataReader(
                time_step=data_config.get("time_step", 500),
                w=config.width,
                h=config.height,
                d=data_config.get("maze_depth", 9),
                config=config,
            )

            train_games = data_reader.ReadAllGames(train_data_dir)
            if len(train_games) == 0:
                print(f"No training games found in {train_data_dir}")
                continue

            # Process training data using chunked approach
            train_data = prepare_data_for_training(
                train_games,
                min_timestep=data_config.get("min_time_steps", 6),
                max_trajectory_length=data_config.get("time_step", 500),
                chunk_size=data_config.get("chunk_size", 5000),  # Process in smaller chunks to avoid memory issues
                output_dir=chunk_dir,
            )

            # Save chunk metadata instead of full data
            print(f"Saving chunk metadata to: {processed_data_path}")
            with open(processed_data_path, "wb") as f:
                pickle.dump(train_data, f)
            print(f"  Successfully saved metadata to {processed_data_path}")
            print(f"  Data chunks saved in {chunk_dir}")

            existing_data[(combo_achiever, combo_blocker)] = train_data

    return existing_data


def load_test_data_all_combinations(config, test_data_dir_base=None):
    """
    Load test data for all combinations following train.py pattern

    Args:
        config: Config object
        test_data_dir_base: Base directory for test data

    Returns:
        dict: Dictionary with (achiever_type, blocker_type) -> test_data mapping
    """
    if test_data_dir_base is None:
        env_name = config.get_env_name()
        test_data_dir_base = f"./data/{env_name}"

    # Get all combinations
    all_combinations = []
    for achiever_type in config.achiever_types.keys():
        for blocker_type in config.blocker_types.keys():
            all_combinations.append((achiever_type, blocker_type))

    existing_data = {}
    missing_combinations = []

    for combo_achiever, combo_blocker in all_combinations:
        # Construct test data directory path
        agent_pair = config.get_agent_pair_name(combo_achiever, combo_blocker)
        test_data_dir = os.path.join(test_data_dir_base, agent_pair, "test")

        processed_test_data_path = os.path.join(
            test_data_dir,
            f"processed_test_data_exp{config.experiment_no}_{combo_achiever}_{combo_blocker}.pkl",
        )

        # Try efficient data loading
        data = load_data_efficient(processed_test_data_path)
        if data is not None:
            print(
                f"Loading existing processed test data for {combo_achiever}_{combo_blocker}..."
            )
            existing_data[(combo_achiever, combo_blocker)] = data
            print(f"  Successfully loaded from {processed_test_data_path}")
        else:
            missing_combinations.append((combo_achiever, combo_blocker))

    # Generate missing data if needed
    if missing_combinations:
        print(f"Processed test data not found for combinations: {missing_combinations}")
        print("Generating test data for missing combinations...")

        for combo_achiever, combo_blocker in missing_combinations:
            # Construct test data directory path
            agent_pair = config.get_agent_pair_name(combo_achiever, combo_blocker)
            test_data_dir = os.path.join(test_data_dir_base, agent_pair, "test")

            processed_test_data_path = os.path.join(
                test_data_dir,
                f"processed_test_data_exp{config.experiment_no}_{combo_achiever}_{combo_blocker}.pkl",
            )

            if not os.path.exists(test_data_dir):
                print(f"Test data directory not found: {test_data_dir}")
                print(f"Skipping combination {combo_achiever}_{combo_blocker}")
                continue

            # Load and process raw test data
            from data_generation import DataGenerator as DataReader

            data_config = config.get_data_config()
            data_reader = DataReader(
                time_step=data_config.get("time_step", 500),
                w=config.width,
                h=config.height,
                d=data_config.get("maze_depth", 9),
                config=config,
            )

            test_games = data_reader.ReadAllGames(test_data_dir)
            if len(test_games) == 0:
                print(f"No test games found in {test_data_dir}")
                continue

            # Process test data
            test_data = prepare_data_for_training(
                test_games,
                min_timestep=data_config.get("min_time_steps", 6),
                max_trajectory_length=data_config.get("time_step", 500),
            )

            # Save processed test data
            print(f"Saving processed test data to: {processed_test_data_path}")
            with open(processed_test_data_path, "wb") as f:
                pickle.dump(test_data, f)
            print(f"  Successfully saved to {processed_test_data_path}")

            existing_data[(combo_achiever, combo_blocker)] = test_data

    return existing_data


def get_data_for_combination(
    all_data, achiever_type, blocker_type, data_type="training"
):
    """
    Get data for a specific combination from loaded data dictionary

    Args:
        all_data: Dictionary with (achiever_type, blocker_type) -> data mapping
        achiever_type: Type of achiever agent
        blocker_type: Type of blocker agent
        data_type: Type of data (for error messages)

    Returns:
        data: Data for the specified combination
    """
    if (achiever_type, blocker_type) in all_data:
        data = all_data[(achiever_type, blocker_type)]
        print(f"Using {data_type} data for {achiever_type}_{blocker_type}")
        return data
    else:
        raise ValueError(
            f"No {data_type} data found for combination {achiever_type}_{blocker_type}. Please generate {data_type} data first."
        )


def load_chunked_data_for_training(train_data, batch_size, samples_per_epoch, pin_memory=True, num_workers=0):
    """
    Create a DataLoader for one epoch from training data
    
    Args:
        train_data: Training dataset or chunk metadata (for backward compatibility)
        batch_size: Batch size
        samples_per_epoch: Number of samples per epoch
        pin_memory: Whether to pin memory
        num_workers: Number of workers
        
    Returns:
        DataLoader for one epoch (if called with multiple args) or combined data dict (if called with single arg)
    """
    # Check if this is the new signature (multiple arguments)
    import torch.utils.data
    if isinstance(train_data, (torch.utils.data.Subset, torch.utils.data.Dataset)) and batch_size is not None:
        # New signature - create epoch data loader
        import random
        from torch.utils.data import DataLoader, Subset
        
        total_samples = len(train_data)
        
        # Randomly sample indices for this epoch
        if samples_per_epoch and samples_per_epoch < total_samples:
            epoch_indices = random.sample(range(total_samples), samples_per_epoch)
            epoch_data = Subset(train_data, epoch_indices)
        else:
            epoch_data = train_data
        
        return DataLoader(
            epoch_data,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=pin_memory,
            num_workers=num_workers,
            persistent_workers=True if num_workers > 0 else False,
            drop_last=True,
        )
    
    # Old signature - load chunks (backward compatibility)
    chunk_metadata = train_data
    if "chunk_metadata" not in chunk_metadata:
        # This is old format data, return as-is
        return chunk_metadata
        
    print(f"Loading {chunk_metadata['num_chunks']} chunks for training...")
    
    # Initialize lists to collect data
    all_trajectories = []
    all_actions = []
    all_goals = []
    all_goal_ranks = []
    all_agents = []
    all_types = []
    all_consumption_labels = []
    all_sr_labels = []
    
    # Load and combine all chunks
    for chunk_info in chunk_metadata['chunk_metadata']:
        print(f"Loading chunk {chunk_info['chunk_idx']} from {chunk_info['file_path']}")
        chunk_data = torch.load(chunk_info['file_path'])
        
        all_trajectories.append(chunk_data['trajectories'])
        all_actions.append(chunk_data['actions'])
        all_goals.append(chunk_data['goals'])
        all_goal_ranks.append(chunk_data['goal_ranks'])
        all_agents.append(chunk_data['agents'])
        all_types.append(chunk_data['types'])
        all_consumption_labels.append(chunk_data['consumption_labels'])
        all_sr_labels.append(chunk_data['sr_labels'])
        
        # Free memory
        del chunk_data
    
    # Combine all data efficiently
    print("Combining all chunks...")
    import gc
    import psutil
    
    # Check available memory before combining
    process = psutil.Process()
    available_memory = psutil.virtual_memory().available / (1024**3)  # GB
    current_usage = process.memory_info().rss / (1024**3)  # GB
    print(f"  Available system memory: {available_memory:.2f} GB")
    print(f"  Current process memory: {current_usage:.2f} GB")
    
    # Use torch.cat with reduced memory footprint
    combined_data = {}
    
    # Process each data type separately to reduce peak memory usage
    for key, data_list in [
        ('trajectories', all_trajectories),
        ('actions', all_actions),
        ('goals', all_goals),
        ('goal_ranks', all_goal_ranks),
        ('agents', all_agents),
        ('types', all_types),
        ('consumption_labels', all_consumption_labels),
        ('sr_labels', all_sr_labels),
    ]:
        print(f"  Combining {key}...")
        
        # Check memory before combining
        if available_memory < 10.0:  # Less than 10GB available
            print(f"    WARNING: Low memory ({available_memory:.2f} GB). Using chunked processing...")
            # Process in smaller chunks if memory is low
            chunk_size = max(1, len(data_list) // 4)
            combined_chunks = []
            
            for i in range(0, len(data_list), chunk_size):
                chunk = torch.cat(data_list[i:i+chunk_size], dim=0)
                combined_chunks.append(chunk.numpy())
                del chunk
                gc.collect()
            
            # Final combination
            combined_data[key] = np.concatenate(combined_chunks, axis=0)
            del combined_chunks
        else:
            # Normal processing if enough memory
            combined_data[key] = torch.cat(data_list, dim=0).numpy()
        
        # Clear the list to free memory
        data_list.clear()
        gc.collect()
        
        # Update available memory
        available_memory = psutil.virtual_memory().available / (1024**3)
    
    # Clear all temporary lists
    del all_trajectories, all_actions, all_goals, all_goal_ranks
    del all_agents, all_types, all_consumption_labels, all_sr_labels
    gc.collect()
    
    print(f"Combined data shapes:")
    for key, value in combined_data.items():
        print(f"  {key}: {value.shape}")
    
    return combined_data


def validate_data_shape(data, data_type="data"):
    """
    Validate that data has the expected shape and structure

    Args:
        data: Data dictionary to validate
        data_type: Type of data (for error messages)

    Returns:
        bool: True if data is valid
    """
    if not isinstance(data, dict):
        raise ValueError(f"{data_type} must be a dictionary")

    required_keys = [
        "trajectories",
        "actions",
        "goals",
        "goal_ranks",
        "agents",
        "types",
        "consumption_labels",
        "sr_labels",
    ]

    for key in required_keys:
        if key not in data:
            raise ValueError(f"{data_type} missing required key: {key}")

    total_samples = data["trajectories"].shape[0]

    print(f"{data_type} validation:")
    print(f"  Total samples: {total_samples}")

    if total_samples == 0:
        raise ValueError(
            f"No {data_type} found. Please generate {data_type} first using appropriate flags."
        )

    return True


def log_data_shapes(data, data_type="data"):
    """
    Log the shapes of all data arrays for verification

    Args:
        data: Data dictionary
        data_type: Type of data (for logging)
    """
    print(f"{data_type} shapes:")
    for key, value in data.items():
        if hasattr(value, "shape"):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {type(value)}")
