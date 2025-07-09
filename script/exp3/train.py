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
    trajectories, goals, batch_size, n_past_min=1, n_past_max=5, max_n_past=5
):
    """
    Generate past episodes by randomly sampling from other trajectories in the batch
    with the same goal

    Args:
        trajectories: Batch of trajectories [batch_size, seq_len, channels, height, width]
        goals: Batch of goal labels [batch_size]
        batch_size: Size of current batch
        n_past_min: Minimum number of past episodes to sample
        n_past_max: Maximum number of past episodes to sample
        max_n_past: Maximum number of past episodes for consistent tensor shape

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

    # Generate random n_past values for all samples
    n_past_values = torch.randint(
        n_past_min, n_past_max + 1, (batch_size,), device=device
    )

    # Create goal similarity matrix
    same_goal_mask = goals.unsqueeze(1) == goals.unsqueeze(
        0
    )  # [batch_size, batch_size]
    same_goal_mask.fill_diagonal_(False)  # Exclude self-matches

    # For each sample, randomly select past episodes
    for i in range(batch_size):
        n_past = n_past_values[i].item()

        # Find indices of samples with same goal
        same_goal_indices = torch.nonzero(same_goal_mask[i], as_tuple=False).squeeze(1)

        if len(same_goal_indices) > 0:
            # Randomly select n_past episodes from same goal samples
            if len(same_goal_indices) >= n_past:
                selected_indices = same_goal_indices[
                    torch.randperm(len(same_goal_indices), device=device)[:n_past]
                ]
            else:
                # If not enough same goal samples, repeat some
                selected_indices = same_goal_indices[
                    torch.randperm(len(same_goal_indices), device=device)
                ]
                while len(selected_indices) < n_past:
                    additional = same_goal_indices[
                        torch.randperm(len(same_goal_indices), device=device)
                    ]
                    selected_indices = torch.cat([selected_indices, additional])
                selected_indices = selected_indices[:n_past]

            # Fill past episodes (ensure we don't exceed max_n_past)
            for j, idx in enumerate(selected_indices):
                if j < max_n_past:  # Ensure we don't exceed the allocated tensor size
                    past_episodes_batch[i, j] = trajectories[idx]

    return past_episodes_batch


def prepare_data_for_training(games, max_trajectory_length=100):
    """
    Prepare game data for training

    Args:
        games: List of game data from DataReader
        max_trajectory_length: Maximum length of trajectory to use

    Returns:
        Dictionary containing prepared training data
    """
    trajectories = []
    actions = []
    goals = []
    goal_rewards = []

    print(f"Preparing data from {len(games)} games...")

    for game in games:
        trajectory = game["trajectory_tensor"]  # [seq_len, channels, height, width]
        action_list = game["actions"]
        goal_tensor = game["goal_tensor"]  # [4] one-hot encoded

        # Truncate trajectory to max length
        seq_len = min(trajectory.shape[0], max_trajectory_length)
        trajectory = trajectory[:seq_len]
        action_list = action_list[:seq_len]

        # Pad if necessary
        if seq_len < max_trajectory_length:
            padding = np.zeros((max_trajectory_length - seq_len, *trajectory.shape[1:]))
            trajectory = np.concatenate([trajectory, padding], axis=0)
            action_list = action_list + [0] * (max_trajectory_length - len(action_list))

        trajectories.append(trajectory)
        actions.append(action_list)
        goals.append(np.argmax(goal_tensor))  # Convert one-hot to index
        goal_rewards.append(game["goal_rewards"])

    # Convert to tensors
    trajectories = torch.tensor(np.array(trajectories), dtype=torch.float32)
    actions = torch.tensor(np.array(actions), dtype=torch.long)
    goals = torch.tensor(np.array(goals), dtype=torch.long)
    goal_rewards = torch.tensor(np.array(goal_rewards), dtype=torch.float32)

    print(f"Data shapes:")
    print(f"  Trajectories: {trajectories.shape}")
    print(f"  Actions: {actions.shape}")
    print(f"  Goals: {goals.shape}")
    print(f"  Goal rewards: {goal_rewards.shape}")

    return {
        "trajectories": trajectories,
        "actions": actions,
        "goals": goals,
        "goal_rewards": goal_rewards,
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
    correct_actions = 0
    correct_goals = 0
    total_samples = 0

    for batch_idx, batch in enumerate(train_loader):
        trajectories, actions, goals = batch[:3]

        trajectories = trajectories.to(device)
        actions = actions.to(device)
        goals = goals.to(device)

        batch_size = trajectories.size(0)

        # Generate past episodes from batch
        past_episodes = generate_past_episodes_from_batch(
            trajectories, goals, batch_size, max_n_past=max_n_past
        )

        # Use trajectory up to previous timestep as input to MentalNet
        # MentalNet processes recent trajectory to predict action at current timestep
        current_timestep = data_config["time_step"] if data_config else 20

        # Recent trajectory: from start to current_timestep-1 (up to previous timestep)
        recent_trajectory = trajectories[
            :, :current_timestep
        ]  # [batch_size, seq_len, channels, height, width]

        # Action target: action at current_timestep
        if current_timestep < actions.size(1):
            action_targets = actions[:, current_timestep]  # Action at current timestep
        else:
            action_targets = actions[:, -1]  # Use last action if trajectory is shorter

        goal_targets = goals

        optimizer.zero_grad()

        # Forward pass
        action_logits, goal_logits, _, _ = model(past_episodes, recent_trajectory)

        # Compute loss
        total_loss_batch, action_loss_batch, goal_loss_batch = loss_fn(
            action_logits, goal_logits, action_targets, goal_targets
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
    action_accuracy = correct_actions / total_samples
    goal_accuracy = correct_goals / total_samples

    return {
        "loss": avg_loss,
        "action_loss": avg_action_loss,
        "goal_loss": avg_goal_loss,
        "action_accuracy": action_accuracy,
        "goal_accuracy": goal_accuracy,
    }


def validate_epoch(model, val_loader, loss_fn, device, max_n_past=5, data_config=None):
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
    correct_actions = 0
    correct_goals = 0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            trajectories, actions, goals = batch[:3]

            trajectories = trajectories.to(device)
            actions = actions.to(device)
            goals = goals.to(device)

            batch_size = trajectories.size(0)

            # Generate past episodes from batch
            past_episodes = generate_past_episodes_from_batch(
                trajectories, goals, batch_size, max_n_past=max_n_past
            )

            # Use trajectory up to previous timestep as input to MentalNet
            # MentalNet processes recent trajectory to predict action at current timestep
            current_timestep = data_config["time_step"] if data_config else 20

            # Recent trajectory: from start to current_timestep-1 (up to previous timestep)
            recent_trajectory = trajectories[:, :current_timestep]

            # Action target: action at current_timestep
            if current_timestep < actions.size(1):
                action_targets = actions[
                    :, current_timestep
                ]  # Action at current timestep
            else:
                action_targets = actions[
                    :, -1
                ]  # Use last action if trajectory is shorter

            goal_targets = goals

            # Forward pass
            action_logits, goal_logits, _, _ = model(past_episodes, recent_trajectory)

            # Compute loss
            total_loss_batch, action_loss_batch, goal_loss_batch = loss_fn(
                action_logits, goal_logits, action_targets, goal_targets
            )

            # Update metrics
            total_loss += total_loss_batch.item()
            total_action_loss += action_loss_batch.item()
            total_goal_loss += goal_loss_batch.item()

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
    action_accuracy = correct_actions / total_samples
    goal_accuracy = correct_goals / total_samples

    return {
        "loss": avg_loss,
        "action_loss": avg_action_loss,
        "goal_loss": avg_goal_loss,
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_save_dir = os.path.join(save_dir, f"exp3_{timestamp}")
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

    # Prepare data
    data = prepare_data_for_training(games, time_step)

    # Create datasets
    dataset = TensorDataset(data["trajectories"], data["actions"], data["goals"])

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
    model = create_model(model_kwargs)
    model = model.to(device)

    print(f"Model created with {count_parameters(model):,} parameters")

    # Loss function and optimizer
    loss_fn = ToMnetLoss(
        action_weight=training_process_config["action_weight"],
        goal_weight=training_process_config["goal_weight"],
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
        "train_action_accuracy": [],
        "train_goal_accuracy": [],
        "val_loss": [],
        "val_action_loss": [],
        "val_goal_loss": [],
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
        )

        # Validation
        val_metrics = validate_epoch(
            model, val_loader, loss_fn, device, max_n_past, data_config
        )

        epoch_time = time.time() - epoch_start_time

        # Update history
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_metrics["loss"])
        history["train_action_loss"].append(train_metrics["action_loss"])
        history["train_goal_loss"].append(train_metrics["goal_loss"])
        history["train_action_accuracy"].append(train_metrics["action_accuracy"])
        history["train_goal_accuracy"].append(train_metrics["goal_accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_action_loss"].append(val_metrics["action_loss"])
        history["val_goal_loss"].append(val_metrics["goal_loss"])
        history["val_action_accuracy"].append(val_metrics["action_accuracy"])
        history["val_goal_accuracy"].append(val_metrics["goal_accuracy"])
        history["epoch_time"].append(epoch_time)

        # Print metrics
        train_loss = train_metrics["loss"]
        train_acc = (
            (train_metrics["action_accuracy"] + train_metrics["goal_accuracy"])
            / 2
            * 100
        )
        val_acc = (
            (val_metrics["action_accuracy"] + val_metrics["goal_accuracy"]) / 2 * 100
        )
        train_action_loss = train_metrics["action_loss"]
        train_consumption_loss = train_metrics["goal_loss"]
        train_sr_loss = 0  # Placeholder for SR loss
        val_action_loss = val_metrics["action_loss"]
        val_consumption_loss = val_metrics["goal_loss"]
        val_sr_loss = 0  # Placeholder for SR loss

        print(
            f"Epoch: {epoch + 1:3d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}% | Val Acc: {val_acc:.4f}% | Time: {epoch_time:.2f}s"
        )
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
