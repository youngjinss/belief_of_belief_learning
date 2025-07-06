import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os
import json
import pickle
import sys
from datetime import datetime

sys.path.append("..")

from tomnet import ToMnet
from data_generation import generate_input_data
from config import Config
from visualize import create_additional_visualizations


"""
Advanced training system for ToMnetF
@Author Filip Borowiak
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
        self.best_loss = float('inf')
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
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False


def generate_past_episodes_from_batch(
    trajectories, goals, batch_size, n_past_min, n_past_max, max_n_past
):
    """
    Generate past episodes by randomly sampling from other trajectories in the batch
    with the same goal, using fully vectorized operations for efficiency

    Args:
        trajectories: Batch of trajectories [batch_size, depth, height, width, time_step]
        goals: Batch of goal labels [batch_size]
        batch_size: Size of current batch
        n_past_min: Minimum number of past episodes to sample
        n_past_max: Maximum number of past episodes to sample
        max_n_past: Maximum number of past episodes for consistent tensor shape

    Returns:
        past_episodes_batch: [batch_size, max_n_past, depth, height, width, time_step]
    """
    device = trajectories.device
    depth, height, width, time_step = trajectories.shape[1:]

    # Initialize past episodes tensor
    past_episodes_batch = torch.zeros(
        (batch_size, max_n_past, depth, height, width, time_step),
        dtype=trajectories.dtype,
        device=device,
    )

    # Generate random n_past values for all samples at once
    n_past_values = torch.randint(
        n_past_min, n_past_max + 1, (batch_size,), device=device
    )

    # Create goal similarity matrix (batch_size x batch_size)
    # same_goal_mask[i, j] = True if sample i and j have the same goal
    goals_expanded = goals.unsqueeze(1)  # [batch_size, 1]
    same_goal_mask = goals_expanded == goals.unsqueeze(0)  # [batch_size, batch_size]

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


def train_tomnet(config=None):
    """
    Train ToMnet model with comprehensive logging and evaluation

    Args:
        config: Config object containing all training parameters. If None, uses default values.
    """
    if config is None:
        config = Config()

    # Extract parameters from config
    data_dir = config.data_dir
    model_dir = config.model_dir
    result_dir = config.result_dir
    plot_dir = config.plot_dir
    log_dir = config.log_dir
    experiment_no = config.experiment_no
    epochs = config.epochs
    batch_size = config.batch_size
    lr = config.lr
    time_step = config.time_step
    height = config.height
    width = config.width
    depth = config.depth
    training_proportion = config.training_proportion
    use_percentage = config.use_percentage
    device = config.device

    # Create directories
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # Create timestamped log directory
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # log_path = os.path.join(log_dir, f"experiment{experiment_no}/{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    print(f"Training ToMnetF for experiment {experiment_no}")
    print(f"Log directory: {log_dir}")

    # Check if processed data exists, if not generate it
    processed_data_path = os.path.join(
        data_dir, f"processed_data_exp{experiment_no}.pkl"
    )
    if not os.path.exists(processed_data_path):
        print("Processed data not found. Generating...")
        processed_data = generate_input_data(
            data_dir=data_dir,
            output_dir=data_dir,
            use_percentage=use_percentage,
            time_step=time_step,
            height=height,
            width=width,
            depth=depth,
            experiment_no=experiment_no,
        )
    else:
        print("Loading existing processed data...")
        with open(processed_data_path, "rb") as f:
            processed_data = pickle.load(f)

    # Extract data
    data_traj = processed_data["data_trajectories"]
    data_curr = processed_data["data_current_state"]
    data_act = processed_data["data_actions"]
    data_labels = processed_data["data_labels"]
    data_consumption = processed_data.get("data_consumption_labels", None)
    data_sr = processed_data.get("data_sr_maps", None)

    print(f"Data shapes:")
    print(f"Trajectories: {data_traj.shape}")
    print(f"Current states: {data_curr.shape}")
    print(f"Actions: {data_act.shape}")
    print(f"Labels: {data_labels.shape}")

    # Convert labels to tensor if not already
    if not isinstance(data_labels, torch.Tensor):
        data_labels = torch.tensor(data_labels, dtype=torch.long)

    # Check if SR and consumption data are available
    has_new_labels = data_consumption is not None and data_sr is not None
    if has_new_labels:
        print(f"Consumption labels: {data_consumption.shape}")
        print(f"SR maps: {data_sr.shape}")
    else:
        print("Warning: SR and consumption labels not found. Using dummy labels.")

    # N_past will be handled by randomly sampling from batch data during training
    print(
        "N_past will be generated by randomly sampling from other trajectories in each batch"
    )

    # Create dataset - include labels for goal-based sampling
    dataset_components = [data_traj, data_curr, data_act, data_labels]
    if has_new_labels:
        dataset_components.extend([data_consumption, data_sr])

    dataset = TensorDataset(*dataset_components)

    # Train/validation split
    total_size = len(dataset)
    train_size = int(total_size * training_proportion)
    val_size = total_size - train_size

    # Create train and validation datasets with the same components
    train_components = [data[:train_size] for data in dataset_components]
    val_components = [data[train_size:] for data in dataset_components]

    train_dataset = TensorDataset(*train_components)
    val_dataset = TensorDataset(*val_components)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=True
    )

    print(f"Training size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")

    # Initialize model
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ToMnet(**config.get_model_kwargs())
    model = model.to(device)

    # Print model parameters
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {pytorch_total_params}")

    # Loss functions and optimizer
    action_loss_fn = torch.nn.CrossEntropyLoss()
    # For consumption: Use BCEWithLogitsLoss for numerical stability
    # This combines sigmoid + BCE loss and matches the negative log-likelihood formulation
    consumption_loss_fn = torch.nn.BCEWithLogitsLoss()
    sr_loss_fn = torch.nn.CrossEntropyLoss()  # Cross-entropy for SR distributions
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)

    # Loss weights
    action_weight = 1.0
    consumption_weight = 1.0
    sr_weight = 1.0
    
    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=config.early_stopping_patience,
        min_delta=config.early_stopping_min_delta,
        restore_best_weights=config.early_stopping_restore_best
    )

    # Training history
    train_history = {
        "train_accuracy": [],
        "train_loss": [],
        "train_action_loss": [],
        "train_consumption_loss": [],
        "train_sr_loss": [],
        "val_accuracy": [],
        "val_loss": [],
        "val_action_loss": [],
        "val_consumption_loss": [],
        "val_sr_loss": [],
        "epoch": [],
        "epoch_time": [],
    }

    best_val_acc = 0.0

    print("\n---TRAINING TOMNET NEURAL NETWORK---\n")
    print("-" * 80)

    # Training loop
    start = datetime.now()
    for epoch in range(epochs):
        epoch_start = datetime.now()

        running_loss_train = 0.0
        running_loss_val = 0.0
        running_action_loss_train = 0.0
        running_consumption_loss_train = 0.0
        running_sr_loss_train = 0.0
        running_action_loss_val = 0.0
        running_consumption_loss_val = 0.0
        running_sr_loss_val = 0.0

        all_pred_train = 0
        all_pred_val = 0

        correct_pred_train = 0
        correct_pred_val = 0

        # Training phase
        model.train()
        for idx, data in enumerate(train_loader):
            # Parse data based on what's available
            traj, curr, act, goals = data[0], data[1], data[2], data[3]
            traj, curr, act, goals = (
                traj.to(device),
                curr.to(device),
                act.to(device),
                goals.to(device),
            )

            data_idx = 4  # Updated since we now have goals at index 3

            # Handle consumption and SR labels
            if has_new_labels and len(data) > data_idx:
                consumption_target = data[data_idx].to(device)
                sr_target = data[data_idx + 1].to(device)
                data_idx += 2
            else:
                # Create dummy targets
                batch_size = act.size(0)
                consumption_target = torch.zeros(batch_size, 4).to(device)
                sr_target = torch.zeros(batch_size, 3, 13, 13).to(device)

            # Generate N_past data by randomly sampling from batch trajectories with same goal
            model_inputs = [traj, curr]

            # Generate past episodes from other trajectories in the batch with same goal
            past_episodes_batch = generate_past_episodes_from_batch(
                trajectories=traj,
                goals=goals,
                batch_size=traj.size(0),
                n_past_min=config.n_past_min,
                n_past_max=config.n_past_max,
                max_n_past=config.n_past_max,
            )
            model_inputs.append(past_episodes_batch)

            act = act.squeeze(-1).type(torch.long)

            optimizer.zero_grad()

            action_pred, consumption_pred, sr_pred = model(model_inputs)

            # Calculate losses
            action_loss = action_loss_fn(action_pred, act)
            consumption_loss = consumption_loss_fn(consumption_pred, consumption_target)

            # For SR loss, we need to reshape and apply cross-entropy per channel
            sr_loss = 0
            for i in range(3):  # 3 discount factors
                sr_pred_i = sr_pred[:, i, :, :].contiguous().view(batch_size, -1)
                sr_target_i = sr_target[:, i, :, :].contiguous().view(batch_size, -1)
                # Convert target to class indices (for now using argmax of dummy data)
                sr_target_indices = torch.argmax(sr_target_i, dim=1)
                sr_loss += sr_loss_fn(sr_pred_i, sr_target_indices)
            sr_loss = sr_loss / 3  # Average over discount factors

            # Combined loss
            loss = (
                action_weight * action_loss
                + consumption_weight * consumption_loss
                + sr_weight * sr_loss
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Calculate training accuracy (action prediction only for now)
            _, y_hat = torch.max(action_pred, 1)
            correct_pred_train += (y_hat == act).sum().item()
            all_pred_train += act.size(0)
            running_loss_train += loss.item()
            running_action_loss_train += action_loss.item()
            running_consumption_loss_train += consumption_loss.item()
            running_sr_loss_train += sr_loss.item()

        train_acc = 100 * correct_pred_train / all_pred_train
        train_loss = running_loss_train / len(train_loader)
        train_action_loss = running_action_loss_train / len(train_loader)
        train_consumption_loss = running_consumption_loss_train / len(train_loader)
        train_sr_loss = running_sr_loss_train / len(train_loader)

        # Validation phase
        model.eval()
        with torch.no_grad():
            for idx, data in enumerate(val_loader):
                # Parse data based on what's available
                traj, curr, act, goals = data[0], data[1], data[2], data[3]
                traj, curr, act, goals = (
                    traj.to(device),
                    curr.to(device),
                    act.to(device),
                    goals.to(device),
                )

                data_idx = 4  # Updated since we now have goals at index 3

                # Handle consumption and SR labels
                if has_new_labels and len(data) > data_idx:
                    consumption_target = data[data_idx].to(device)
                    sr_target = data[data_idx + 1].to(device)
                    data_idx += 2
                else:
                    # Create dummy targets
                    batch_size = act.size(0)
                    consumption_target = torch.zeros(batch_size, 4).to(device)
                    sr_target = torch.zeros(batch_size, 3, 13, 13).to(device)

                # Generate N_past data by randomly sampling from batch trajectories with same goal
                model_inputs = [traj, curr]

                # Generate past episodes from other trajectories in the batch with same goal
                past_episodes_batch = generate_past_episodes_from_batch(
                    trajectories=traj,
                    goals=goals,
                    batch_size=traj.size(0),
                    n_past_min=config.n_past_min,
                    n_past_max=config.n_past_max,
                    max_n_past=config.n_past_max,
                )
                model_inputs.append(past_episodes_batch)

                act = act.squeeze(-1).type(torch.long)

                action_pred, consumption_pred, sr_pred = model(model_inputs)

                # Calculate losses
                action_loss = action_loss_fn(action_pred, act)
                consumption_loss = consumption_loss_fn(
                    consumption_pred, consumption_target
                )

                # For SR loss
                sr_loss = 0
                for i in range(3):  # 3 discount factors
                    sr_pred_i = sr_pred[:, i, :, :].contiguous().view(batch_size, -1)
                    sr_target_i = (
                        sr_target[:, i, :, :].contiguous().view(batch_size, -1)
                    )
                    sr_target_indices = torch.argmax(sr_target_i, dim=1)
                    sr_loss += sr_loss_fn(sr_pred_i, sr_target_indices)
                sr_loss = sr_loss / 3

                # Combined loss
                loss = (
                    action_weight * action_loss
                    + consumption_weight * consumption_loss
                    + sr_weight * sr_loss
                )

                # Calculate validation accuracy (action prediction only for now)
                _, y_hat = torch.max(action_pred.data, 1)
                correct_pred_val += (y_hat == act).sum().item()
                all_pred_val += act.size(0)
                running_loss_val += loss.item()
                running_action_loss_val += action_loss.item()
                running_consumption_loss_val += consumption_loss.item()
                running_sr_loss_val += sr_loss.item()

        # Handle case where validation set is empty
        if all_pred_val > 0:
            val_acc = 100 * correct_pred_val / all_pred_val
            val_loss = running_loss_val / len(val_loader)
            val_action_loss = running_action_loss_val / len(val_loader)
            val_consumption_loss = running_consumption_loss_val / len(val_loader)
            val_sr_loss = running_sr_loss_val / len(val_loader)
        else:
            val_acc = 0.0
            val_loss = 0.0
            val_action_loss = 0.0
            val_consumption_loss = 0.0
            val_sr_loss = 0.0
            print(f"Warning: No validation data in epoch {epoch}")

        # Calculate epoch timing
        epoch_time = (datetime.now() - epoch_start).total_seconds()

        # Store history
        train_history["epoch"].append(int(epoch))
        train_history["train_accuracy"].append(float(train_acc))
        train_history["train_loss"].append(float(train_loss))
        train_history["train_action_loss"].append(float(train_action_loss))
        train_history["train_consumption_loss"].append(float(train_consumption_loss))
        train_history["train_sr_loss"].append(float(train_sr_loss))
        train_history["val_accuracy"].append(float(val_acc))
        train_history["val_loss"].append(float(val_loss))
        train_history["val_action_loss"].append(float(val_action_loss))
        train_history["val_consumption_loss"].append(float(val_consumption_loss))
        train_history["val_sr_loss"].append(float(val_sr_loss))
        train_history["epoch_time"].append(float(epoch_time))

        # Save best model if validation accuracy improved
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(model_dir, f"exp{experiment_no}_best.pth")
            torch.save(model.state_dict(), best_model_path)
        
        # Check early stopping
        if early_stopping(val_loss, model):
            print(f"Early stopping triggered at epoch {epoch}")
            print(f"Best validation loss: {early_stopping.best_loss:.4f}")
            break

        print(
            f"Epoch: {epoch:3d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}% | Val Acc: {val_acc:.4f}% | Time: {epoch_time:.2f}s"
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

    print("Finished Training!")

    # Save final model
    final_model_path = os.path.join(model_dir, f"exp{experiment_no}_final.pth")
    torch.save(model.state_dict(), final_model_path)

    # Ensure best_model_path is defined (fallback to final model if no best was saved)
    if "best_model_path" not in locals():
        best_model_path = final_model_path

    # Save training history
    history_path = os.path.join(result_dir, f"exp{experiment_no}_training_history.json")
    with open(history_path, "w") as f:
        json.dump(train_history, f, indent=2)

    # Create plots
    create_training_plots(train_history, plot_dir, experiment_no)

    # Create additional visualizations with model
    # Always use has_n_past=True since we generate past episodes from batch data
    create_additional_visualizations(
        model, val_loader, plot_dir, experiment_no, device, has_n_past=True
    )

    # Save training results
    results = {
        "experiment_no": experiment_no,
        "best_val_accuracy": float(best_val_acc),
        "final_train_accuracy": float(train_history["train_accuracy"][-1]),
        "final_val_accuracy": float(train_history["val_accuracy"][-1]),
        "epochs": epochs,
        "actual_epochs": len(train_history["epoch"]),
        "batch_size": batch_size,
        "learning_rate": lr,
        "model_parameters": int(pytorch_total_params),
        "timestamp": (datetime.now() - start).seconds,
        "best_model_path": best_model_path,
        "final_model_path": final_model_path,
        "early_stopping_triggered": len(train_history["epoch"]) < epochs,
        "early_stopping_best_loss": float(early_stopping.best_loss),
    }

    results_path = os.path.join(result_dir, f"exp{experiment_no}_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {results_path}")
    print(f"Best validation accuracy: {best_val_acc:.4f}%")

    return model, train_history, results


def create_training_plots(train_history, plot_dir, experiment_no):
    """Create and save training plots"""

    epochs = train_history["epoch"]

    # Accuracy plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_history["train_accuracy"], label="Training accuracy")
    plt.plot(epochs, train_history["val_accuracy"], label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.legend()
    plt.title("Training and Validation Accuracy")
    acc_path = os.path.join(plot_dir, f"exp{experiment_no}_accuracy.png")
    plt.savefig(acc_path)
    plt.close()

    # Loss plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_history["train_loss"], label="Training loss")
    plt.plot(epochs, train_history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.title("Training and Validation Loss")
    loss_path = os.path.join(plot_dir, f"exp{experiment_no}_loss.png")
    plt.savefig(loss_path)
    plt.close()

    print(f"Plots saved to: {plot_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train ToMnet model for Experiment 2")
    parser.add_argument(
        "--config_override",
        action="store_true",
        help="Override config with command line arguments",
    )
    parser.add_argument(
        "--data_dir", type=str, help="Directory containing training data"
    )
    parser.add_argument("--model_dir", type=str, help="Directory to save models")
    parser.add_argument("--result_dir", type=str, help="Directory to save results")
    parser.add_argument("--plot_dir", type=str, help="Directory to save plots")
    parser.add_argument("--log_dir", type=str, help="Directory to save logs")
    parser.add_argument("--experiment_no", type=int, help="Experiment number")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, help="Training batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--time_step", type=int, help="Trajectory size")
    parser.add_argument("--height", type=int, help="Map height")
    parser.add_argument("--width", type=int, help="Map width")
    parser.add_argument("--depth", type=int, help="Tensor depth")
    parser.add_argument(
        "--training_proportion", type=float, help="Train/val split proportion"
    )
    parser.add_argument(
        "--use_percentage", type=float, help="Percentage of data to use"
    )
    parser.add_argument("--device", type=str, help="CUDA device (e.g., cuda:0)")
    parser.add_argument("--early_stopping_patience", type=int, help="Early stopping patience")
    parser.add_argument("--early_stopping_min_delta", type=float, help="Early stopping minimum delta")

    args = parser.parse_args()

    config = Config()

    # Override config with command line arguments if specified
    if args.config_override:
        if args.data_dir is not None:
            config.data_dir = args.data_dir
        if args.model_dir is not None:
            config.model_dir = args.model_dir
        if args.result_dir is not None:
            config.result_dir = args.result_dir
        if args.plot_dir is not None:
            config.plot_dir = args.plot_dir
        if args.log_dir is not None:
            config.log_dir = args.log_dir
        if args.experiment_no is not None:
            config.experiment_no = args.experiment_no
        if args.epochs is not None:
            config.epochs = args.epochs
        if args.batch_size is not None:
            config.batch_size = args.batch_size
        if args.lr is not None:
            config.lr = args.lr
        if args.time_step is not None:
            config.time_step = args.time_step
        if args.height is not None:
            config.height = args.height
        if args.width is not None:
            config.width = args.width
        if args.depth is not None:
            config.depth = args.depth
        if args.training_proportion is not None:
            config.training_proportion = args.training_proportion
        if args.use_percentage is not None:
            config.use_percentage = args.use_percentage
        if args.device is not None:
            config.device = args.device
        if args.early_stopping_patience is not None:
            config.early_stopping_patience = args.early_stopping_patience
        if args.early_stopping_min_delta is not None:
            config.early_stopping_min_delta = args.early_stopping_min_delta

    # Train model using config parameters
    model, history, results = train_tomnet(config)
