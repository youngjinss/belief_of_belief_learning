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

"""
Advanced training system for ToMnetF
@Author Filip Borowiak
"""


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
    ts = config.ts
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"exp{experiment_no}_{timestamp}")
    os.makedirs(log_path, exist_ok=True)

    print(f"Training ToMnetF for experiment {experiment_no}")
    print(f"Log directory: {log_path}")

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
            ts=ts,
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

    # Check if SR and consumption data are available
    has_new_labels = data_consumption is not None and data_sr is not None
    if has_new_labels:
        print(f"Consumption labels: {data_consumption.shape}")
        print(f"SR maps: {data_sr.shape}")
    else:
        print("Warning: SR and consumption labels not found. Using dummy labels.")

    # Create dataset
    if has_new_labels:
        dataset = TensorDataset(
            data_traj, data_curr, data_act, data_consumption, data_sr
        )
    else:
        dataset = TensorDataset(data_traj, data_curr, data_act)

    # Train/validation split
    total_size = len(dataset)
    train_size = int(total_size * training_proportion)
    val_size = total_size - train_size

    if has_new_labels:
        train_dataset = TensorDataset(
            data_traj[:train_size],
            data_curr[:train_size],
            data_act[:train_size],
            data_consumption[:train_size],
            data_sr[:train_size],
        )
        val_dataset = TensorDataset(
            data_traj[train_size:],
            data_curr[train_size:],
            data_act[train_size:],
            data_consumption[train_size:],
            data_sr[train_size:],
        )
    else:
        train_dataset = TensorDataset(
            data_traj[:train_size], data_curr[:train_size], data_act[:train_size]
        )
        val_dataset = TensorDataset(
            data_traj[train_size:], data_curr[train_size:], data_act[train_size:]
        )

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

    # Training history
    train_history = {
        "train_accuracy": [],
        "train_loss": [],
        "val_accuracy": [],
        "val_loss": [],
        "epoch": [],
    }

    best_val_acc = 0.0

    print("\n---TRAINING TOMNET NEURAL NETWORK---\n")
    print("-" * 80)

    # Training loop
    for epoch in range(epochs):
        running_loss_train = 0.0
        running_loss_val = 0.0

        all_pred_train = 0
        all_pred_val = 0

        correct_pred_train = 0
        correct_pred_val = 0

        # Training phase
        model.train()
        for idx, data in enumerate(train_loader):
            if len(data) == 5:  # With SR and consumption labels
                traj, curr, act, consumption_target, sr_target = data
                traj, curr, act = traj.to(device), curr.to(device), act.to(device)
                consumption_target = consumption_target.to(device)
                sr_target = sr_target.to(device)
            else:  # Without SR and consumption labels
                traj, curr, act = data
                traj, curr, act = traj.to(device), curr.to(device), act.to(device)
                # Create dummy targets
                batch_size = act.size(0)
                consumption_target = torch.zeros(batch_size, 4).to(device)
                sr_target = torch.zeros(batch_size, 3, 13, 13).to(device)

            act = act.squeeze(-1).type(torch.long)

            optimizer.zero_grad()

            action_pred, consumption_pred, sr_pred = model([traj, curr])

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

        train_acc = 100 * correct_pred_train / all_pred_train
        train_loss = running_loss_train / len(train_loader)

        # Validation phase
        model.eval()
        with torch.no_grad():
            for idx, data in enumerate(val_loader):
                if len(data) == 5:  # With SR and consumption labels
                    traj, curr, act, consumption_target, sr_target = data
                    traj, curr, act = traj.to(device), curr.to(device), act.to(device)
                    consumption_target = consumption_target.to(device)
                    sr_target = sr_target.to(device)
                else:  # Without SR and consumption labels
                    traj, curr, act = data
                    traj, curr, act = traj.to(device), curr.to(device), act.to(device)
                    # Create dummy targets
                    batch_size = act.size(0)
                    consumption_target = torch.zeros(batch_size, 4).to(device)
                    sr_target = torch.zeros(batch_size, 3, 13, 13).to(device)

                act = act.squeeze(-1).type(torch.long)

                action_pred, consumption_pred, sr_pred = model([traj, curr])

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

        # Handle case where validation set is empty
        if all_pred_val > 0:
            val_acc = 100 * correct_pred_val / all_pred_val
            val_loss = running_loss_val / len(val_loader)
        else:
            val_acc = 0.0
            val_loss = 0.0
            print(f"Warning: No validation data in epoch {epoch}")

        # Store history
        train_history["epoch"].append(int(epoch))
        train_history["train_accuracy"].append(float(train_acc))
        train_history["train_loss"].append(float(train_loss))
        train_history["val_accuracy"].append(float(val_acc))
        train_history["val_loss"].append(float(val_loss))

        # Save best model if validation accuracy improved
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(model_dir, f"exp{experiment_no}_best.pth")
            torch.save(model.state_dict(), best_model_path)

        print(
            f"Epoch: {epoch:3d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}% | Val Acc: {val_acc:.4f}%"
        )
        print("-" * 80)

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

    # Save training results
    results = {
        "experiment_no": experiment_no,
        "best_val_accuracy": float(best_val_acc),
        "final_train_accuracy": float(train_history["train_accuracy"][-1]),
        "final_val_accuracy": float(train_history["val_accuracy"][-1]),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "model_parameters": int(pytorch_total_params),
        "timestamp": timestamp,
        "best_model_path": best_model_path,
        "final_model_path": final_model_path,
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
    parser.add_argument("--ts", type=int, help="Trajectory size")
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
        if args.ts is not None:
            config.ts = args.ts
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

    # Train model using config parameters
    model, history, results = train_tomnet(config)
