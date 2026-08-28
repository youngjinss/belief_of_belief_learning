"""Per-epoch console output and training-curve plots."""

import os

import matplotlib.pyplot as plt

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
