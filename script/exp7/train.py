import os
import json
import sys

import torch

try:
    from torch.amp import autocast
except ImportError:
    from torch.cuda.amp import autocast

# Add current directory to path
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from config import Config
from utils import (
    set_seed,
    load_chunked_data_for_training,
    generate_past_episodes_from_batch,
    save_training_plots,
    print_epoch_metrics,
    setup_training_environment,
    setup_model_and_data,
)

# Set seed using Config default value
config = Config()
set_seed(config.seed)

"""
Training system for KeyDoor ToMnet implementation
Adapted from ToMnetF experiment5 for KeyDoor environment
@author: Based on ToMnetF implementation, adapted for KeyDoor
"""


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

    # Separate metrics for achievers and blockers
    achiever_correct_actions = 0
    achiever_correct_goals = 0
    achiever_total_samples = 0
    blocker_correct_actions = 0
    blocker_correct_goals = 0
    blocker_total_samples = 0

    # Agent-specific loss tracking
    achiever_total_action_loss = 0
    achiever_total_goal_loss = 0
    achiever_total_consumption_loss = 0
    achiever_total_sr_loss = 0
    blocker_total_action_loss = 0
    blocker_total_goal_loss = 0
    blocker_total_consumption_loss = 0
    blocker_total_sr_loss = 0

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
        # Each sample has a different effective length
        batch_size = trajectories.size(0)
        seq_len = trajectories.size(1)  # Fixed sequence length

        # Concatenate current and past episodes
        combined_episodes = torch.cat([trajectories.unsqueeze(1), past_episodes], dim=1)

        # Prepare inputs for ToMnet
        # Split combined_episodes into past_trajectories and recent_trajectory
        recent_trajectory = combined_episodes[:, 0]  # Current trajectory
        past_trajectories = (
            combined_episodes[:, 1:] if combined_episodes.size(1) > 1 else None
        )  # Past episodes

        # Extract current state from recent trajectory (last timestep)
        current_state = recent_trajectory[:, -1]  # Last timestep of recent trajectory

        if scaler is not None:
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            with autocast(device_type):
                # Forward pass
                outputs = model(past_trajectories, recent_trajectory, current_state)

                # Compute loss
                loss_dict = loss_fn(
                    outputs["action_logits"],
                    outputs["goal_logits"],
                    outputs["agent_logits"],
                    outputs["type_logits"],
                    outputs["consumption_logits"],
                    outputs["sr_pred"],
                    actions[:, 0],  # Get action target (first element)
                    torch.argmax(goals, dim=1),  # Convert one-hot to class indices
                    agents,
                    types,
                    consumption_labels,
                    sr_labels,
                )

                loss = loss_dict["loss"]

                # Scale loss for gradient accumulation
                loss = loss / gradient_accumulation_steps

                # Accumulate gradients
                accumulation_loss += loss.item()
        else:
            # Forward pass without mixed precision
            outputs = model(past_trajectories, recent_trajectory, current_state)

            # Compute loss
            loss_dict = loss_fn(
                outputs["action_logits"],
                outputs["goal_logits"],
                outputs["agent_logits"],
                outputs["type_logits"],
                outputs["consumption_logits"],
                outputs["sr_pred"],
                actions[:, 0],  # Get action target (first element)
                torch.argmax(goals, dim=1),  # Convert one-hot to class indices
                agents,
                types,
                consumption_labels,
                sr_labels,
            )

            loss = loss_dict["loss"]

            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps

            # Accumulate gradients
            accumulation_loss += loss.item()

        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Update parameters every gradient_accumulation_steps
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

            # Add accumulated loss to total
            total_loss += accumulation_loss
            accumulation_loss = 0

        # Accumulate metrics
        total_action_loss += loss_dict["action_loss"].item()
        total_goal_loss += loss_dict["goal_loss"].item()
        total_agent_loss += loss_dict["agent_loss"].item()
        total_type_loss += loss_dict["type_loss"].item()
        total_consumption_loss += loss_dict["consumption_loss"].item()
        total_sr_loss += loss_dict["sr_loss"].item()

        # Calculate accuracies
        action_preds = torch.argmax(outputs["action_logits"], dim=1)
        goal_preds = torch.argmax(outputs["goal_logits"], dim=1)
        agent_preds = torch.argmax(outputs["agent_logits"], dim=1)
        type_preds = torch.argmax(outputs["type_logits"], dim=1)

        # Convert goals to class indices (they are one-hot encoded)
        goals_indices = torch.argmax(goals, dim=1)

        # Mask padded actions (-1) for accuracy calculation
        action_mask = actions[:, 0] != -1
        valid_action_samples = action_mask.sum().item()
        
        if valid_action_samples > 0:
            correct_actions += (action_preds[action_mask] == actions[action_mask, 0]).sum().item()
            # Update total samples to only count valid actions
            total_samples += valid_action_samples
        else:
            total_samples += batch_size
        
        correct_goals += (goal_preds == goals_indices).sum().item()
        correct_agents += (agent_preds == agents).sum().item()
        correct_types += (type_preds == types).sum().item()

        # Agent-specific metrics
        achiever_mask = agents == 0
        blocker_mask = agents == 1

        # Apply action mask for agent-specific metrics
        achiever_valid_mask = achiever_mask & action_mask
        blocker_valid_mask = blocker_mask & action_mask

        achiever_correct_actions += (
            (action_preds[achiever_valid_mask] == actions[achiever_valid_mask, 0]).sum().item()
        )
        achiever_correct_goals += (
            (goal_preds[achiever_mask] == goals_indices[achiever_mask]).sum().item()
        )
        achiever_total_samples += achiever_valid_mask.sum().item()

        blocker_correct_actions += (
            (action_preds[blocker_valid_mask] == actions[blocker_valid_mask, 0]).sum().item()
        )
        blocker_correct_goals += (
            (goal_preds[blocker_mask] == goals_indices[blocker_mask]).sum().item()
        )
        blocker_total_samples += blocker_valid_mask.sum().item()

        # Agent-specific loss accumulation
        if achiever_mask.sum() > 0:
            achiever_total_action_loss += (
                loss_dict["action_loss"].item()
                * achiever_mask.sum().item()
                / batch_size
            )
            achiever_total_goal_loss += (
                loss_dict["goal_loss"].item() * achiever_mask.sum().item() / batch_size
            )
            achiever_total_consumption_loss += (
                loss_dict["consumption_loss"].item()
                * achiever_mask.sum().item()
                / batch_size
            )
            achiever_total_sr_loss += (
                loss_dict["sr_loss"].item() * achiever_mask.sum().item() / batch_size
            )

        if blocker_mask.sum() > 0:
            blocker_total_action_loss += (
                loss_dict["action_loss"].item() * blocker_mask.sum().item() / batch_size
            )
            blocker_total_goal_loss += (
                loss_dict["goal_loss"].item() * blocker_mask.sum().item() / batch_size
            )
            blocker_total_consumption_loss += (
                loss_dict["consumption_loss"].item()
                * blocker_mask.sum().item()
                / batch_size
            )
            blocker_total_sr_loss += (
                loss_dict["sr_loss"].item() * blocker_mask.sum().item() / batch_size
            )

    # Handle remaining gradients if gradient accumulation is used
    if accumulation_loss > 0:
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()
        total_loss += accumulation_loss

    # Calculate averages
    num_batches = len(train_loader)
    avg_loss = total_loss / num_batches
    avg_action_loss = total_action_loss / num_batches
    avg_goal_loss = total_goal_loss / num_batches
    avg_agent_loss = total_agent_loss / num_batches
    avg_type_loss = total_type_loss / num_batches
    avg_consumption_loss = total_consumption_loss / num_batches
    avg_sr_loss = total_sr_loss / num_batches

    # Calculate accuracies
    action_accuracy = correct_actions / total_samples if total_samples > 0 else 0
    goal_accuracy = correct_goals / total_samples if total_samples > 0 else 0
    agent_accuracy = correct_agents / total_samples if total_samples > 0 else 0
    type_accuracy = correct_types / total_samples if total_samples > 0 else 0

    # Agent-specific accuracies
    achiever_action_accuracy = (
        achiever_correct_actions / achiever_total_samples
        if achiever_total_samples > 0
        else 0
    )
    achiever_goal_accuracy = (
        achiever_correct_goals / achiever_total_samples
        if achiever_total_samples > 0
        else 0
    )
    blocker_action_accuracy = (
        blocker_correct_actions / blocker_total_samples
        if blocker_total_samples > 0
        else 0
    )
    blocker_goal_accuracy = (
        blocker_correct_goals / blocker_total_samples
        if blocker_total_samples > 0
        else 0
    )

    # Agent-specific average losses
    achiever_avg_action_loss = achiever_total_action_loss / num_batches
    achiever_avg_goal_loss = achiever_total_goal_loss / num_batches
    achiever_avg_consumption_loss = achiever_total_consumption_loss / num_batches
    achiever_avg_sr_loss = achiever_total_sr_loss / num_batches

    blocker_avg_action_loss = blocker_total_action_loss / num_batches
    blocker_avg_goal_loss = blocker_total_goal_loss / num_batches
    blocker_avg_consumption_loss = blocker_total_consumption_loss / num_batches
    blocker_avg_sr_loss = blocker_total_sr_loss / num_batches

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
        "achiever_action_accuracy": achiever_action_accuracy,
        "achiever_goal_accuracy": achiever_goal_accuracy,
        "blocker_action_accuracy": blocker_action_accuracy,
        "blocker_goal_accuracy": blocker_goal_accuracy,
        "achiever_action_loss": achiever_avg_action_loss,
        "achiever_goal_loss": achiever_avg_goal_loss,
        "achiever_consumption_loss": achiever_avg_consumption_loss,
        "achiever_sr_loss": achiever_avg_sr_loss,
        "blocker_action_loss": blocker_avg_action_loss,
        "blocker_goal_loss": blocker_avg_goal_loss,
        "blocker_consumption_loss": blocker_avg_consumption_loss,
        "blocker_sr_loss": blocker_avg_sr_loss,
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

    # Separate metrics for achievers and blockers
    achiever_correct_actions = 0
    achiever_correct_goals = 0
    achiever_total_samples = 0
    blocker_correct_actions = 0
    blocker_correct_goals = 0
    blocker_total_samples = 0

    # Agent-specific loss tracking
    achiever_total_action_loss = 0
    achiever_total_goal_loss = 0
    achiever_total_consumption_loss = 0
    achiever_total_sr_loss = 0
    blocker_total_action_loss = 0
    blocker_total_goal_loss = 0
    blocker_total_consumption_loss = 0
    blocker_total_sr_loss = 0

    with torch.no_grad():
        for batch in val_loader:
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
            # Each sample has a different effective length

            # Concatenate current and past episodes
            combined_episodes = torch.cat(
                [trajectories.unsqueeze(1), past_episodes], dim=1
            )

            # Prepare inputs for ToMnet
            # Split combined_episodes into past_trajectories and recent_trajectory
            recent_trajectory = combined_episodes[:, 0]  # Current trajectory
            past_trajectories = (
                combined_episodes[:, 1:] if combined_episodes.size(1) > 1 else None
            )  # Past episodes

            # Extract current state from recent trajectory (last timestep)
            current_state = recent_trajectory[
                :, -1
            ]  # Last timestep of recent trajectory

            # Forward pass
            if scaler is not None:
                device_type = "cuda" if torch.cuda.is_available() else "cpu"
                with autocast(device_type):
                    outputs = model(past_trajectories, recent_trajectory, current_state)
                    loss_dict = loss_fn(
                        outputs["action_logits"],
                        outputs["goal_logits"],
                        outputs["agent_logits"],
                        outputs["type_logits"],
                        outputs["consumption_logits"],
                        outputs["sr_pred"],
                        actions[:, 0],  # Get action target (first element)
                        torch.argmax(goals, dim=1),  # Convert one-hot to class indices
                        agents,
                        types,
                        consumption_labels,
                        sr_labels,
                    )
            else:
                outputs = model(past_trajectories, recent_trajectory, current_state)
                loss_dict = loss_fn(
                    outputs["action_logits"],
                    outputs["goal_logits"],
                    outputs["agent_logits"],
                    outputs["type_logits"],
                    outputs["consumption_logits"],
                    outputs["sr_pred"],
                    actions[:, 0],  # Get action target (first element)
                    torch.argmax(goals, dim=1),  # Convert one-hot to class indices
                    agents,
                    types,
                    consumption_labels,
                    sr_labels,
                )

            # Accumulate losses
            total_loss += loss_dict["loss"].item()
            total_action_loss += loss_dict["action_loss"].item()
            total_goal_loss += loss_dict["goal_loss"].item()
            total_agent_loss += loss_dict["agent_loss"].item()
            total_type_loss += loss_dict["type_loss"].item()
            total_consumption_loss += loss_dict["consumption_loss"].item()
            total_sr_loss += loss_dict["sr_loss"].item()

            # Calculate accuracies
            action_preds = torch.argmax(outputs["action_logits"], dim=1)
            goal_preds = torch.argmax(outputs["goal_logits"], dim=1)
            agent_preds = torch.argmax(outputs["agent_logits"], dim=1)
            type_preds = torch.argmax(outputs["type_logits"], dim=1)

            # Convert goals to class indices (they are one-hot encoded)
            goals_indices = torch.argmax(goals, dim=1)

            # Mask padded actions (-1) for accuracy calculation
            action_mask = actions[:, 0] != -1
            valid_action_samples = action_mask.sum().item()
            
            if valid_action_samples > 0:
                correct_actions += (action_preds[action_mask] == actions[action_mask, 0]).sum().item()
                total_samples += valid_action_samples
            else:
                total_samples += batch_size

            correct_goals += (goal_preds == goals_indices).sum().item()
            correct_agents += (agent_preds == agents).sum().item()
            correct_types += (type_preds == types).sum().item()

            # Agent-specific metrics
            achiever_mask = agents == 0
            blocker_mask = agents == 1

            # Apply action mask for agent-specific metrics
            achiever_valid_mask = achiever_mask & action_mask
            blocker_valid_mask = blocker_mask & action_mask

            achiever_correct_actions += (
                (action_preds[achiever_valid_mask] == actions[achiever_valid_mask, 0]).sum().item()
            )
            achiever_correct_goals += (
                (goal_preds[achiever_mask] == goals_indices[achiever_mask]).sum().item()
            )
            achiever_total_samples += achiever_valid_mask.sum().item()

            blocker_correct_actions += (
                (action_preds[blocker_valid_mask] == actions[blocker_valid_mask, 0]).sum().item()
            )
            blocker_correct_goals += (
                (goal_preds[blocker_mask] == goals_indices[blocker_mask]).sum().item()
            )
            blocker_total_samples += blocker_valid_mask.sum().item()

            # Agent-specific loss accumulation
            if achiever_mask.sum() > 0:
                achiever_total_action_loss += (
                    loss_dict["action_loss"].item()
                    * achiever_mask.sum().item()
                    / batch_size
                )
                achiever_total_goal_loss += (
                    loss_dict["goal_loss"].item()
                    * achiever_mask.sum().item()
                    / batch_size
                )
                achiever_total_consumption_loss += (
                    loss_dict["consumption_loss"].item()
                    * achiever_mask.sum().item()
                    / batch_size
                )
                achiever_total_sr_loss += (
                    loss_dict["sr_loss"].item()
                    * achiever_mask.sum().item()
                    / batch_size
                )

            if blocker_mask.sum() > 0:
                blocker_total_action_loss += (
                    loss_dict["action_loss"].item()
                    * blocker_mask.sum().item()
                    / batch_size
                )
                blocker_total_goal_loss += (
                    loss_dict["goal_loss"].item()
                    * blocker_mask.sum().item()
                    / batch_size
                )
                blocker_total_consumption_loss += (
                    loss_dict["consumption_loss"].item()
                    * blocker_mask.sum().item()
                    / batch_size
                )
                blocker_total_sr_loss += (
                    loss_dict["sr_loss"].item() * blocker_mask.sum().item() / batch_size
                )

    # Calculate averages
    num_batches = len(val_loader)
    avg_loss = total_loss / num_batches
    avg_action_loss = total_action_loss / num_batches
    avg_goal_loss = total_goal_loss / num_batches
    avg_agent_loss = total_agent_loss / num_batches
    avg_type_loss = total_type_loss / num_batches
    avg_consumption_loss = total_consumption_loss / num_batches
    avg_sr_loss = total_sr_loss / num_batches

    # Calculate accuracies
    action_accuracy = correct_actions / total_samples if total_samples > 0 else 0
    goal_accuracy = correct_goals / total_samples if total_samples > 0 else 0
    agent_accuracy = correct_agents / total_samples if total_samples > 0 else 0
    type_accuracy = correct_types / total_samples if total_samples > 0 else 0

    # Agent-specific accuracies
    achiever_action_accuracy = (
        achiever_correct_actions / achiever_total_samples
        if achiever_total_samples > 0
        else 0
    )
    achiever_goal_accuracy = (
        achiever_correct_goals / achiever_total_samples
        if achiever_total_samples > 0
        else 0
    )
    blocker_action_accuracy = (
        blocker_correct_actions / blocker_total_samples
        if blocker_total_samples > 0
        else 0
    )
    blocker_goal_accuracy = (
        blocker_correct_goals / blocker_total_samples
        if blocker_total_samples > 0
        else 0
    )

    # Agent-specific average losses
    achiever_avg_action_loss = achiever_total_action_loss / num_batches
    achiever_avg_goal_loss = achiever_total_goal_loss / num_batches
    achiever_avg_consumption_loss = achiever_total_consumption_loss / num_batches
    achiever_avg_sr_loss = achiever_total_sr_loss / num_batches

    blocker_avg_action_loss = blocker_total_action_loss / num_batches
    blocker_avg_goal_loss = blocker_total_goal_loss / num_batches
    blocker_avg_consumption_loss = blocker_total_consumption_loss / num_batches
    blocker_avg_sr_loss = blocker_total_sr_loss / num_batches

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
        "achiever_action_accuracy": achiever_action_accuracy,
        "achiever_goal_accuracy": achiever_goal_accuracy,
        "blocker_action_accuracy": blocker_action_accuracy,
        "blocker_goal_accuracy": blocker_goal_accuracy,
        "achiever_action_loss": achiever_avg_action_loss,
        "achiever_goal_loss": achiever_avg_goal_loss,
        "achiever_consumption_loss": achiever_avg_consumption_loss,
        "achiever_sr_loss": achiever_avg_sr_loss,
        "blocker_action_loss": blocker_avg_action_loss,
        "blocker_goal_loss": blocker_avg_goal_loss,
        "blocker_consumption_loss": blocker_avg_consumption_loss,
        "blocker_sr_loss": blocker_avg_sr_loss,
    }


def run_training_loop(
    model,
    train_data,
    val_loader,
    optimizer,
    loss_fn,
    device,
    scaler,
    early_stopping,
    epochs,
    max_n_past,
    data_config,
    training_process_config,
    model_config,
    gradient_accumulation_steps,
    experiment_save_dir,
    samples_per_epoch,
    batch_size,
    pin_memory,
    num_workers,
):
    """
    Main training loop extracted from train_tomnet function

    Returns:
        dict: Training history
    """
    import time
    import sys
    import json
    import gc

    # Initialize training history
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
        "train_achiever_action_accuracy": [],
        "train_achiever_goal_accuracy": [],
        "train_blocker_action_accuracy": [],
        "train_blocker_goal_accuracy": [],
        "train_achiever_action_loss": [],
        "train_achiever_goal_loss": [],
        "train_achiever_consumption_loss": [],
        "train_achiever_sr_loss": [],
        "train_blocker_action_loss": [],
        "train_blocker_goal_loss": [],
        "train_blocker_consumption_loss": [],
        "train_blocker_sr_loss": [],
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
        "val_achiever_action_accuracy": [],
        "val_achiever_goal_accuracy": [],
        "val_blocker_action_accuracy": [],
        "val_blocker_goal_accuracy": [],
        "val_achiever_action_loss": [],
        "val_achiever_goal_loss": [],
        "val_achiever_consumption_loss": [],
        "val_achiever_sr_loss": [],
        "val_blocker_action_loss": [],
        "val_blocker_goal_loss": [],
        "val_blocker_consumption_loss": [],
        "val_blocker_sr_loss": [],
    }

    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        epoch_start_time = time.time()

        # Create training data loader for this epoch
        train_loader = load_chunked_data_for_training(
            train_data,
            batch_size,
            samples_per_epoch,
            pin_memory,
            num_workers,
        )

        # Training phase
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

        # Validation phase
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

        # Training metrics
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
        history["train_achiever_action_accuracy"].append(
            train_metrics["achiever_action_accuracy"]
        )
        history["train_achiever_goal_accuracy"].append(
            train_metrics["achiever_goal_accuracy"]
        )
        history["train_blocker_action_accuracy"].append(
            train_metrics["blocker_action_accuracy"]
        )
        history["train_blocker_goal_accuracy"].append(
            train_metrics["blocker_goal_accuracy"]
        )
        history["train_achiever_action_loss"].append(
            train_metrics["achiever_action_loss"]
        )
        history["train_achiever_goal_loss"].append(train_metrics["achiever_goal_loss"])
        history["train_achiever_consumption_loss"].append(
            train_metrics["achiever_consumption_loss"]
        )
        history["train_achiever_sr_loss"].append(train_metrics["achiever_sr_loss"])
        history["train_blocker_action_loss"].append(
            train_metrics["blocker_action_loss"]
        )
        history["train_blocker_goal_loss"].append(train_metrics["blocker_goal_loss"])
        history["train_blocker_consumption_loss"].append(
            train_metrics["blocker_consumption_loss"]
        )
        history["train_blocker_sr_loss"].append(train_metrics["blocker_sr_loss"])

        # Validation metrics
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
        history["val_achiever_action_accuracy"].append(
            val_metrics["achiever_action_accuracy"]
        )
        history["val_achiever_goal_accuracy"].append(
            val_metrics["achiever_goal_accuracy"]
        )
        history["val_blocker_action_accuracy"].append(
            val_metrics["blocker_action_accuracy"]
        )
        history["val_blocker_goal_accuracy"].append(
            val_metrics["blocker_goal_accuracy"]
        )
        history["val_achiever_action_loss"].append(val_metrics["achiever_action_loss"])
        history["val_achiever_goal_loss"].append(val_metrics["achiever_goal_loss"])
        history["val_achiever_consumption_loss"].append(
            val_metrics["achiever_consumption_loss"]
        )
        history["val_achiever_sr_loss"].append(val_metrics["achiever_sr_loss"])
        history["val_blocker_action_loss"].append(val_metrics["blocker_action_loss"])
        history["val_blocker_goal_loss"].append(val_metrics["blocker_goal_loss"])
        history["val_blocker_consumption_loss"].append(
            val_metrics["blocker_consumption_loss"]
        )
        history["val_blocker_sr_loss"].append(val_metrics["blocker_sr_loss"])

        # Print epoch metrics
        print_epoch_metrics(epoch, epoch_time, train_metrics, val_metrics)

        # Save model if validation loss improved
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0

            # Save best model
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

            print(f"New best validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1

        # Check early stopping
        if early_stopping(val_metrics["loss"], model):
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(
                experiment_save_dir, f"checkpoint_epoch_{epoch + 1}.pth"
            )
            if isinstance(model, torch.nn.DataParallel):
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.module.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_val_loss": best_val_loss,
                        "history": history,
                    },
                    checkpoint_path,
                )
            else:
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_val_loss": best_val_loss,
                        "history": history,
                    },
                    checkpoint_path,
                )

        # Memory cleanup
        del train_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("Training completed!")
    return history


def train_tomnet(
    data_dir=None,
    save_dir="./results/exp7/combined",
    config=None,
):
    """
    Main training function for KeyDoor ToMnet
    Trains on data from all achiever-blocker combinations in a single training process

    Args:
        data_dir: Directory containing training data
        save_dir: Directory to save results
        config: Configuration object

    Returns:
        Training history dictionary
    """
    if config is None:
        config = Config()

    # Use default data directory if not provided
    if data_dir is None:
        # Get the appropriate data directory based on single-agent vs multi-agent mode
        if config.is_single_agent_mode():
            # For single-agent mode, use the first achiever type
            achiever_type = list(config.achiever_types.keys())[0]
            data_dir = config.get_training_data_path(achiever_type, None, is_test=False)
        else:
            # For multi-agent mode, use combined data
            env_name = config.get_env_name()
            data_dir = f"./data/{env_name}/combined"

    # Agent type (we'll focus on one agent type for now)
    agent_type = "achiever"  # or "blocker"

    # Extract configuration
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
    device_setting = training_kwargs["device"]
    patience = training_kwargs["patience"]
    min_delta = training_kwargs["min_delta"]

    # Setup
    experiment_save_dir = save_dir
    os.makedirs(experiment_save_dir, exist_ok=True)

    # Setup training environment
    (
        device,
        use_parallel,
        device_ids,
        use_amp,
        gradient_accumulation_steps,
        other_configs,
    ) = setup_training_environment(
        config, training_kwargs, training_config, device_setting
    )

    pin_memory = other_configs["pin_memory"]
    num_workers = other_configs["num_workers"]

    if config.is_single_agent_mode():
        print(f"Training on single-agent data from: {data_dir}")
    else:
        print(f"Training on combined data from all achiever-blocker combinations")
    print(f"Results will be saved to: {experiment_save_dir}")

    # Setup model, data, and training components using combined data
    (
        model,
        train_data,
        val_loader,
        optimizer,
        loss_fn,
        scaler,
        early_stopping,
        samples_per_epoch,
        batch_size,
        pin_memory,
        num_workers,
    ) = setup_model_and_data(
        config,
        model_kwargs,
        data_dir,
        agent_type,
        training_proportion,
        device,
        use_parallel,
        device_ids,
        pin_memory,
        num_workers,
        training_process_config,
        batch_size,
        lr,
        training_kwargs.get("weight_decay", 0.001),
        patience,
        min_delta,
        experiment_save_dir,
    )

    # Run training loop
    history = run_training_loop(
        model,
        train_data,
        val_loader,
        optimizer,
        loss_fn,
        device,
        scaler,
        early_stopping,
        epochs,
        max_n_past,
        data_config,
        training_process_config,
        model_config,
        gradient_accumulation_steps,
        experiment_save_dir,
        samples_per_epoch,
        batch_size,
        pin_memory,
        num_workers,
    )

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
        "combined_training": True,  # Flag to indicate this was combined training
        "achiever_types": list(config.achiever_types.keys()),
        "blocker_types": list(config.blocker_types.keys()),
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

    print("Training completed successfully!")
    print(f"Results saved to: {experiment_save_dir}")

    if history["val_loss"]:
        history["best_val_loss"] = min(history["val_loss"])
    else:
        history["best_val_loss"] = float("inf")

    return history




if __name__ == "__main__":
    import argparse
    import os
    import torch.multiprocessing as mp

    # Set multiprocessing start method to prevent file descriptor issues
    mp.set_start_method("spawn", force=True)

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
        default="./results/exp7",
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

    # Run combined training on all achiever-blocker combinations
    print(f"\n{'='*60}")
    print(f"Training on combined data from all achiever-blocker combinations")
    print(f"Achiever types: {list(config.achiever_types.keys())}")
    print(f"Blocker types: {list(config.blocker_types.keys())}")
    print(f"Total combinations: {len(config.achiever_types)} x {len(config.blocker_types)} = {len(config.achiever_types) * len(config.blocker_types)}")
    print(f"{'='*60}")

    # Run combined training
    results = train_tomnet(
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        config=config,
    )

    print(f"\n{'='*60}")
    print(f"Training completed on combined dataset")
    print(f"Best validation loss: {results['best_val_loss']:.4f}")
    print(f"{'='*60}")
