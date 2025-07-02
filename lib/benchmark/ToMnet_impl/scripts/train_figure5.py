#!/usr/bin/env python3
"""
Training script for Figure 5 experiments with goal-directed agents
This script trains ToMnet models on goal-directed agent data for Figure 5 reproduction
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os
import json
from tqdm import tqdm
from typing import Dict, Optional
import platform
import glob
import random
from sklearn.metrics import accuracy_score, mean_squared_error
from scipy.spatial.distance import cosine

from tomnet import ToMnet, create_tomnet
from data_generation import DataGenerator, ToMnetDataset, collate_fn
from evaluate import evaluate_model, compute_kl_divergence
from typing import Dict, Optional, Union


class Figure5ExperimentConfig:
    """Configuration for Figure 5 experiment"""

    def __init__(self):
        self.char_embedding_dim = 8
        self.use_mental_state_net = (
            False  # Figure 5 does NOT use mental state net (per README line 591)
        )
        self.agent_type = "goal_directed"

        # Figure 5 specific parameters
        self.high_cost_ratio = 0.2
        self.n_agents = 100

        # Regularization
        self.dropout_rate = 0.3
        self.patience = 10  # Early stopping patience (reduced for simple task)

        self.loss_weights = {
            "action_loss": 1.0,
            "consumption_loss": 0.5,
            "sr_loss": 0.3,
        }
        self.predictions = ["action", "consumption", "successor_representation"]


class Figure5ToMnetTrainer:
    """Trainer for Figure 5 ToMnet experiments"""

    def __init__(
        self,
        model: ToMnet,
        config: Figure5ExperimentConfig,
        device: str = "cuda",
        learning_rate: float = 1e-3,
    ):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=config.patience, factor=0.5
        )

        # Loss weights from config
        self.loss_weights = config.loss_weights

        # Track metrics for early stopping
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

        # Additional detailed metrics tracking
        self.detailed_metrics = {
            "consumption_mse": {"train": [], "val": []},
            "sr_cosine_similarity": {"train": [], "val": []}
        }

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_losses = {}
        all_predictions = {"action": [], "consumption": [], "sr": []}
        all_targets = {"action": [], "consumption": [], "sr": []}
        n_batches = 0

        for batch in tqdm(dataloader, desc="Training"):
            # Move to device
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Forward pass
            predictions = self.model(
                batch["past_trajectories"],
                batch["current_trajectory"],
                batch["current_state"],
            )

            # Compute losses
            losses = {}

            # Action prediction loss
            action_key = (
                "action_logits" if "action_logits" in predictions else "action_pred"
            )
            action_targets = batch.get("true_actions", batch.get("true_action"))
            action_loss = nn.CrossEntropyLoss()(predictions[action_key], action_targets)
            losses["action_loss"] = action_loss

            # Object consumption prediction loss
            if "consumption" in predictions and "true_consumption" in batch:
                consumption_loss = nn.BCEWithLogitsLoss()(
                    predictions["consumption"], batch["true_consumption"]
                )
                losses["consumption_loss"] = consumption_loss

            # Successor representation loss - using cross-entropy as specified in README line 66
            if "successor_representation" in predictions and "true_sr" in batch:
                # Cross-entropy between predicted and empirical successor representation
                # L_SR = Σ_τ Σ_s -SR_τ(s) log ŜR_τ(s)
                sr_loss = nn.CrossEntropyLoss()(
                    predictions["successor_representation"], batch["true_sr"]
                )
                losses["sr_loss"] = sr_loss

            # Weighted total loss
            total_loss = sum(
                self.loss_weights.get(k, 1.0) * v for k, v in losses.items()
            )

            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # Track losses
            for k, v in losses.items():
                if k not in total_losses:
                    total_losses[k] = 0.0
                total_losses[k] += v.item()

            if "total_loss" not in total_losses:
                total_losses["total_loss"] = 0.0
            total_losses["total_loss"] += total_loss.item()

            # Track predictions for accuracy calculation
            with torch.no_grad():
                all_predictions["action"].extend(
                    torch.argmax(predictions[action_key], dim=1).cpu().numpy()
                )
                action_targets = batch.get("true_actions", batch.get("true_action"))
                all_targets["action"].extend(action_targets.cpu().numpy())

                if "consumption" in predictions:
                    consumption_preds = (
                        torch.sigmoid(predictions["consumption"]).cpu().numpy()
                    )
                    consumption_targets = batch["true_consumption"].cpu().numpy()
                    all_predictions["consumption"].extend(consumption_preds)
                    all_targets["consumption"].extend(consumption_targets)

                if "successor_representation" in predictions:
                    sr_preds = (
                        torch.softmax(predictions["successor_representation"], dim=-1)
                        .cpu()
                        .numpy()
                    )
                    sr_targets = batch["true_sr"].cpu().numpy()
                    all_predictions["sr"].extend(sr_preds)
                    all_targets["sr"].extend(sr_targets)

            n_batches += 1

        # Average losses
        avg_losses = {k: v / n_batches for k, v in total_losses.items()}

        # Calculate accuracies
        action_accuracy = accuracy_score(
            all_targets["action"], all_predictions["action"]
        )
        avg_losses["action_accuracy"] = action_accuracy

        # Calculate detailed consumption metrics
        if all_predictions["consumption"] and all_targets["consumption"]:
            consumption_preds = np.array(all_predictions["consumption"])
            consumption_targets = np.array(all_targets["consumption"])

            avg_losses["consumption_mse"] = mean_squared_error(
                consumption_targets, consumption_preds
            )

        # Calculate detailed SR metrics
        if all_predictions["sr"] and all_targets["sr"]:
            sr_preds = np.array(all_predictions["sr"])
            sr_targets = np.array(all_targets["sr"])

            # Cosine similarity between predicted and true SR
            cosine_similarities = []
            for i in range(len(sr_preds)):
                pred_flat = sr_preds[i].flatten()
                target_flat = sr_targets[i].flatten()

                # Cosine similarity (1 - cosine distance)
                if np.linalg.norm(pred_flat) > 0 and np.linalg.norm(target_flat) > 0:
                    cosine_sim = 1 - cosine(pred_flat, target_flat)
                    cosine_similarities.append(cosine_sim)

            avg_losses["sr_cosine_similarity"] = (
                np.mean(cosine_similarities) if cosine_similarities else 0.0
            )

        return avg_losses

    def validate_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        total_losses = {}
        all_predictions = {"action": [], "consumption": [], "sr": []}
        all_targets = {"action": [], "consumption": [], "sr": []}
        n_batches = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating"):
                # Move to device
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                # Forward pass
                predictions = self.model(
                    batch["past_trajectories"],
                    batch["current_trajectory"],
                    batch["current_state"],
                )

                # Compute losses
                losses = {}

                # Action prediction loss
                action_key = (
                    "action_logits" if "action_logits" in predictions else "action_pred"
                )
                action_targets = batch.get("true_actions", batch.get("true_action"))
                action_loss = nn.CrossEntropyLoss()(
                    predictions[action_key], action_targets
                )
                losses["action_loss"] = action_loss

                # Object consumption prediction loss
                if "consumption" in predictions and "true_consumption" in batch:
                    consumption_loss = nn.BCEWithLogitsLoss()(
                        predictions["consumption"], batch["true_consumption"]
                    )
                    losses["consumption_loss"] = consumption_loss

                # Successor representation loss - using cross-entropy as specified in README line 66
                if "successor_representation" in predictions and "true_sr" in batch:
                    # Cross-entropy between predicted and empirical successor representation
                    # L_SR = Σ_τ Σ_s -SR_τ(s) log ŜR_τ(s)
                    sr_loss = nn.CrossEntropyLoss()(
                        predictions["successor_representation"], batch["true_sr"]
                    )
                    losses["sr_loss"] = sr_loss

                # Weighted total loss
                total_loss = sum(
                    self.loss_weights.get(k, 1.0) * v for k, v in losses.items()
                )

                # Track losses
                for k, v in losses.items():
                    if k not in total_losses:
                        total_losses[k] = 0.0
                    total_losses[k] += v.item()

                if "total_loss" not in total_losses:
                    total_losses["total_loss"] = 0.0
                total_losses["total_loss"] += total_loss.item()

                # Track predictions for accuracy calculation
                all_predictions["action"].extend(
                    torch.argmax(predictions[action_key], dim=1).cpu().numpy()
                )
                action_targets = batch.get("true_actions", batch.get("true_action"))
                all_targets["action"].extend(action_targets.cpu().numpy())

                if "consumption" in predictions:
                    consumption_preds = (
                        torch.sigmoid(predictions["consumption"]).cpu().numpy()
                    )
                    consumption_targets = batch["true_consumption"].cpu().numpy()
                    all_predictions["consumption"].extend(consumption_preds)
                    all_targets["consumption"].extend(consumption_targets)

                if "successor_representation" in predictions:
                    sr_preds = (
                        torch.softmax(predictions["successor_representation"], dim=-1)
                        .cpu()
                        .numpy()
                    )
                    sr_targets = batch["true_sr"].cpu().numpy()
                    all_predictions["sr"].extend(sr_preds)
                    all_targets["sr"].extend(sr_targets)

                n_batches += 1

        # Average losses
        avg_losses = {k: v / n_batches for k, v in total_losses.items()}

        # Calculate accuracies
        action_accuracy = accuracy_score(
            all_targets["action"], all_predictions["action"]
        )
        avg_losses["action_accuracy"] = action_accuracy

        # Calculate detailed consumption metrics
        if all_predictions["consumption"] and all_targets["consumption"]:
            consumption_preds = np.array(all_predictions["consumption"])
            consumption_targets = np.array(all_targets["consumption"])

            avg_losses["consumption_mse"] = mean_squared_error(
                consumption_targets, consumption_preds
            )

        # Calculate detailed SR metrics
        if all_predictions["sr"] and all_targets["sr"]:
            sr_preds = np.array(all_predictions["sr"])
            sr_targets = np.array(all_targets["sr"])

            # Cosine similarity between predicted and true SR
            cosine_similarities = []
            for i in range(len(sr_preds)):
                pred_flat = sr_preds[i].flatten()
                target_flat = sr_targets[i].flatten()

                # Cosine similarity (1 - cosine distance)
                if np.linalg.norm(pred_flat) > 0 and np.linalg.norm(target_flat) > 0:
                    cosine_sim = 1 - cosine(pred_flat, target_flat)
                    cosine_similarities.append(cosine_sim)

            avg_losses["sr_cosine_similarity"] = (
                np.mean(cosine_similarities) if cosine_similarities else 0.0
            )

        return avg_losses

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        n_epochs: int,
        save_path: str,
    ) -> Dict[str, list]:
        """Full training loop"""
        print(f"Training for {n_epochs} epochs...")
        print(f"Model will be saved to: {save_path}")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        for epoch in range(n_epochs):
            print(f"\nEpoch {epoch + 1}/{n_epochs}")

            # Training
            train_metrics = self.train_epoch(train_dataloader)
            self.train_losses.append(train_metrics["total_loss"])
            self.train_accuracies.append(train_metrics["action_accuracy"])

            # Validation
            val_metrics = self.validate_epoch(val_dataloader)
            self.val_losses.append(val_metrics["total_loss"])
            self.val_accuracies.append(val_metrics["action_accuracy"])

            # Store detailed metrics for both training and validation
            self._store_detailed_metrics(train_metrics, val_metrics)

            # Scheduler step
            self.scheduler.step(val_metrics["total_loss"])

            # Print metrics
            print(
                f"Train Loss: {train_metrics['total_loss']:.4f}, Train Acc: {train_metrics['action_accuracy']:.4f}"
            )
            print(
                f"Val Loss: {val_metrics['total_loss']:.4f}, Val Acc: {val_metrics['action_accuracy']:.4f}"
            )

            # Print detailed metrics if available
            if "consumption_mse" in train_metrics:
                print(
                    f"Consumption MSE - Train: {train_metrics['consumption_mse']:.4f}, Val: {val_metrics['consumption_mse']:.4f}"
                )

            if "sr_cosine_similarity" in train_metrics:
                print(
                    f"SR Cosine Similarity - Train: {train_metrics['sr_cosine_similarity']:.4f}, Val: {val_metrics['sr_cosine_similarity']:.4f}"
                )

            # Early stopping
            if val_metrics["total_loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["total_loss"]
                self.patience_counter = 0

                # Save best model
                config_dict = self.config.__dict__.copy()
                if hasattr(self, "state_dim"):
                    config_dict["state_dim"] = self.state_dim

                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "epoch": epoch,
                        "val_loss": val_metrics["total_loss"],
                        "val_accuracy": val_metrics["action_accuracy"],
                        "config": config_dict,
                    },
                    save_path,
                )
                print(f"Saved best model to {save_path}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.patience:
                    print(f"Early stopping after {epoch + 1} epochs")
                    break

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_accuracies": self.train_accuracies,
            "val_accuracies": self.val_accuracies,
            "detailed_metrics": self.detailed_metrics,
        }

    def _store_detailed_metrics(
        self, train_metrics: Dict[str, float], val_metrics: Dict[str, float]
    ):
        """Store detailed metrics for consumption and SR predictions"""

        # Consumption metrics
        for metric_key in ["consumption_mse"]:
            if metric_key not in self.detailed_metrics:
                self.detailed_metrics[metric_key] = {"train": [], "val": []}

            self.detailed_metrics[metric_key]["train"].append(
                train_metrics.get(metric_key, 0.0)
            )
            self.detailed_metrics[metric_key]["val"].append(
                val_metrics.get(metric_key, 0.0)
            )

        # SR metrics
        for metric_key in ["sr_cosine_similarity"]:
            if metric_key not in self.detailed_metrics:
                self.detailed_metrics[metric_key] = {"train": [], "val": []}

            self.detailed_metrics[metric_key]["train"].append(
                train_metrics.get(metric_key, 0.0)
            )
            self.detailed_metrics[metric_key]["val"].append(
                val_metrics.get(metric_key, 0.0)
            )


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train ToMnet for Figure 5")

    # Data parameters
    parser.add_argument("--n_agents", type=int, default=100, help="Number of agents")
    parser.add_argument(
        "--n_episodes_per_agent", type=int, default=100, help="Episodes per agent"
    )
    parser.add_argument(
        "--alpha_reward",
        type=float,
        default=0.01,
        help="Dirichlet concentration for rewards",
    )
    parser.add_argument(
        "--high_cost_ratio", type=float, default=0.2, help="Ratio of high-cost agents"
    )

    # Training parameters
    parser.add_argument(
        "--n_epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Learning rate"
    )
    parser.add_argument("--device", type=str, default="cuda:3", help="Device to use")
    parser.add_argument(
        "--val_split", type=float, default=0.2, help="Validation split ratio"
    )

    # Output parameters
    parser.add_argument(
        "--data_dir", type=str, default="data/figure5", help="Data directory"
    )
    parser.add_argument(
        "--model_dir", type=str, default="models/figure5", help="Model save directory"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="figure5_goal_directed",
        help="Experiment name",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Figure 5 ToMnet Training")
    print("=" * 60)
    print(f"Agent type: Goal-directed")
    print(f"Agents: {args.n_agents}")
    print(f"Episodes per agent: {args.n_episodes_per_agent}")
    print(f"Alpha reward: {args.alpha_reward}")
    print(f"High cost ratio: {args.high_cost_ratio}")
    print(f"Epochs: {args.n_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")
    print("=" * 60)

    # Set device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    # Generate or load data
    data_path = os.path.join(args.data_dir, "goal_directed_training_data.pkl")

    if not os.path.exists(data_path):
        print("Generating training data...")
        generator = DataGenerator()
        dataset = generator.generate_goal_directed_agent_data(
            n_agents=args.n_agents,
            n_episodes_per_agent=args.n_episodes_per_agent,
            alpha_reward=args.alpha_reward,
            high_cost_ratio=args.high_cost_ratio,
            min_past=0,
            max_past=10,
            save_path=data_path,
        )
        print(f"Generated {len(dataset['data'])} training samples")
    else:
        print(f"Loading existing data from {data_path}")
        import pickle

        with open(data_path, "rb") as f:
            dataset = pickle.load(f)
        print(f"Loaded {len(dataset['data'])} training samples")

    # Create dataset and split
    tomnet_dataset = ToMnetDataset(dataset, experiment_type="figure5")

    # Train/validation split
    n_samples = len(tomnet_dataset)
    n_val = int(n_samples * args.val_split)
    n_train = n_samples - n_val

    train_dataset, val_dataset = torch.utils.data.random_split(
        tomnet_dataset, [n_train, n_val]
    )

    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    print(f"Training samples: {n_train}")
    print(f"Validation samples: {n_val}")

    # Create model
    config = Figure5ExperimentConfig()

    model = create_tomnet(
        experiment_type="figure5",
        state_dim=dataset["meta"]["state_dim"],
        char_embedding_dim=config.char_embedding_dim,
        dropout_rate=config.dropout_rate,
    )

    print(
        f"Created ToMnet model with {sum(p.numel() for p in model.parameters())} parameters"
    )

    # Create trainer
    trainer = Figure5ToMnetTrainer(
        model=model,
        config=config,
        device=args.device,
        learning_rate=args.learning_rate,
    )

    # Store state_dim for saving in checkpoint
    trainer.state_dim = dataset["meta"]["state_dim"]

    # Train model
    save_path = os.path.join(
        args.model_dir, f"{args.experiment_name}_alpha{args.alpha_reward}_model.pth"
    )
    os.makedirs(args.model_dir, exist_ok=True)

    training_history = trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        n_epochs=args.n_epochs,
        save_path=save_path,
    )

    # Save training history
    history_path = os.path.join(
        args.model_dir, f"{args.experiment_name}_alpha{args.alpha_reward}_history.json"
    )
    with open(history_path, "w") as f:
        json.dump(training_history, f, indent=2)

    print(f"\nTraining completed!")
    print(f"Best model saved to: {save_path}")
    print(f"Training history saved to: {history_path}")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")


if __name__ == "__main__":
    main()
