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
from sklearn.metrics import accuracy_score

from tomnet import ToMnet, create_tomnet
from data_generation import DataGenerator, ToMnetDataset, collate_fn
from evaluate import evaluate_model, compute_kl_divergence
from typing import Dict, Optional, Union


class ExperimentConfig:
    """Configuration for different experiment types"""

    def __init__(self, experiment_type: str):
        self.experiment_type = experiment_type

        if experiment_type == "figure3":
            self.char_embedding_dim = 10
            self.use_mental_state_net = False
            self.agent_type = "random"
            self.alpha_values = [
                0.01,
                0.03,
                0.1,
                0.3,
                1.0,
                3.0,
            ]  # Alpha values for Figure 3 cross-species analysis
            self.n_agents = 1000

            # Regularization
            self.dropout_rate = 0.3
            self.patience = 30  # Early stopping patience

            self.loss_weights = {"action_loss": 1.0}
            self.predictions = ["action"]
        else:
            raise ValueError(f"Unknown experiment type: {experiment_type}")


class ToMnetTrainer:
    """Unified trainer for ToMnet experiments"""

    def __init__(
        self,
        model: ToMnet,
        config: ExperimentConfig,
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

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_losses = {}
        all_predictions = []
        all_targets = []
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
            targets = {"true_actions": batch["true_actions"]}
            losses = self.model.compute_loss(predictions, targets)

            # Weighted total loss
            weighted_loss = 0
            for loss_name, loss_value in losses.items():
                if loss_name != "total_loss":
                    weight = self.loss_weights.get(loss_name, 1.0)
                    weighted_loss += weight * loss_value

            # Backward pass
            self.optimizer.zero_grad()
            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Accumulate losses
            for loss_name, loss_value in losses.items():
                if loss_name not in total_losses:
                    total_losses[loss_name] = 0
                total_losses[loss_name] += loss_value.item()

            total_losses["weighted_loss"] = (
                total_losses.get("weighted_loss", 0) + weighted_loss.item()
            )

            # Track accuracy
            if "action_pred" in predictions:
                pred_actions = torch.argmax(predictions["action_pred"], dim=1)
                all_predictions.extend(pred_actions.cpu().numpy())
                all_targets.extend(batch["true_actions"].cpu().numpy())

            n_batches += 1

        # Average losses
        avg_losses = {k: v / n_batches for k, v in total_losses.items()}

        # Calculate accuracy
        if all_predictions:
            accuracy = accuracy_score(all_targets, all_predictions)
            avg_losses["accuracy"] = accuracy
            self.train_accuracies.append(accuracy)

        self.train_losses.append(
            avg_losses.get("total_loss", avg_losses.get("weighted_loss", 0))
        )

        return avg_losses

    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        total_losses = {}
        all_predictions = []
        all_targets = []
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
                targets = {"true_actions": batch["true_actions"]}
                losses = self.model.compute_loss(predictions, targets)

                # Accumulate losses
                for loss_name, loss_value in losses.items():
                    if loss_name not in total_losses:
                        total_losses[loss_name] = 0
                    total_losses[loss_name] += loss_value.item()

                # Track accuracy
                if "action_pred" in predictions:
                    pred_actions = torch.argmax(predictions["action_pred"], dim=1)
                    all_predictions.extend(pred_actions.cpu().numpy())
                    all_targets.extend(batch["true_actions"].cpu().numpy())

                n_batches += 1

        # Average losses
        avg_losses = {k: v / n_batches for k, v in total_losses.items()}

        # Calculate accuracy
        if all_predictions:
            accuracy = accuracy_score(all_targets, all_predictions)
            avg_losses["accuracy"] = accuracy
            self.val_accuracies.append(accuracy)

        val_loss = avg_losses.get("total_loss", avg_losses.get("action_loss", 0))
        self.val_losses.append(val_loss)

        # Early stopping logic
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        return avg_losses

    def should_stop_early(self) -> bool:
        """Check if training should stop early"""
        return self.patience_counter >= self.config.patience

    def save_checkpoint(self, epoch: int, val_loss: float, save_path: str):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_loss": val_loss,
            "best_val_loss": self.best_val_loss,
            "loss_weights": self.loss_weights,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_accuracies": self.train_accuracies,
            "val_accuracies": self.val_accuracies,
        }
        torch.save(checkpoint, save_path)
        print(f"Saved checkpoint to {save_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Load training history
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])
        self.train_accuracies = checkpoint.get("train_accuracies", [])
        self.val_accuracies = checkpoint.get("val_accuracies", [])

        return checkpoint["epoch"], checkpoint["val_loss"]


def create_model(
    experiment_type: str, state_dim: int, config: ExperimentConfig
) -> ToMnet:
    """Create ToMnet model based on experiment type"""
    return create_tomnet(
        experiment_type=experiment_type,
        state_dim=state_dim,
        char_embedding_dim=config.char_embedding_dim,
        dropout_rate=config.dropout_rate,
    )


def generate_data(
    experiment_type: str, config: ExperimentConfig, args
) -> Union[Dict, Dict[str, Dict]]:
    """Generate training data based on experiment type"""
    # Check if data files already exist
    data_files = glob.glob("data/figure*.pkl")
    if data_files:
        print(f"Found existing data files: {data_files}")
        print("Skipping data generation...")

        # Load existing data files
        if experiment_type == "figure3":
            datasets = {}
            alpha_values = getattr(args.n_agents, "alpha_values", config.alpha_values)

            for alpha in alpha_values:
                file_path = f"data/{experiment_type}/alpha_{alpha}.pkl"
                if os.path.exists(file_path):
                    print(f"Loading existing data for alpha={alpha}")
                    import pickle

                    with open(file_path, "rb") as f:
                        datasets[alpha] = pickle.load(f)
                else:
                    print(
                        f"Warning: Expected file {file_path} not found, generating new data"
                    )
                    # Generate missing data
                    data_generator = DataGenerator()
                    dataset = data_generator.generate_random_agent_data(
                        n_agents=args.n_agents or config.n_agents,
                        n_episodes_per_agent=args.n_episodes_per_agent,
                        alpha=alpha,
                        save_path=file_path,
                        n_workers=args.n_workers,
                    )
                    datasets[alpha] = dataset

            # Create mixed dataset if requested
            if getattr(args, "mixed_training", False):
                print("Creating mixed dataset")
                mixed_data = []
                for alpha, dataset in datasets.items():
                    mixed_data.extend(dataset["data"])

                mixed_dataset = {
                    "data": mixed_data,
                    "meta": {
                        "mixed": True,
                        "alpha_values": list(datasets.keys()),
                        "state_dim": datasets[list(datasets.keys())[0]]["meta"][
                            "state_dim"
                        ],
                    },
                }
                datasets["mixed"] = mixed_dataset

            return datasets

    # Generate new data if not found
    data_generator = DataGenerator()

    if experiment_type == "figure3":
        # Generate datasets for different alpha values
        datasets = {}
        alpha_values = getattr(args, "alpha_values", config.alpha_values)

        for alpha in alpha_values:
            print(f"Generating data for alpha={alpha}")
            dataset = data_generator.generate_random_agent_data(
                n_agents=args.n_agents or config.n_agents,
                n_episodes_per_agent=args.n_episodes_per_agent,
                alpha=alpha,
                save_path=f"data/{experiment_type}/alpha_{alpha}.pkl",
                n_workers=args.n_workers,
            )
            datasets[alpha] = dataset

        # Create mixed dataset if requested
        if getattr(args, "mixed_training", False):
            print("Creating mixed dataset")
            mixed_data = []
            for alpha, dataset in datasets.items():
                mixed_data.extend(dataset["data"])

            mixed_dataset = {
                "data": mixed_data,
                "meta": {
                    "mixed": True,
                    "alpha_values": list(datasets.keys()),
                    "state_dim": datasets[list(datasets.keys())[0]]["meta"][
                        "state_dim"
                    ],
                },
            }
            datasets["mixed"] = mixed_dataset

        return datasets

    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")


def generate_cross_species_evaluation_files(
    results, datasets, experiment_type, device="cpu"
):
    """Generate JSON files for cross-species evaluation"""
    if experiment_type != "figure3":
        print(f"Cross-species evaluation files not applicable for {experiment_type}")
        return

    print("\n=== Generating Cross-Species Evaluation Files ===")

    # Create evaluation directory
    eval_dir = f"result/{experiment_type}"
    os.makedirs(eval_dir, exist_ok=True)

    # Generate model_paths.json
    model_paths = {}
    data_paths = {}

    for dataset_name, result in results.items():
        if dataset_name == "mixed":
            model_paths["mixed"] = os.path.abspath(result["model_path"])
        else:
            # Convert dataset name to alpha format
            if isinstance(dataset_name, (int, float)):
                alpha_key = f"alpha_{dataset_name}"
                model_paths[alpha_key] = os.path.abspath(result["model_path"])
            else:
                model_paths[dataset_name] = os.path.abspath(result["model_path"])

    # Generate data_paths.json from datasets
    for dataset_name, dataset in datasets.items():
        if dataset_name == "mixed":
            continue  # Skip mixed dataset for testing

        if isinstance(dataset_name, (int, float)):
            alpha_key = f"alpha_{dataset_name}"
            # Use the original data file path
            data_file_path = f"data/{experiment_type}/alpha_{dataset_name}.pkl"
            data_paths[alpha_key] = os.path.abspath(data_file_path)
        else:
            data_file_path = f"data/{experiment_type}/{dataset_name}.pkl"
            data_paths[dataset_name] = os.path.abspath(data_file_path)

    # Save model_paths.json
    model_paths_file = os.path.join(eval_dir, "model_paths.json")
    with open(model_paths_file, "w") as f:
        json.dump(model_paths, f, indent=2)
    print(f"✓ Saved model paths to: {model_paths_file}")
    print(f"  Models: {list(model_paths.keys())}")

    # Save data_paths.json
    data_paths_file = os.path.join(eval_dir, "data_paths.json")
    with open(data_paths_file, "w") as f:
        json.dump(data_paths, f, indent=2)
    print(f"✓ Saved data paths to: {data_paths_file}")
    print(f"  Datasets: {list(data_paths.keys())}")

    # Generate evaluation script
    eval_script_content = f"""#!/bin/bash
# Auto-generated cross-species evaluation script

echo "Running Figure 3 Cross-Species Evaluation..."

# Run cross-species evaluation
python scripts/evaluate.py \\
    --experiment figure3 \\
    --model_paths_json {model_paths_file} \\
    --data_paths_json {data_paths_file} \\
    --output_path {eval_dir}/figure3_cross_species_results.pkl \\
    --device {device}

echo "Evaluation completed!"
echo "Results saved to: {eval_dir}/figure3_cross_species_results.pkl"
echo ""
echo "To visualize results, run:"
echo "python scripts/visualize_figure3.py --results_path {eval_dir}/figure3_cross_species_results.pkl --save_plots"
"""

    eval_script_file = os.path.join(eval_dir, "run_cross_species_evaluation.sh")
    with open(eval_script_file, "w") as f:
        f.write(eval_script_content)

    # Make script executable
    os.chmod(eval_script_file, 0o755)
    print(f"✓ Saved evaluation script to: {eval_script_file}")

    # Generate summary info
    summary = {
        "experiment_type": experiment_type,
        "models_trained": list(model_paths.keys()),
        "test_datasets": list(data_paths.keys()),
        "model_paths_file": model_paths_file,
        "data_paths_file": data_paths_file,
        "evaluation_script": eval_script_file,
        "next_steps": [
            f"Run evaluation: bash {eval_script_file}",
            "View results: jupyter notebook visualize_figure3.ipynb",
            "Results will be saved to: result/figure3_cross_species_results.pkl",
        ],
    }

    summary_file = os.path.join(eval_dir, "evaluation_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved evaluation summary to: {summary_file}")

    print(f"\n=== Cross-Species Evaluation Setup Complete ===")
    print(f"Next steps:")
    print(f"1. Run evaluation: bash {eval_script_file}")
    print(
        f"2. View results: python scripts/visualize_figure3.py --results_path {eval_dir}/figure3_cross_species_results.pkl --save_plots"
    )
    print("")


def train_unified_model(experiment_type: str, args):
    """Unified training function for Figure 3 experiments"""
    print(f"Training ToMnet for {experiment_type.title()} Experiment")

    # Create experiment configuration
    config = ExperimentConfig(experiment_type)

    # Generate data
    data = generate_data(experiment_type, config, args)

    if experiment_type == "figure3":
        # Train separate models for each dataset
        results = {}
        datasets = data  # data is a dict of datasets for figure3

        for dataset_name, dataset in datasets.items():
            print(f"\nTraining model for {dataset_name}")
            print(f"Dataset size: {len(dataset['data'])} samples")

            # Create model
            state_dim = dataset["meta"]["state_dim"]
            model = create_model(experiment_type, state_dim, config)

            # Split data into train/validation
            data_samples = dataset["data"]
            n_samples = len(data_samples)
            n_train = int(0.8 * n_samples)

            # Shuffle data
            shuffled_indices = list(range(n_samples))
            random.shuffle(shuffled_indices)

            train_data = [data_samples[i] for i in shuffled_indices[:n_train]]
            val_data = [data_samples[i] for i in shuffled_indices[n_train:]]

            train_dataset_dict = {"data": train_data, "meta": dataset["meta"]}
            val_dataset_dict = {"data": val_data, "meta": dataset["meta"]}

            # Create datasets and dataloaders
            train_dataset = ToMnetDataset(
                train_dataset_dict, experiment_type=experiment_type
            )
            val_dataset = ToMnetDataset(
                val_dataset_dict, experiment_type=experiment_type
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=min(4, os.cpu_count()),  # Parallel loading
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=min(4, os.cpu_count()),
            )

            # Create trainer
            trainer = ToMnetTrainer(model, config, args.device, args.learning_rate)

            # Training loop
            print(f"Training for up to {args.n_epochs} epochs with early stopping...")
            best_val_loss = float("inf")

            for epoch in range(args.n_epochs):
                # Train
                train_losses = trainer.train_epoch(train_loader)

                # Validate
                val_losses = trainer.validate(val_loader)

                # Update scheduler
                trainer.scheduler.step(
                    val_losses.get("total_loss", val_losses.get("action_loss", 0))
                )

                # Log progress
                print(f"Epoch {epoch+1}/{args.n_epochs}")
                print(
                    f"  Train Loss: {train_losses.get('total_loss', train_losses.get('weighted_loss', 0)):.4f}"
                )
                print(
                    f"  Val Loss: {val_losses.get('total_loss', val_losses.get('action_loss', 0)):.4f}"
                )
                if "accuracy" in train_losses:
                    print(f"  Train Acc: {train_losses['accuracy']:.3f}")
                if "accuracy" in val_losses:
                    print(f"  Val Acc: {val_losses['accuracy']:.3f}")
                print(f"  LR: {trainer.optimizer.param_groups[0]['lr']:.6f}")

                # Save best model
                current_val_loss = val_losses.get(
                    "total_loss", val_losses.get("action_loss", 0)
                )
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    os.makedirs(f"models/{experiment_type}", exist_ok=True)
                    save_path = f"models/{experiment_type}/{dataset_name}_best.pth"
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    trainer.save_checkpoint(epoch, best_val_loss, save_path)
                    print(f"  ✓ New best model saved!")

                # Check early stopping
                if trainer.should_stop_early():
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

            results[dataset_name] = {
                "best_val_loss": best_val_loss,
                "model_path": save_path,
                "final_train_acc": (
                    trainer.train_accuracies[-1] if trainer.train_accuracies else 0
                ),
                "final_val_acc": (
                    trainer.val_accuracies[-1] if trainer.val_accuracies else 0
                ),
            }

            print(f"✓ Completed training for {dataset_name}")
            print(f"  Best validation loss: {best_val_loss:.4f}")
            if trainer.val_accuracies:
                print(f"  Final validation accuracy: {trainer.val_accuracies[-1]:.3f}")

        # Generate cross-species evaluation JSON files
        generate_cross_species_evaluation_files(
            results, datasets, experiment_type, args.device
        )

        return results


def main():
    parser = argparse.ArgumentParser(description="Train ToMnet")
    parser.add_argument(
        "--experiment",
        choices=["figure3"],
        default="figure3",
        help="Which experiment to run",
    )
    # Detect device based on platform
    if platform.system() == "Darwin":  # macOS
        default_device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        default_device = "cuda:3" if torch.cuda.is_available() else "cpu"

    parser.add_argument("--device", default=default_device, help="Device to use")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Learning rate"
    )
    parser.add_argument("--n_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--n_agents", type=int, default=100, help="Number of agents")
    parser.add_argument(
        "--n_episodes_per_agent", type=int, default=100, help="Episodes per agent"
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=None,
        help="Number of parallel workers for data generation",
    )

    # Figure 3 specific
    parser.add_argument(
        "--alpha_values",
        nargs="+",
        type=float,
        default=[0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
        help="Alpha values for Figure 3 cross-species evaluation",
    )
    parser.add_argument(
        "--mixed_training", action="store_true", help="Train on mixed alpha dataset"
    )

    args = parser.parse_args()

    # Create directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    results = {}

    if args.experiment in ["figure3"]:
        results["figure3"] = train_unified_model("figure3", args)

    # Save results
    # Ensure result directory exists
    os.makedirs(f"result/{args.experiment}", exist_ok=True)
    with open(f"result/{args.experiment}/training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED!")
    print("=" * 60)
    print(f"Results saved to result/{args.experiment}/training_results.json")

    # Print summary
    if args.experiment == "figure3" and "figure3" in results:
        print("\nTrained Models Summary:")
        for model_name, result in results["figure3"].items():
            print(f"  • {model_name}:")
            print(f"    - Best val loss: {result['best_val_loss']:.4f}")
            if "final_val_acc" in result:
                print(f"    - Final val accuracy: {result['final_val_acc']:.3f}")
            print(f"    - Model path: {result['model_path']}")

    # Print next steps for Figure 3
    if args.experiment in ["figure3"]:
        print("\n=== Figure 3 Cross-Species Evaluation Ready ===")
        if os.path.exists("result/figure3/evaluation_summary.json"):
            with open("result/figure3/evaluation_summary.json", "r") as f:
                summary = json.load(f)

            print("✓ Cross-species evaluation files generated:")
            print(f"  - Models trained: {summary['models_trained']}")
            print(f"  - Test datasets: {summary['test_datasets']}")
            print("\nTo run cross-species evaluation:")
            print("  bash result/figure3/run_cross_species_evaluation.sh")
            print("\nTo visualize results:")
            print(
                "  python scripts/visualize_figure3.py --results_path result/figure3/figure3_cross_species_results.pkl --save_plots"
            )
        else:
            print("⚠ Cross-species evaluation files not found")
            print("Make sure Figure 3 training completed successfully")


if __name__ == "__main__":
    main()
