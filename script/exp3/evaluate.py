import sys
import json
import os
import pickle

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.dirname(__file__))
from tomnet import ToMnet, create_model
from config import Config
from train import prepare_data_for_training, generate_past_episodes_from_batch
from data_generation import DataReader

"""
Evaluation and metrics for KeyDoor ToMnet experiment
Adapted from ToMnetF experiment5 for KeyDoor environment
"""


def load_model(model_path, device, model_kwargs):
    """Load trained ToMnet model"""
    model = create_model(model_kwargs)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def evaluate_model_with_n_past(
    model,
    test_loader,
    device,
    n_past_values,
    n_past_max,
    data_config=None,
):
    """
    Evaluate model performance with different N_past values

    Args:
        model: Trained ToMnet model
        test_loader: Test data loader (must include goals as 4th element)
        device: Computing device
        n_past_values: List of N_past values to test
        n_past_max: Maximum number of past episodes
        data_config: Data configuration for processing

    Returns:
        dict: Evaluation metrics by N_past
    """
    model.eval()
    results_by_n_past = {}

    for n_past in n_past_values:
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for _, batch in enumerate(test_loader):
                if len(batch) >= 3:
                    trajectories, actions, goals = batch[:3]
                    trajectories = trajectories.to(device)
                    actions = actions.to(device)
                    goals = goals.to(device)

                    batch_size = trajectories.size(0)

                    # Generate past episodes with fixed n_past using actual goals
                    past_episodes = generate_past_episodes_from_batch(
                        trajectories, goals, batch_size, n_past, n_past, n_past_max
                    )

                    # Get current trajectory for MentalNet processing
                    current_timestep = data_config["time_step"] if data_config else 20
                    recent_trajectory = trajectories[:, :current_timestep]

                    # Extract current state for PredNet
                    current_state = trajectories[
                        :, current_timestep - 1
                    ]  # [batch, channels, height, width]

                    # Get action targets - use actions[:, 0] for trajectory slicing
                    action_targets = actions[
                        :, 0
                    ]  # Target action for each sliced trajectory

                    # Model forward pass (model returns 6 outputs)
                    action_logits, _, _, _, _, _ = model(
                        past_episodes, recent_trajectory, current_state
                    )

                    # Get predictions
                    _, predicted = torch.max(action_logits, 1)

                    all_predictions.extend(predicted.cpu().numpy())
                    all_targets.extend(action_targets.cpu().numpy())

        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average="weighted"
        )

        results_by_n_past[n_past] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "predictions": all_predictions,
            "targets": all_targets,
        }

        print(f"N_past={n_past}: Accuracy={accuracy:.4f}, F1={f1:.4f}")

    return results_by_n_past


def evaluate_model(
    model,
    test_loader,
    device,
    data_config=None,
    save_predictions=False,
    output_dir=None,
):
    """
    Evaluate model performance

    Args:
        model: Trained ToMnet model
        test_loader: Test data loader
        device: Computing device
        data_config: Data configuration for processing
        save_predictions: Whether to save predictions
        output_dir: Directory to save predictions

    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    all_predictions = []
    all_targets = []
    all_probabilities = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if len(batch) >= 3:
                trajectories, actions, goals = batch[:3]
                trajectories = trajectories.to(device)
                actions = actions.to(device)
                goals = goals.to(device)

                batch_size = trajectories.size(0)

                # Generate past episodes for proper evaluation
                past_episodes = generate_past_episodes_from_batch(
                    trajectories,
                    goals,
                    batch_size,
                    n_past_min=data_config.get("n_past_min", 1) if data_config else 1,
                    n_past_max=data_config.get("n_past_max", 1) if data_config else 1,
                    max_n_past=data_config.get("max_n_past", 1) if data_config else 1,
                )

                # With trajectory slicing, we use dynamic timesteps
                # Each sample has a different effective length, stored in actions[:,0]

                # For trajectory slicing, use the action at index 0 (the target action for this slice)
                action_targets = actions[
                    :, 0
                ]  # Target action for each sliced trajectory

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
                    non_zero_mask,
                    seq_indices,
                    torch.tensor(-1, device=trajectories.device),
                )
                # Find the maximum index for each batch (last non-zero timestep)
                effective_lengths = masked_indices.max(dim=1)[0].clamp(min=0).tolist()
                # Apply max(1, length) constraint
                effective_lengths = [max(1, length) for length in effective_lengths]

                # Use full trajectory for MentalNet (up to effective length)
                recent_trajectory = (
                    trajectories  # [batch_size, seq_len, channels, height, width]
                )

                # Extract current state for PredNet (last non-padded timestep)
                current_state_channels = 8  # Assuming 8 channels for current state
                current_state = torch.zeros(
                    batch_size,
                    current_state_channels,
                    trajectories.size(3),
                    trajectories.size(4),
                    device=device,
                )

                # Vectorized: Extract current state using advanced indexing on the same device
                batch_indices = torch.arange(batch_size, device=trajectories.device)
                last_timesteps = torch.tensor(
                    [max(0, length - 1) for length in effective_lengths],
                    device=trajectories.device,
                )

                # Extract current state using advanced indexing
                current_state = trajectories[
                    batch_indices, last_timesteps, :current_state_channels
                ]

                # Model forward pass (model returns 6 outputs)
                (
                    action_logits,
                    goal_logits,
                    consumption_logits,
                    sr_pred,
                    char_emb,
                    mental_state,
                ) = model(past_episodes, recent_trajectory, current_state)

                # Get predictions
                probabilities = F.softmax(action_logits, dim=1)
                _, predicted = torch.max(action_logits, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(action_targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

    # Convert to numpy arrays
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    probabilities = np.array(all_probabilities)

    # Calculate metrics
    accuracy = accuracy_score(targets, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, predictions, average="weighted"
    )
    conf_matrix = confusion_matrix(targets, predictions)

    # Action-wise accuracy - KeyDoor has 7 actions
    action_accuracy = {}
    for action in range(7):  # 7 actions in KeyDoor
        mask = targets == action
        if np.sum(mask) > 0:
            action_acc = accuracy_score(targets[mask], predictions[mask])
            action_accuracy[f"action_{action}"] = float(action_acc)

    # Confidence statistics
    confidence_stats = {
        "mean_confidence": float(np.mean(np.max(probabilities, axis=1))),
        "std_confidence": float(np.std(np.max(probabilities, axis=1))),
        "min_confidence": float(np.min(np.max(probabilities, axis=1))),
        "max_confidence": float(np.max(np.max(probabilities, axis=1))),
    }

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": conf_matrix.tolist(),
        "action_accuracy": action_accuracy,
        "confidence_stats": confidence_stats,
        "n_samples": len(targets),
    }

    # Save predictions if requested
    if save_predictions and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        predictions_data = {
            "predictions": predictions.tolist(),
            "targets": targets.tolist(),
            "probabilities": probabilities.tolist(),
            "metrics": metrics,
        }
        pred_path = os.path.join(output_dir, "predictions.pkl")
        with open(pred_path, "wb") as f:
            pickle.dump(predictions_data, f)
        print(f"Predictions saved to: {pred_path}")

    return metrics


def evaluate_keydoor_model(
    config=None,
    model_path=None,
    test_data_dir=None,
    results_dir=None,
):
    """
    Perform evaluation on KeyDoor ToMnet model

    Args:
        config: Config object containing all evaluation parameters
        model_path: Path to trained model
        test_data_dir: Directory containing test data
        results_dir: Directory to save results
    """
    if config is None:
        config = Config()

    # Use provided paths or default from config
    if model_path is None:
        # Find the best model in results directory
        model_path = os.path.join(config.model_dir, "best_model.pth")

        if model_path is None:
            raise FileNotFoundError(f"No trained model found in {config.model_dir}")

    if test_data_dir is None:
        test_data_dir = config.test_data_dir

    if results_dir is None:
        results_dir = config.result_dir

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if config.evaluation_config.get("device") == "auto":
        device = device
    else:
        device = config.evaluation_config.get("device", device)

    model_kwargs = config.get_model_kwargs()
    data_config = config.get_data_config()
    eval_config = config.get_evaluation_config()

    os.makedirs(results_dir, exist_ok=True)

    print(f"Evaluating KeyDoor ToMnet model")
    print(f"Model: {model_path}")
    print(f"Test data: {test_data_dir}")
    print(f"Device: {device}")
    print("-" * 60)

    # Load model
    model = load_model(model_path, device, model_kwargs)
    print(f"Model loaded successfully")

    # Load test data
    print("Loading test data...")
    data_reader = DataReader()
    test_games = data_reader.ReadAllGames(test_data_dir)

    if len(test_games) == 0:
        raise ValueError(f"No test games found in {test_data_dir}")

    # Prepare test data using trajectory slicing (like training)
    test_data = prepare_data_for_training(
        test_games,
        min_timestep=6,  # Same as training
        max_trajectory_length=data_config["max_moves"],
    )

    # Create test dataset and loader
    test_dataset = TensorDataset(
        test_data["trajectories"], test_data["actions"], test_data["goals"]
    )
    test_loader = DataLoader(
        test_dataset, batch_size=eval_config["batch_size"], shuffle=False
    )

    print(f"Test data loaded: {len(test_dataset)} samples")

    # Evaluate model
    print("Evaluating model...")
    metrics = evaluate_model(
        model,
        test_loader,
        device,
        data_config=data_config,
        save_predictions=eval_config["save_predictions"],
        output_dir=results_dir,
    )

    print(f"Evaluation completed!")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")

    # Save results
    results_path = os.path.join(
        results_dir, f"evaluation_results_exp{config.experiment_no}.json"
    )
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Results saved to: {results_path}")

    return metrics


def evaluate_n_past_experiment(
    model, test_loader, output_dir, data_config=None, config=None
):
    """
    Evaluate model performance across different N_past values

    Args:
        model: Loaded ToMnet model
        test_loader: DataLoader for test data
        output_dir: Directory to save results
        data_config: Data configuration
        config: Config object containing parameters
    """
    if config is None:
        config = Config()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_past_config = config.get_n_past_evaluation_config()
    n_past_min = n_past_config["n_past_min"]
    n_past_max = n_past_config["n_past_max"]

    print(f"Evaluating N_past performance from {n_past_min} to {n_past_max}")
    print(f"Device: {device}")
    print("-" * 60)

    # Define N_past values to test
    n_past_values = list(range(n_past_min, n_past_max + 1))
    n_past_maximum = n_past_config["n_past_infer"]

    # Evaluate model with different N_past values
    print("Running evaluation...")
    results_by_n_past = evaluate_model_with_n_past(
        model, test_loader, device, n_past_values, n_past_maximum, data_config
    )

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save results
    results_file = os.path.join(output_dir, "n_past_evaluation_results.json")

    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for n_past, metrics in results_by_n_past.items():
        json_results[str(n_past)] = {
            "accuracy": float(metrics["accuracy"]),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "f1_score": float(metrics["f1_score"]),
        }

    with open(results_file, "w") as f:
        json.dump(json_results, f, indent=2)

    print(f"Results saved to: {results_file}")

    # Create visualizations
    print("Creating visualizations...")
    from visualize import (
        plot_accuracy_by_n_past,
        plot_accuracy_heatmap_by_n_past,
        plot_character_embeddings,
        create_additional_visualizations,
    )

    plot_accuracy_by_n_past(results_by_n_past, output_dir)
    plot_accuracy_heatmap_by_n_past(results_by_n_past, output_dir)

    # Create character embeddings visualization
    print("Creating character embeddings visualization...")
    plot_character_embeddings(
        model,
        test_loader,
        device,
        output_dir,
        experiment_no=config.experiment_no if config else 3,
        n_samples=1000,
    )

    print(f"Visualizations saved to: {output_dir}")

    return results_by_n_past


def analyze_action_likelihood(
    config=None, model=None, test_loader=None, n_samples=1000
):
    """
    Analyze action likelihood distributions

    Args:
        config: Config object containing evaluation parameters
        model: Trained model (if None, loads from config)
        test_loader: Test data loader (if None, creates from config)
        n_samples: Number of samples to analyze
    """
    if config is None:
        config = Config()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = config.result_dir
    data_config = config.get_data_config()

    if model is None:
        # Find the best model in results directory
        model_path = os.path.join(config.model_dir, "best_model.pth")
        if model_path is None:
            raise FileNotFoundError(f"No trained model found in {config.model_dir}")

        model_kwargs = config.get_model_kwargs()
        model = load_model(model_path, device, model_kwargs)

    if test_loader is None:
        # Load test data
        data_reader = DataReader()
        test_games = data_reader.ReadAllGames(config.test_data_dir)
        test_data = prepare_data_for_training(
            test_games,
            min_timestep=6,  # Same as training
            max_trajectory_length=data_config["max_moves"],
        )
        test_dataset = TensorDataset(
            test_data["trajectories"], test_data["actions"], test_data["goals"]
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.evaluation_config["batch_size"],
            shuffle=False,
        )

    model.eval()
    action_likelihoods = {i: [] for i in range(7)}  # 7 actions in KeyDoor

    sample_count = 0
    with torch.no_grad():
        for batch in test_loader:
            if sample_count >= n_samples:
                break

            if len(batch) >= 3:
                trajectories, actions, goals = batch[:3]
                trajectories = trajectories.to(device)
                actions = actions.to(device)
                goals = goals.to(device)

                batch_size = trajectories.size(0)

                # Generate past episodes
                past_episodes = generate_past_episodes_from_batch(
                    trajectories,
                    goals,
                    batch_size,
                    n_past_min=data_config.get("n_past_min", 1),
                    n_past_max=data_config.get("n_past_max", 1),
                    max_n_past=data_config.get("max_n_past", 1),
                )

                # Get current trajectory
                current_timestep = data_config["time_step"]
                recent_trajectory = trajectories[:, :current_timestep]

                # Extract current state for PredNet
                current_state = trajectories[
                    :, current_timestep - 1
                ]  # [batch, channels, height, width]

                # Get action targets - use actions[:, 0] for trajectory slicing
                action_targets = actions[
                    :, 0
                ]  # Target action for each sliced trajectory

                # Model forward pass (model returns 6 outputs)
                action_logits, _, _, _, _, _ = model(
                    past_episodes, recent_trajectory, current_state
                )
                probabilities = F.softmax(action_logits, dim=1)

                for i in range(len(action_targets)):
                    if sample_count >= n_samples:
                        break

                    action = action_targets[i].item()
                    if action < 7:  # Ensure valid action
                        likelihood = probabilities[i, action].item()
                        action_likelihoods[action].append(likelihood)
                    sample_count += 1

    # Save analysis
    os.makedirs(output_dir, exist_ok=True)
    analysis_path = os.path.join(output_dir, "action_likelihood_analysis.pkl")
    with open(analysis_path, "wb") as f:
        pickle.dump(action_likelihoods, f)

    # Calculate statistics
    stats = {}
    for action, likelihoods in action_likelihoods.items():
        if likelihoods:
            stats[f"action_{action}"] = {
                "mean": float(np.mean(likelihoods)),
                "std": float(np.std(likelihoods)),
                "median": float(np.median(likelihoods)),
                "count": len(likelihoods),
            }

    stats_path = os.path.join(output_dir, "action_likelihood_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Action likelihood analysis saved to: {output_dir}")
    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate KeyDoor ToMnet model")
    parser.add_argument(
        "--config_override",
        action="store_true",
        help="Override config with command line arguments",
    )
    parser.add_argument(
        "--test_data_dir", type=str, help="Directory containing test data"
    )
    parser.add_argument(
        "--model_dir", type=str, help="Directory containing trained models"
    )
    parser.add_argument("--model_path", type=str, help="Path to specific trained model")
    parser.add_argument(
        "--result_dir", type=str, help="Directory to save evaluation results"
    )
    parser.add_argument("--experiment_no", type=int, help="Experiment number")
    parser.add_argument("--batch_size", type=int, help="Evaluation batch size")
    parser.add_argument("--device", type=str, help="CUDA device (e.g., cuda:0)")
    parser.add_argument(
        "--n_samples", type=int, default=1000, help="Number of samples for analysis"
    )
    parser.add_argument(
        "--n_past_min", type=int, default=0, help="Minimum N_past value"
    )
    parser.add_argument(
        "--n_past_max", type=int, default=4, help="Maximum N_past value"
    )
    parser.add_argument(
        "--save_predictions", action="store_true", help="Save predictions to file"
    )
    parser.add_argument(
        "--plot_type",
        type=str,
        choices=["basic", "embeddings", "n_past", "all"],
        default="basic",
        help="Type of visualization to create",
    )

    args = parser.parse_args()

    config = Config()

    # Override config with command line arguments if specified
    if args.config_override:
        config.update_from_args(args)

    # Run evaluation
    results = evaluate_keydoor_model(
        config=config,
        model_path=args.model_path,
        test_data_dir=args.test_data_dir,
        results_dir=args.result_dir,
    )
    print("Evaluation completed successfully!")

    # Create additional visualizations if requested
    if args.plot_type in ["embeddings", "all"]:
        print("Creating additional visualizations...")
        from visualize import plot_character_embeddings

        # Load model and test data for visualization
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = args.model_path or os.path.join(config.model_dir, "best_model.pth")
        model_kwargs = config.get_model_kwargs()
        model = load_model(model_path, device, model_kwargs)

        # Load test data
        data_reader = DataReader()
        test_games = data_reader.ReadAllGames(
            args.test_data_dir or config.test_data_dir
        )
        data_config = config.get_data_config()
        test_data = prepare_data_for_training(
            test_games, min_timestep=6, max_trajectory_length=data_config["max_moves"]
        )
        test_dataset = TensorDataset(
            test_data["trajectories"], test_data["actions"], test_data["goals"]
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.evaluation_config["batch_size"],
            shuffle=False,
        )

        # Plot character embeddings
        plot_character_embeddings(
            model,
            test_loader,
            device,
            args.result_dir or config.result_dir,
            experiment_no=config.experiment_no,
            n_samples=args.n_samples,
        )
        print("Additional visualizations completed!")

    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Overall Accuracy: {results['accuracy']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"Samples evaluated: {results['n_samples']}")

    # Per-action accuracy
    print("\nPer-Action Accuracy:")
    for action_key, accuracy in results["action_accuracy"].items():
        print(f"  {action_key}: {accuracy:.4f}")
