import torch
import sys
sys.path.append("..")
import torch.nn.functional as F
import numpy as np
import json
import os
import pickle
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from tomnet import ToMnet
from torch.utils.data import DataLoader, TensorDataset
from config import Config

"""
Cross-species evaluation and metrics for ToMnetF
@Author Filip Borowiak
"""


def load_model(model_path, device, **model_kwargs):
    """Load trained ToMnet model"""
    model = ToMnet(**model_kwargs)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def evaluate_model(model, test_loader, device, save_predictions=False, output_dir=None):
    """
    Evaluate model performance

    Args:
        model: Trained ToMnet model
        test_loader: Test data loader
        device: Computing device
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
        for batch_idx, (traj, curr, act) in enumerate(test_loader):
            traj, curr, act = traj.to(device), curr.to(device), act.to(device)
            act = act.squeeze(-1).type(torch.long)

            # Get model predictions - experiment2 model returns 3 outputs
            model_output = model([traj, curr])
            if isinstance(model_output, tuple) and len(model_output) == 3:
                action_pred, consumption_pred, sr_pred = model_output
                output = action_pred  # Use action predictions for evaluation
            else:
                output = model_output  # Single output case
            probabilities = F.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(act.cpu().numpy())
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

    # Action-wise accuracy
    action_accuracy = {}
    for action in range(4):  # 4 actions: UP, RIGHT, DOWN, LEFT
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


def cross_species_evaluation(config=None, model_paths=None, test_data_paths=None):
    """
    Perform cross-species evaluation across different model types

    Args:
        config: Config object containing all evaluation parameters
        model_paths: Optional list of paths to trained models (overrides config)
        test_data_paths: Optional list of paths to test datasets (overrides config)
    """
    if config is None:
        config = Config()

    # Use provided paths or default from config
    if model_paths is None:
        model_paths = [
            os.path.join(config.model_dir, f"exp{config.experiment_no}_best.pth")
        ]
    if test_data_paths is None:
        test_data_paths = [
            os.path.join(
                config.data_dir, f"processed_data_exp{config.experiment_no}.pkl"
            )
        ]

    device = config.device if torch.cuda.is_available() else "cpu"
    result_dir = config.result_dir
    experiment_no = config.experiment_no
    model_kwargs = config.get_model_kwargs()

    os.makedirs(result_dir, exist_ok=True)

    results = {"experiment_no": experiment_no, "cross_species_results": {}}

    print(f"Performing cross-species evaluation for experiment {experiment_no}")
    print("-" * 60)

    for model_idx, model_path in enumerate(model_paths):
        model_name = os.path.basename(model_path).replace(".pth", "")
        print(f"Evaluating model: {model_name}")

        # Load model
        model = load_model(model_path, device, **model_kwargs)

        model_results = {}

        for data_idx, test_data_path in enumerate(test_data_paths):
            data_name = os.path.basename(test_data_path).replace(".pkl", "")
            print(f"  Testing on data: {data_name}")

            # Load test data
            with open(test_data_path, "rb") as f:
                test_data = pickle.load(f)

            # Create test loader
            test_dataset = TensorDataset(
                test_data["data_trajectories"],
                test_data["data_current_state"],
                test_data["data_actions"],
            )
            test_loader = DataLoader(
                test_dataset, batch_size=config.batch_size, shuffle=False
            )

            # Evaluate
            metrics = evaluate_model(
                model,
                test_loader,
                device,
                save_predictions=True,
                output_dir=os.path.join(result_dir, f"{model_name}_{data_name}"),
            )

            model_results[data_name] = metrics
            print(f"    Accuracy: {metrics['accuracy']:.4f}")
            print(f"    F1 Score: {metrics['f1_score']:.4f}")

        results["cross_species_results"][model_name] = model_results
        print()

    # Save results
    results_path = os.path.join(
        result_dir, f"cross_species_evaluation_exp{experiment_no}.json"
    )
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Cross-species evaluation results saved to: {results_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("CROSS-SPECIES EVALUATION SUMMARY")
    print("=" * 60)

    for model_name, model_results in results["cross_species_results"].items():
        print(f"\nModel: {model_name}")
        for data_name, metrics in model_results.items():
            print(
                f"  {data_name}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}"
            )

    return results


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

    device = config.device if torch.cuda.is_available() else "cpu"
    output_dir = config.result_dir

    if model is None:
        model_path = os.path.join(
            config.model_dir, f"exp{config.experiment_no}_best.pth"
        )
        model_kwargs = config.get_model_kwargs()
        model = load_model(model_path, device, **model_kwargs)

    if test_loader is None:
        test_data_path = os.path.join(
            config.data_dir, f"processed_data_exp{config.experiment_no}.pkl"
        )
        with open(test_data_path, "rb") as f:
            test_data = pickle.load(f)
        test_dataset = TensorDataset(
            test_data["data_trajectories"],
            test_data["data_current_state"],
            test_data["data_actions"],
        )
        test_loader = DataLoader(
            test_dataset, batch_size=config.batch_size, shuffle=False
        )

    model.eval()
    action_likelihoods = {i: [] for i in range(4)}

    sample_count = 0
    with torch.no_grad():
        for traj, curr, act in test_loader:
            if sample_count >= n_samples:
                break

            traj, curr, act = traj.to(device), curr.to(device), act.to(device)
            act = act.squeeze(-1).type(torch.long)

            # Get model predictions - experiment2 model returns 3 outputs
            model_output = model([traj, curr])
            if isinstance(model_output, tuple) and len(model_output) == 3:
                action_pred, consumption_pred, sr_pred = model_output
                output = action_pred  # Use action predictions for evaluation
            else:
                output = model_output  # Single output case
            probabilities = F.softmax(output, dim=1)

            for i in range(len(act)):
                if sample_count >= n_samples:
                    break

                action = act[i].item()
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

    parser = argparse.ArgumentParser(
        description="Evaluate ToMnet model for Experiment 2"
    )
    parser.add_argument(
        "--config_override",
        action="store_true",
        help="Override config with command line arguments",
    )
    parser.add_argument("--data_dir", type=str, help="Directory containing test data")
    parser.add_argument(
        "--model_dir", type=str, help="Directory containing trained models"
    )
    parser.add_argument(
        "--result_dir", type=str, help="Directory to save evaluation results"
    )
    parser.add_argument("--experiment_no", type=int, help="Experiment number")
    parser.add_argument("--batch_size", type=int, help="Evaluation batch size")
    parser.add_argument("--device", type=str, help="CUDA device (e.g., cuda:0)")
    parser.add_argument(
        "--model_paths", type=str, nargs="+", help="Paths to trained models"
    )
    parser.add_argument(
        "--test_data_paths", type=str, nargs="+", help="Paths to test datasets"
    )
    parser.add_argument(
        "--analysis_only",
        action="store_true",
        help="Run action likelihood analysis only",
    )
    parser.add_argument(
        "--n_samples", type=int, default=1000, help="Number of samples for analysis"
    )

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
        if args.experiment_no is not None:
            config.experiment_no = args.experiment_no
        if args.batch_size is not None:
            config.batch_size = args.batch_size
        if args.device is not None:
            config.device = args.device

    # Use command line model and data paths if provided
    model_paths = args.model_paths
    test_data_paths = args.test_data_paths

    if model_paths is None:
        model_paths = [
            os.path.join(config.model_dir, f"exp{config.experiment_no}_best.pth")
        ]
    if test_data_paths is None:
        test_data_paths = [
            os.path.join(
                config.data_dir, f"processed_data_exp{config.experiment_no}.pkl"
            )
        ]

    if args.analysis_only:
        # Run action likelihood analysis only
        print("Running action likelihood analysis...")
        stats = analyze_action_likelihood(config=config, n_samples=args.n_samples)
        print(f"Action likelihood analysis completed!")
        for action, action_stats in stats.items():
            print(
                f"{action}: mean={action_stats['mean']:.4f}, std={action_stats['std']:.4f}"
            )
    else:
        # Run full cross-species evaluation
        if all(os.path.exists(path) for path in model_paths + test_data_paths):
            results = cross_species_evaluation(
                config=config, model_paths=model_paths, test_data_paths=test_data_paths
            )
            print(f"Evaluation completed successfully!")
        else:
            missing_files = [
                path
                for path in model_paths + test_data_paths
                if not os.path.exists(path)
            ]
            print(f"Missing files: {missing_files}")
            print("Please train the model and generate data first.")
