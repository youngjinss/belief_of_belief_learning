import torch
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

            # Get model predictions
            output = model([traj, curr])
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


def cross_species_evaluation(
    model_paths,
    test_data_paths,
    device="cuda:0",
    result_dir="../../result/experiment1",
    experiment_no=1,
    **model_kwargs,
):
    """
    Perform cross-species evaluation across different model types

    Args:
        model_paths: List of paths to trained models
        test_data_paths: List of paths to test datasets
        device: Computing device
        result_dir: Directory to save results
        experiment_no: Experiment number
        **model_kwargs: Model initialization parameters
    """

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
            test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

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


def analyze_action_likelihood(model, test_loader, device, output_dir, n_samples=1000):
    """
    Analyze action likelihood distributions

    Args:
        model: Trained model
        test_loader: Test data loader
        device: Computing device
        output_dir: Output directory
        n_samples: Number of samples to analyze
    """

    model.eval()
    action_likelihoods = {i: [] for i in range(4)}

    sample_count = 0
    with torch.no_grad():
        for traj, curr, act in test_loader:
            if sample_count >= n_samples:
                break

            traj, curr, act = traj.to(device), curr.to(device), act.to(device)
            act = act.squeeze(-1).type(torch.long)

            output = model([traj, curr])
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
        description="Cross-species evaluation and metrics for ToMnetF"
    )
    parser.add_argument(
        "--model_paths",
        type=str,
        nargs="+",
        default=["../../models/experiment1/exp1_best.pth"],
        help="List of paths to trained models",
    )
    parser.add_argument(
        "--test_data_paths",
        type=str,
        nargs="+",
        default=["../../data/experiment1/processed_data_exp1.pkl"],
        help="List of paths to test datasets",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Computing device (cuda:0, cpu)"
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="../../result/experiment1",
        help="Directory to save results",
    )
    parser.add_argument(
        "--experiment_no", type=int, default=1, help="Experiment number"
    )
    parser.add_argument("--batch_size", type=int, default=512, help="Model batch size")
    parser.add_argument(
        "--residual_blocks", type=int, default=5, help="Number of residual blocks"
    )
    parser.add_argument(
        "--n_echar", type=int, default=8, help="Character embedding size"
    )
    parser.add_argument("--out_channels", type=int, default=32, help="Output channels")
    parser.add_argument(
        "--trajectory_size", type=int, default=10, help="Max trajectory size"
    )
    parser.add_argument("--width", type=int, default=13, help="Map width")
    parser.add_argument("--height", type=int, default=13, help="Map height")
    parser.add_argument("--depth", type=int, default=10, help="Input depth")

    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    # Model parameters
    model_kwargs = {
        "Batch": args.batch_size,
        "ResidualBlocks": args.residual_blocks,
        "N_echar": args.n_echar,
        "out_channels": args.out_channels,
        "Max_trajectory_size": args.trajectory_size,
        "Width": args.width,
        "Height": args.height,
        "Depth": args.depth,
    }

    if all(os.path.exists(path) for path in args.model_paths + args.test_data_paths):
        results = cross_species_evaluation(
            model_paths=args.model_paths,
            test_data_paths=args.test_data_paths,
            device=device,
            result_dir=args.result_dir,
            experiment_no=args.experiment_no,
            **model_kwargs,
        )
        print(f"Evaluation completed successfully!")
    else:
        missing_files = [
            path
            for path in args.model_paths + args.test_data_paths
            if not os.path.exists(path)
        ]
        print(f"Missing files: {missing_files}")
        print("Please train the model and generate data first.")
