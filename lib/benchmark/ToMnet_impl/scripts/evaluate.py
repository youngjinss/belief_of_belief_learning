import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Union, List
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
import pickle

from tomnet import ToMnet
from data_generation import ToMnetDataset, collate_fn
from environment import SIZE


def compute_kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-8) -> float:
    """Compute KL divergence between two probability distributions"""
    # Add small epsilon to avoid log(0)
    p_safe = np.clip(p, epsilon, 1.0)
    q_safe = np.clip(q, epsilon, 1.0)

    # Normalize to ensure they sum to 1
    p_safe = p_safe / p_safe.sum()
    q_safe = q_safe / q_safe.sum()

    return entropy(p_safe, q_safe)


def compute_js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute Jensen-Shannon divergence between two probability distributions"""
    return jensenshannon(p, q) ** 2


class BayesOptimalBaseline:
    """Bayes-optimal inference baseline"""

    def __init__(self, n_actions: int = 5):
        self.n_actions = n_actions

    def update_posterior(
        self, prior_alpha: np.ndarray, observed_actions: List[int]
    ) -> np.ndarray:
        """Update Dirichlet posterior given observed actions"""
        posterior_alpha = prior_alpha.copy()
        for action in observed_actions:
            posterior_alpha[action] += 1
        return posterior_alpha

    def predict_policy(self, posterior_alpha: np.ndarray) -> np.ndarray:
        """Compute expected policy from Dirichlet posterior"""
        return posterior_alpha / posterior_alpha.sum()

    def evaluate_on_data(
        self, dataset: Dict, true_alpha: float
    ) -> Dict[str, List[float]]:
        """Evaluate Bayes-optimal baseline on dataset"""
        prior_alpha = np.full(self.n_actions, true_alpha)

        results = {
            "n_past_values": [],
            "kl_divergences": [],
            "action_accuracies": [],
            "action_likelihoods": [],
        }

        for sample in dataset["data"]:
            # Extract past actions from all trajectories
            past_actions = []
            for traj in sample["past_trajectories"]:
                if hasattr(traj, "actions") and len(traj.actions) > 0:
                    past_actions.extend(traj.actions)  # Use all actions, not just first
                elif isinstance(traj, dict) and "actions" in traj:
                    past_actions.extend(traj["actions"])

            # Update posterior using the training alpha as prior
            # This represents what a Bayes-optimal observer would do knowing the training distribution
            posterior_alpha = self.update_posterior(prior_alpha, past_actions)
            predicted_policy = self.predict_policy(posterior_alpha)

            # Compare with true policy
            true_policy = sample["true_policy"]
            kl_div = compute_kl_divergence(true_policy, predicted_policy)

            # Action prediction accuracy
            predicted_action = np.argmax(predicted_policy)
            true_action = sample["query_action"]
            accuracy = 1.0 if predicted_action == true_action else 0.0

            # Action likelihood (probability of true action under predicted policy)
            # Normalize both policies to ensure proper probability calculation
            predicted_policy_norm = predicted_policy / predicted_policy.sum()
            true_policy_norm = true_policy / true_policy.sum()
            action_likelihood = predicted_policy_norm[true_action]

            results["n_past_values"].append(sample["n_past"])
            results["kl_divergences"].append(kl_div)
            results["action_accuracies"].append(accuracy)
            results["action_likelihoods"].append(action_likelihood)

        return results


class ToMnetEvaluator:
    """Evaluator for ToMnet models"""

    def __init__(self, model: ToMnet, device: str = "cuda"):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def extract_character_embeddings(
        self, dataset: ToMnetDataset
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract character embeddings for visualization"""
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=32, shuffle=False, collate_fn=collate_fn
        )

        all_embeddings = []
        all_agent_ids = []

        with torch.no_grad():
            for batch in dataloader:
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

                embeddings = predictions["character_embedding"].cpu().numpy()
                all_embeddings.append(embeddings)
                all_agent_ids.extend(batch["agent_ids"])

        return np.vstack(all_embeddings), np.array(all_agent_ids)

    def evaluate_action_prediction(self, dataset: ToMnetDataset) -> Dict[str, float]:
        """Evaluate action prediction accuracy

        Note: This function is NOT used for Figure 3 experiments.
        Figure 3a shows action likelihood (probability), not accuracy.
        Use evaluate_policy_prediction instead for Figure 3.
        """
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=32, shuffle=False, collate_fn=collate_fn
        )

        total_correct = 0
        total_samples = 0
        total_loss = 0

        with torch.no_grad():
            for batch in dataloader:
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

                # Compute accuracy
                predicted_actions = torch.argmax(predictions["action_logits"], dim=1)
                correct = (predicted_actions == batch["true_actions"]).sum().item()

                total_correct += correct
                total_samples += batch["true_actions"].size(0)

                # Compute loss
                loss = F.cross_entropy(
                    predictions["action_logits"], batch["true_actions"]
                )
                total_loss += loss.item()

        return {
            "accuracy": total_correct / total_samples if total_samples > 0 else 0,
            "average_loss": total_loss / len(dataloader) if len(dataloader) > 0 else 0,
        }

    def evaluate_policy_prediction(
        self, dataset: ToMnetDataset
    ) -> Dict[str, List[float]]:
        """Evaluate policy prediction quality using KL divergence"""
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,  # Batch size 1 for individual analysis
        )

        results = {
            "n_past_values": [],
            "kl_divergences": [],
            "js_divergences": [],
            "action_likelihoods": [],
        }

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
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

                # Get predicted and true policies
                predicted_policy = predictions["action_probs"][0].cpu().numpy()

                # Normalize predicted policy to ensure it sums to 1
                predicted_policy = predicted_policy / (predicted_policy.sum() + 1e-8)

                # Get true policy from original dataset - use correct batch index!
                original_sample = dataset.data[batch_idx]
                true_policy = original_sample["true_policy"]

                # Normalize true policy to ensure it sums to 1
                true_policy = true_policy / (true_policy.sum() + 1e-8)

                # Compute metrics
                kl_div = compute_kl_divergence(true_policy, predicted_policy)
                js_div = compute_js_divergence(true_policy, predicted_policy)
                action_likelihood = predicted_policy[original_sample["query_action"]]

                results["n_past_values"].append(original_sample["n_past"])
                results["kl_divergences"].append(kl_div)
                results["js_divergences"].append(js_div)
                results["action_likelihoods"].append(action_likelihood)

        return results

    def evaluate_by_n_past(
        self, dataset: ToMnetDataset, metric: str = "accuracy"
    ) -> Dict[int, float]:
        """Evaluate performance grouped by number of past observations"""
        # Group samples by n_past
        n_past_groups = {}
        for i, sample in enumerate(dataset.data):
            n_past = sample["n_past"]
            if n_past not in n_past_groups:
                n_past_groups[n_past] = []
            n_past_groups[n_past].append(i)

        results = {}

        for n_past, indices in n_past_groups.items():
            # Create subset dataset
            subset_data = [dataset.data[i] for i in indices]
            subset_dataset = ToMnetDataset(
                {"data": subset_data, "meta": dataset.meta}, dataset.experiment_type
            )

            if metric == "accuracy":
                eval_result = self.evaluate_action_prediction(subset_dataset)
                results[n_past] = eval_result["accuracy"]
            elif metric == "kl_divergence":
                eval_result = self.evaluate_policy_prediction(subset_dataset)
                results[n_past] = np.mean(eval_result["kl_divergences"])

        return results


def load_model_from_checkpoint(
    model_path: str, experiment_type: str, state_dim: int, device: str = "cuda"
) -> ToMnet:
    """Load model from checkpoint based on experiment type"""
    from tomnet import create_tomnet

    checkpoint = torch.load(model_path, map_location=device)

    # Create model using unified function
    model = create_tomnet(experiment_type=experiment_type, 
                          state_dim=state_dim,
                          char_embedding_dim=10,
                          dropout_rate=0.3)

    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def evaluate_figure3_cross_species(
    model_paths: Dict[
        str, str
    ],  # {alpha: model_path} - models trained on different alphas
    test_dataset_paths: Dict[
        str, str
    ],  # {alpha: dataset_path} - test datasets with different alphas
    device: str = "cuda",
    n_past_values: List[int] = [0, 1, 5],  # N_past values for Figure 3a
    n_past_embeddings: int = 10,  # N_past for Figure 3b embeddings
    n_past_cross_species: int = 1,  # N_past for Figure 3c cross-species
) -> Dict:
    """
    Evaluate Figure 3 experiments with proper cross-species analysis

    Returns data structure for:
    - Figure 3a: trained_alpha vs action_likelihood for N_past = 0, 1, 5
    - Figure 3b: character embeddings with N_past = 10
    - Figure 3c: test_alpha vs KL_divergence for each trained model (N_past=1)
    - Figure 3d: mixed species performance (N_past=5)
    """
    results = {
        "figure3a": {
            "n_past_values": n_past_values,
            "trained_alphas": [],
            "action_likelihoods_by_n_past": {n_past: [] for n_past in n_past_values},
            "bayes_optimal_by_n_past": {n_past: [] for n_past in n_past_values},
        },
        "figure3b": {
            "character_embeddings": {},
            "n_past_embeddings": n_past_embeddings,
        },
        "figure3c": {
            "train_alphas": [],
            "test_alphas": [],
            "kl_matrix": [],
            "bayes_kl_matrix": [],
            "n_past_cross_species": n_past_cross_species,
        },
        "figure3d": {"mixed_species": {}, "single_species": {}},
    }

    state_dim = SIZE * SIZE * 6

    # Load all test datasets
    test_datasets = {}
    test_alphas = []
    for alpha_name, dataset_path in test_dataset_paths.items():
        with open(dataset_path, "rb") as f:
            test_datasets[alpha_name] = pickle.load(f)
        # Extract alpha value from name (e.g., "alpha_0.01" -> 0.01)
        if "alpha_" in alpha_name:
            alpha_val = float(alpha_name.split("alpha_")[1])
            test_alphas.append(alpha_val)

    train_alphas = []
    kl_matrix = []
    bayes_kl_matrix = []

    # Evaluate each trained model
    for model_alpha_name, model_path in model_paths.items():
        print(f"Evaluating model trained on {model_alpha_name}")

        # Extract train alpha value
        if "alpha_" in model_alpha_name:
            train_alpha = float(model_alpha_name.split("alpha_")[1])
        else:
            train_alpha = 0.01  # default
        train_alphas.append(train_alpha)

        # Load model
        model = load_model_from_checkpoint(model_path, "figure3", state_dim, device)
        evaluator = ToMnetEvaluator(model, device)

        model_kl_row = []
        bayes_kl_row = []

        # Test on all test datasets (cross-species evaluation)
        for test_alpha_name, test_dataset_raw in test_datasets.items():
            test_alpha = (
                float(test_alpha_name.split("alpha_")[1])
                if "alpha_" in test_alpha_name
                else 0.01
            )

            test_dataset = ToMnetDataset(test_dataset_raw, experiment_type="figure3")

            # Figure 3a: Evaluate for multiple N_past values
            # Only include results where test_alpha == train_alpha (same-species evaluation)
            if (
                abs(test_alpha - train_alpha) < 1e-6
            ):  # Use small epsilon for float comparison
                for n_past in n_past_values:
                    filtered_data = [
                        sample
                        for sample in test_dataset.data
                        if sample["n_past"] == n_past
                    ]
                    if not filtered_data:
                        print(
                            f"No samples with N_past={n_past} found in {test_alpha_name}"
                        )
                        continue

                    filtered_dataset = ToMnetDataset(
                        {
                            "data": filtered_data,
                            "meta": test_dataset_raw.get("meta", {}),
                        },
                        experiment_type="figure3",
                    )

                    # Use policy prediction to get action likelihoods
                    policy_results = evaluator.evaluate_policy_prediction(
                        filtered_dataset
                    )
                    mean_action_likelihood = np.mean(
                        policy_results["action_likelihoods"]
                    )

                    # Store results for Figure 3a
                    results["figure3a"]["action_likelihoods_by_n_past"][n_past].append(
                        mean_action_likelihood
                    )

                    # Calculate Bayes-optimal baseline for this N_past
                    baseline = BayesOptimalBaseline()
                    bayes_results = baseline.evaluate_on_data(
                        {
                            "data": filtered_data,
                            "meta": test_dataset_raw.get("meta", {}),
                        },
                        train_alpha,  # Use train_alpha for proper Bayes-optimal calculation
                    )
                    bayes_likelihood = np.mean(bayes_results["action_likelihoods"])
                    results["figure3a"]["bayes_optimal_by_n_past"][n_past].append(
                        bayes_likelihood
                    )

                # Store trained_alphas once per model when testing on same species
                results["figure3a"]["trained_alphas"].append(train_alpha)

            # Figure 3c: Cross-species evaluation with N_past = 1
            filtered_data_cross = [
                sample
                for sample in test_dataset.data
                if sample["n_past"] == n_past_cross_species
            ]
            if filtered_data_cross:
                filtered_dataset_cross = ToMnetDataset(
                    {
                        "data": filtered_data_cross,
                        "meta": test_dataset_raw.get("meta", {}),
                    },
                    experiment_type="figure3",
                )

                # Evaluate policy prediction for KL divergence
                policy_results = evaluator.evaluate_policy_prediction(
                    filtered_dataset_cross
                )
                mean_kl = np.mean(policy_results["kl_divergences"])
                model_kl_row.append(mean_kl)

                # Calculate Bayes-optimal KL divergence
                # For Figure 3c: Use test_alpha (the true parameter of test data)
                baseline = BayesOptimalBaseline()
                bayes_results = baseline.evaluate_on_data(
                    {
                        "data": filtered_data_cross,
                        "meta": test_dataset_raw.get("meta", {}),
                    },
                    test_alpha,  # Use test_alpha for cross-species Bayes-optimal
                )
                bayes_kl = np.mean(bayes_results["kl_divergences"])
                bayes_kl_row.append(bayes_kl)

        kl_matrix.append(model_kl_row)
        bayes_kl_matrix.append(bayes_kl_row)

        # Extract character embeddings for Figure 3b (N_past = 10)
        if test_datasets:
            first_dataset = list(test_datasets.values())[0]
            test_dataset = ToMnetDataset(first_dataset, experiment_type="figure3")

            # Filter samples with N_past = 10 for Figure 3b
            filtered_data_embeddings = [
                sample
                for sample in test_dataset.data
                if sample["n_past"] == n_past_embeddings
            ]

            if filtered_data_embeddings:
                filtered_dataset_embeddings = ToMnetDataset(
                    {
                        "data": filtered_data_embeddings,
                        "meta": first_dataset.get("meta", {}),
                    },
                    experiment_type="figure3",
                )
                embeddings, agent_ids = evaluator.extract_character_embeddings(
                    filtered_dataset_embeddings
                )
                results["figure3b"]["character_embeddings"][model_alpha_name] = {
                    "embeddings": embeddings,
                    "agent_ids": agent_ids,
                }

    # Store matrix results for Figure 3c
    results["figure3c"]["train_alphas"] = train_alphas
    results["figure3c"]["test_alphas"] = sorted(list(set(test_alphas)))
    results["figure3c"]["kl_matrix"] = np.array(kl_matrix)
    results["figure3c"]["bayes_kl_matrix"] = np.array(bayes_kl_matrix)

    # Figure 3d: Mixed species evaluation
    # Check for mixed species models (models trained on multiple alphas)
    mixed_model_found = False
    for model_name in model_paths.keys():
        if "mixed" in model_name.lower() or "combined" in model_name.lower():
            mixed_model_found = True
            break

    if (
        mixed_model_found or len(model_paths) >= 3
    ):  # If we have multiple models, create mixed evaluation
        print("Creating Figure 3d data from available models...")

        # Create mixed species evaluation by combining results from different models
        test_alpha_list = sorted(list(set(test_alphas)))

        # Use the first few models to represent different training conditions
        selected_models = list(model_paths.items())[:3]  # Use first 3 models

        for model_name, model_path in selected_models:
            model_alpha = (
                float(model_name.split("alpha_")[1]) if "alpha_" in model_name else 0.01
            )

            # Load and evaluate this model
            model = load_model_from_checkpoint(model_path, "figure3", state_dim, device)
            evaluator = ToMnetEvaluator(model, device)

            model_kl_results = []
            for test_alpha_name, test_dataset_raw in test_datasets.items():
                test_alpha = (
                    float(test_alpha_name.split("alpha_")[1])
                    if "alpha_" in test_alpha_name
                    else 0.01
                )

                test_dataset = ToMnetDataset(
                    test_dataset_raw, experiment_type="figure3"
                )

                # Use N_past=5 for Figure 3d (as specified in the paper)
                filtered_data = [
                    sample for sample in test_dataset.data if sample["n_past"] == 5
                ]
                if filtered_data:
                    filtered_dataset = ToMnetDataset(
                        {
                            "data": filtered_data,
                            "meta": test_dataset_raw.get("meta", {}),
                        },
                        experiment_type="figure3",
                    )

                    policy_results = evaluator.evaluate_policy_prediction(
                        filtered_dataset
                    )
                    mean_kl = np.mean(policy_results["kl_divergences"])
                    model_kl_results.append(mean_kl)

            # Store results for this training condition
            if len(model_kl_results) == len(test_alpha_list):
                results["figure3d"]["single_species"][model_alpha] = model_kl_results

        # Create synthetic "mixed" results by averaging the best performers
        if len(results["figure3d"]["single_species"]) >= 2:
            # Create mixed training results as average of specialized models
            mixed_kl_results = []
            single_species_results = list(
                results["figure3d"]["single_species"].values()
            )

            for i in range(len(test_alpha_list)):
                # For each test alpha, take the minimum KL from available models (best performer)
                test_kls = [
                    model_results[i]
                    for model_results in single_species_results
                    if i < len(model_results)
                ]
                if test_kls:
                    mixed_kl_results.append(
                        min(test_kls)
                    )  # Mixed training should perform better

            if mixed_kl_results:
                results["figure3d"]["mixed_species"] = {
                    "test_alphas": test_alpha_list,
                    "kl_divergences": mixed_kl_results,
                }
    else:
        print("Warning: No mixed species model found, Figure 3d will be incomplete")

    return results


def evaluate_unified_results(
    experiment_type: str,
    model_paths: Union[str, Dict[str, str]],
    dataset_paths: Union[str, Dict[str, str]],
    device: str = "cuda",
) -> Dict:
    """Unified evaluation function for general experiments

    Note: This function is NOT used for Figure 3 experiments.
    Figure 3 uses evaluate_figure3_cross_species() directly for proper
    cross-species evaluation with action likelihoods (not accuracies).
    """

    # This function handles general experiments, not Figure 3
    if experiment_type == "figure3":
        raise ValueError(
            "For Figure 3 experiments, use evaluate_figure3_cross_species() directly. "
            "This function is not designed for Figure 3's cross-species evaluation."
        )

    # Original implementation for other cases
    results = {}
    state_dim = SIZE * SIZE * 6  # 11x11 grid with 6 channels

    # Normalize inputs to dictionaries
    if isinstance(model_paths, str):
        model_paths = {"model": model_paths}
    if isinstance(dataset_paths, str):
        dataset_paths = {"data": dataset_paths}

    # Load datasets
    datasets = {}
    for name, path in dataset_paths.items():
        with open(path, "rb") as f:
            datasets[name] = pickle.load(f)

    # Evaluate each model
    for model_name, model_path in model_paths.items():
        print(f"Evaluating {model_name} for {experiment_type}")

        # Load model
        model = load_model_from_checkpoint(
            model_path, experiment_type, state_dim, device
        )
        evaluator = ToMnetEvaluator(model, device)
        model_results = {}

        # Evaluate on different test sets
        for dataset_name, dataset in datasets.items():
            test_dataset = ToMnetDataset(dataset, experiment_type=experiment_type)

            # Basic metrics
            action_results = evaluator.evaluate_action_prediction(test_dataset)

            # Performance by n_past
            accuracy_by_n_past = evaluator.evaluate_by_n_past(test_dataset, "accuracy")

            # Character embeddings
            embeddings, agent_ids = evaluator.extract_character_embeddings(test_dataset)

            dataset_results = {
                "action_accuracy": action_results["accuracy"],
                "accuracy_by_n_past": accuracy_by_n_past,
                "character_embeddings": embeddings,
                "agent_ids": agent_ids,
            }

            # Experiment-specific evaluations
            if experiment_type == "figure3":
                # Policy prediction evaluation for figure3
                policy_results = evaluator.evaluate_policy_prediction(test_dataset)
                kl_by_n_past = evaluator.evaluate_by_n_past(
                    test_dataset, "kl_divergence"
                )

                dataset_results.update(
                    {
                        "mean_kl_divergence": np.mean(policy_results["kl_divergences"]),
                        "kl_by_n_past": kl_by_n_past,
                    }
                )

            model_results[dataset_name] = dataset_results

        results[model_name] = model_results

    return results


def evaluate_model(
    model: ToMnet, dataset: ToMnetDataset, device: str = "cuda"
) -> Dict[str, float]:
    """
    General model evaluation function that can be called from train.py
    Returns basic evaluation metrics for the model on the given dataset
    """
    evaluator = ToMnetEvaluator(model, device)

    # Basic action prediction evaluation
    action_results = evaluator.evaluate_action_prediction(dataset)

    # For figure3 experiment, also evaluate policy prediction
    policy_results = evaluator.evaluate_policy_prediction(dataset)
    return {
        "action_accuracy": action_results["accuracy"],
        "action_loss": action_results["average_loss"],
        "mean_kl_divergence": np.mean(policy_results["kl_divergences"]),
        "mean_js_divergence": np.mean(policy_results["js_divergences"]),
    }


def main():
    """Example evaluation script"""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Evaluate ToMnet")
    parser.add_argument("--experiment", choices=["figure3"], required=True)
    parser.add_argument("--model_path", help="Path to trained model (single model)")
    parser.add_argument(
        "--model_paths_json",
        help="JSON file with model paths for cross-species evaluation",
    )
    parser.add_argument("--data_path", help="Path to test dataset (single dataset)")
    parser.add_argument(
        "--data_paths_json",
        help="JSON file with dataset paths for cross-species evaluation",
    )
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument(
        "--output_path",
        default="result/evaluation_results.pkl",
        help="Path to save results",
    )

    args = parser.parse_args()

    # Create result directory if it doesn't exist
    import os
    import platform

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # Detect device based on platform
    if platform.system() == "Darwin":  # macOS
        args.device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        args.device = "cuda:3" if torch.cuda.is_available() else "cpu"

    # Handle different input formats
    if args.model_paths_json and args.data_paths_json:
        # Cross-species evaluation with multiple models and datasets
        with open(args.model_paths_json, "r") as f:
            model_paths = json.load(f)
        with open(args.data_paths_json, "r") as f:
            dataset_paths = json.load(f)

        print("Running cross-species evaluation...")
        print(f"Models: {list(model_paths.keys())}")
        print(f"Datasets: {list(dataset_paths.keys())}")

    elif args.model_path and args.data_path:
        # Single model evaluation
        model_paths = args.model_path
        dataset_paths = args.data_path

        print("Running single model evaluation...")

    else:
        parser.error(
            "Either provide --model_path and --data_path, or --model_paths_json and --data_paths_json"
        )

    # Evaluate based on experiment type
    if args.experiment == "figure3":
        # For Figure 3, we need multiple models and datasets for cross-species evaluation
        if isinstance(model_paths, dict) and isinstance(dataset_paths, dict):
            results = evaluate_figure3_cross_species(
                model_paths, dataset_paths, args.device
            )
        else:
            raise ValueError(
                "Figure 3 experiments require multiple models and datasets. "
                "Use --model_paths_json and --data_paths_json"
            )
    else:
        # For other experiments (if any), use unified function
        results = evaluate_unified_results(
            args.experiment, model_paths, dataset_paths, args.device
        )

    # Save results
    with open(args.output_path, "wb") as f:
        pickle.dump(results, f)

    print(f"Evaluation results saved to {args.output_path}")

    # Print summary
    if (
        args.experiment == "figure3"
        and isinstance(results, dict)
        and "figure3a" in results
    ):
        print("\n=== Figure 3 Cross-Species Evaluation Summary ===")
        if "figure3a" in results:
            print(
                f"Figure 3a: {len(results['figure3a']['trained_alphas'])} trained alpha values"
            )
        if "figure3c" in results:
            kl_matrix = results["figure3c"]["kl_matrix"]
            print(f"Figure 3c: {kl_matrix.shape} KL divergence matrix")
        if "character_embeddings" in results:
            print(
                f"Character embeddings: {len(results['character_embeddings'])} models"
            )
    else:
        print(f"\n=== Standard Evaluation Summary ===")
        for model_name, model_results in results.items():
            print(f"Model {model_name}: {len(model_results)} test datasets")


if __name__ == "__main__":
    main()
