import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
import pickle

from tomnet import ToMnet
from data_generation import ToMnetDataset, collate_fn
from agents import RandomAgent


def compute_kl_divergence(
    p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10
) -> float:
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
    """Bayes-optimal inference baseline for Figure 3"""

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

        results = {"n_past_values": [], "kl_divergences": [], "action_accuracies": []}

        for sample in dataset["data"]:
            # Extract past actions
            past_actions = []
            for traj in sample["past_trajectories"]:
                if len(traj.actions) > 0:
                    past_actions.append(traj.actions[0])  # Use first action

            # Update posterior
            posterior_alpha = self.update_posterior(prior_alpha, past_actions)
            predicted_policy = self.predict_policy(posterior_alpha)

            # Compare with true policy
            true_policy = sample["true_policy"]
            kl_div = compute_kl_divergence(true_policy, predicted_policy)

            # Action prediction accuracy
            predicted_action = np.argmax(predicted_policy)
            true_action = sample["query_action"]
            accuracy = 1.0 if predicted_action == true_action else 0.0

            results["n_past_values"].append(sample["n_past"])
            results["kl_divergences"].append(kl_div)
            results["action_accuracies"].append(accuracy)

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
        """Evaluate action prediction accuracy"""
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

                # Get predicted and true policies
                predicted_policy = predictions["action_probs"][0].cpu().numpy()

                # Get true policy from original dataset
                sample_idx = 0  # Since batch_size=1
                original_sample = dataset.data[sample_idx]
                true_policy = original_sample["true_policy"]

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
    model = create_tomnet(experiment_type=experiment_type, state_dim=state_dim)

    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def evaluate_unified_results(
    experiment_type: str,
    model_paths: Union[str, Dict[str, str]],
    dataset_paths: Union[str, Dict[str, str]],
    device: str = "cuda",
) -> Dict:
    """Unified evaluation function for both Figure 3 and Figure 5 experiments"""
    results = {}
    state_dim = 11 * 11 * 6  # 11x11 grid with 6 channels

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

            elif experiment_type == "figure5":
                # Extract agent rewards for embedding analysis
                agent_rewards = {}
                for sample in dataset["data"]:
                    agent_id = sample["agent_id"]
                    if agent_id not in agent_rewards:
                        agent_rewards[agent_id] = sample["rewards"]

                dataset_results["agent_rewards"] = agent_rewards

            model_results[dataset_name] = dataset_results

        # Add baseline comparison for figure3
        if experiment_type == "figure3":
            # Try to find alpha datasets for baseline comparison
            alpha_dataset = None
            for dataset_name in datasets.keys():
                if "alpha_0.01" in dataset_name or "alpha" in dataset_name:
                    alpha_dataset = datasets[dataset_name]
                    break

            if alpha_dataset is not None:
                baseline = BayesOptimalBaseline()
                baseline_results = baseline.evaluate_on_data(alpha_dataset, 0.01)
                model_results["bayes_optimal_baseline"] = baseline_results

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

    # If it's a figure3 experiment, also evaluate policy prediction
    if dataset.experiment_type == "figure3":
        policy_results = evaluator.evaluate_policy_prediction(dataset)
        return {
            "action_accuracy": action_results["accuracy"],
            "action_loss": action_results["average_loss"],
            "mean_kl_divergence": np.mean(policy_results["kl_divergences"]),
            "mean_js_divergence": np.mean(policy_results["js_divergences"]),
        }
    else:
        return {
            "action_accuracy": action_results["accuracy"],
            "action_loss": action_results["average_loss"],
        }


def main():
    """Example evaluation script"""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate ToMnet")
    parser.add_argument("--experiment", choices=["figure3", "figure5"], required=True)
    parser.add_argument("--model_path", required=True, help="Path to trained model")
    parser.add_argument("--data_path", required=True, help="Path to test dataset")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument(
        "--output_path", default="evaluation_results.pkl", help="Path to save results"
    )

    args = parser.parse_args()

    # Evaluate using unified function
    results = evaluate_unified_results(
        args.experiment, args.model_path, args.data_path, args.device
    )

    # Save results
    with open(args.output_path, "wb") as f:
        pickle.dump(results, f)

    print(f"Evaluation results saved to {args.output_path}")


if __name__ == "__main__":
    main()
