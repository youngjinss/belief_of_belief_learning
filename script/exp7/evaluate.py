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
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from tomnet import create_model
from config import Config
from utils import generate_past_episodes_from_batch
from utils import (
    set_seed,
    load_test_data_all_combinations,
    load_chunked_data_for_training,
    combine_all_combinations_data,
)

# Set seed using Config default value
config = Config()
set_seed(config.seed)

"""
Evaluation and metrics for AchieverBlocker ToMnet experiment
Adapted from ToMnetF experiment5 for multi-agent AchieverBlocker environment
"""


# Data loading functions moved to utils.py


def _calculate_trajectory_lengths(self_states):
    """
    Optimized calculation of effective self states lengths

    Args:
        self_states: Tensor of shape [batch_size, seq_len, channels, height, width]

    Returns:
        list: Effective lengths for each sample in batch
    """
    batch_size = self_states.size(0)

    # Vectorized calculation: sum over spatial dimensions for each timestep
    traj_sums = self_states.sum(dim=(2, 3, 4))  # [batch_size, seq_len]

    # Find last non-zero timestep for each batch sample
    non_zero_mask = traj_sums > 0

    # Use efficient masking to find last valid timestep
    seq_indices = torch.arange(self_states.size(1), device=self_states.device).expand(
        batch_size, -1
    )
    masked_indices = torch.where(
        non_zero_mask, seq_indices, torch.tensor(-1, device=self_states.device)
    )
    effective_lengths = masked_indices.max(dim=1)[0].clamp(min=0)

    # Convert to list and apply constraint
    return [max(1, length.item()) for length in effective_lengths]


def load_model(model_path, device, model_kwargs):
    """Load trained ToMnet model with enhanced error handling"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Try to load model configuration from saved files
    model_dir = os.path.dirname(model_path)

    # First try to load model_config.json
    model_config_path = os.path.join(model_dir, "model_config.json")
    if os.path.exists(model_config_path):
        print(f"Loading model configuration from: {model_config_path}")
        with open(model_config_path, "r") as f:
            saved_model_kwargs = json.load(f)
        # Use saved configuration instead of passed kwargs
        model_kwargs = saved_model_kwargs
    else:
        # Fallback: try to load full_config.json
        full_config_path = os.path.join(model_dir, "full_config.json")
        if os.path.exists(full_config_path):
            print(f"Loading model configuration from full config: {full_config_path}")
            with open(full_config_path, "r") as f:
                full_config = json.load(f)
                if "model_config" in full_config:
                    # Extract model kwargs from model_config
                    model_config = full_config["model_config"]
                    model_kwargs = {
                        "use_mentalnet": model_config.get("use_mentalnet", False),
                        "batch": model_kwargs.get(
                            "batch", 32
                        ),  # Keep batch size from current config
                        "residual_blocks": model_config.get("residual_blocks", 3),
                        "n_echar": model_config.get("n_echar", 64),
                        "n_ement": model_config.get("n_ement", 64),
                        "out_channels": model_config.get("out_channels", 32),
                        "channels_in": model_config.get("channels_in", 10),
                        "time_step": model_kwargs.get("time_step", 500),
                        "action_space": model_config.get("action_space", 7),
                        "goal_space": model_config.get("goal_space", 4),
                        "max_n_past": model_kwargs.get("max_n_past", 10),
                        "use_n_past": model_kwargs.get("use_n_past", True),
                        "env_width": model_config.get("env_width", 9),
                        "env_height": model_config.get("env_height", 9),
                        "hidden_size_lstm": model_config.get("hidden_size_lstm", 64),
                    }
        else:
            print(
                f"Warning: No saved model configuration found. Using provided kwargs."
            )

    print(
        f"Model configuration: use_mentalnet={model_kwargs.get('use_mentalnet', False)}"
    )
    model = create_model(model_kwargs)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        # Checkpoint format with optimizer state
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict) and "model_state_dict" not in checkpoint:
        # Direct state dict format
        model.load_state_dict(checkpoint)
    else:
        # Fallback: assume it's a direct state dict
        model.load_state_dict(checkpoint)

    # Move model to device and set to eval mode (always execute this)
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
                # Unpack all data including goal_ranks and types
                # Handle both old format (8 fields) and new format (10 fields)
                if len(batch) == 10:
                    # New format with opponent data
                    (
                        self_states,
                        self_actions,
                        goals,
                        goal_ranks,
                        agents,
                        types,
                        consumption_labels,
                        sr_labels,
                        oppo_states,
                        oppo_actions,
                    ) = batch
                elif len(batch) == 8:
                    # Old format without opponent data
                    (
                        self_states,
                        self_actions,
                        goals,
                        goal_ranks,
                        agents,
                        types,
                        consumption_labels,
                        sr_labels,
                    ) = batch
                    # Create dummy opponent data
                    oppo_states = torch.zeros_like(self_states)
                    oppo_actions = torch.zeros_like(self_actions)
                else:
                    raise ValueError(f"Unexpected batch format with {len(batch)} fields. Expected 8 (old format) or 10 (new format).")
                self_states = self_states.to(device)
                self_actions = self_actions.to(device)
                goals = goals.to(device)
                goal_ranks = goal_ranks.to(device)
                agents = agents.to(device)
                types = types.to(device)
                oppo_states = oppo_states.to(device)
                oppo_actions = oppo_actions.to(device)

                batch_size = self_states.size(0)

                # Generate past episodes with fixed n_past using goal_ranks
                past_episodes = generate_past_episodes_from_batch(
                    self_states,
                    goal_ranks,  # Use goal_ranks to match training
                    agents,
                    batch_size,
                    n_past,
                    n_past,
                    n_past_max,
                    rank_threshold=(
                        data_config.get("rank_threshold", 4) if data_config else 4
                    ),
                )

                # Use dynamic trajectory slicing (same as training/main evaluation)
                # Find the effective length for each sample
                traj_sums = self_states.sum(dim=(2, 3, 4))
                non_zero_mask = traj_sums > 0
                seq_indices = (
                    torch.arange(self_states.size(1), device=self_states.device)
                    .unsqueeze(0)
                    .expand(batch_size, -1)
                )
                masked_indices = torch.where(
                    non_zero_mask,
                    seq_indices,
                    torch.tensor(-1, device=self_states.device),
                )
                effective_lengths = masked_indices.max(dim=1)[0].clamp(min=0)

                # Use full trajectory (all channels)
                recent_trajectory = self_states

                # Extract current state using advanced indexing
                batch_indices = torch.arange(batch_size, device=self_states.device)
                current_state = self_states[
                    batch_indices, effective_lengths, :
                ]

                # Get action targets - use actions[:, -1] for trajectory slicing
                action_targets = self_actions[
                    :, -1
                ]  # Target action for each sliced trajectory

                # Create masked actions for temporal masking (mask target action at last position)
                # This matches the training process
                masked_self_actions = self_actions.clone()
                masked_self_actions[:, -1] = -1  # Mask the target action so model can't see it

                # Model forward pass (model returns dictionary)
                outputs = model(past_episodes, recent_trajectory, masked_self_actions, current_state, 
                              oppo_states=oppo_states, oppo_actions=oppo_actions)
                action_logits = outputs["action_logits"]

                # Get predictions
                _, predicted = torch.max(action_logits, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(action_targets.cpu().numpy())

        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average="weighted", zero_division=0
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
    model_kwargs=None,
):
    """
    Evaluate model performance with optimized memory usage and error handling

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
    try:
        model.eval()

        # Validate inputs
        if not test_loader:
            raise ValueError("test_loader cannot be None")
        if len(test_loader.dataset) == 0:
            raise ValueError("test_loader dataset is empty")

        # Pre-allocate arrays for better memory efficiency
        total_samples = len(test_loader.dataset)
        action_space = model_kwargs.get("action_space", 7) if model_kwargs else 7

        all_predictions = np.empty(total_samples, dtype=np.int64)
        all_targets = np.empty(total_samples, dtype=np.int64)
        all_probabilities = np.empty((total_samples, action_space), dtype=np.float32)
        
        sample_idx = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):

                # Unpack all data including goal_ranks and types
                # Handle both old format (8 fields) and new format (10 fields)
                if len(batch) == 10:
                    # New format with opponent data
                    (
                        self_states,
                        self_actions,
                        goals,
                        goal_ranks,
                        agents,
                        types,
                        consumption_labels,
                        sr_labels,
                        oppo_states,
                        oppo_actions,
                    ) = batch
                elif len(batch) == 8:
                    # Old format without opponent data
                    (
                        self_states,
                        self_actions,
                        goals,
                        goal_ranks,
                        agents,
                        types,
                        consumption_labels,
                        sr_labels,
                    ) = batch
                    # Create dummy opponent data
                    oppo_states = torch.zeros_like(self_states)
                    oppo_actions = torch.zeros_like(self_actions)
                else:
                    raise ValueError(f"Unexpected batch format with {len(batch)} fields. Expected 8 (old format) or 10 (new format).")

                batch_size = self_states.size(0)

                # Optimized GPU transfers with non_blocking for better performance
                self_states = self_states.to(device, non_blocking=True)
                self_actions = self_actions.to(device, non_blocking=True)
                goals = goals.to(device, non_blocking=True)
                goal_ranks = goal_ranks.to(device, non_blocking=True)
                agents = agents.to(device, non_blocking=True)
                oppo_states = oppo_states.to(device, non_blocking=True)
                oppo_actions = oppo_actions.to(device, non_blocking=True)

                # Generate past episodes using goal_ranks (same as training)
                past_episodes = generate_past_episodes_from_batch(
                    self_states,
                    goal_ranks,  # Use goal_ranks instead of goals to match training
                    agents,
                    batch_size,
                    n_past_min=data_config.get("n_past_min", 1) if data_config else 1,
                    n_past_max=data_config.get("n_past_max", 1) if data_config else 1,
                    max_n_past=data_config.get("max_n_past", 1) if data_config else 1,
                    rank_threshold=(
                        data_config.get("rank_threshold", 4) if data_config else 4
                    ),
                )

                # With trajectory slicing, we use dynamic timesteps
                # Each sample has a different effective length, stored in actions[:,0]

                # For trajectory slicing, use the action at the last index (the target action for this slice)
                action_targets = self_actions[
                    :, -1
                ]  # Target action for each sliced trajectory

                # Optimized trajectory length calculation
                effective_lengths = _calculate_trajectory_lengths(self_states)

                # Use full trajectory (all channels)
                recent_trajectory = self_states  # [batch_size, seq_len, channels_in, height, width]

                # Extract current state for PredNet (last non-padded timestep)
                current_state = torch.zeros(
                    batch_size,
                    self_states.size(2),  # Use full channels
                    self_states.size(3),
                    self_states.size(4),
                    device=device,
                )

                # Vectorized: Extract current state using advanced indexing on the same device
                batch_indices = torch.arange(batch_size, device=self_states.device)
                last_timesteps = torch.tensor(
                    [max(0, length - 1) for length in effective_lengths],
                    device=self_states.device,
                )

                # Extract current state using advanced indexing
                current_state = self_states[
                    batch_indices, last_timesteps, :
                ]

                # Create masked actions for temporal masking (mask target action at last position)
                # This matches the training process
                masked_self_actions = self_actions.clone()
                masked_self_actions[:, -1] = -1  # Mask the target action so model can't see it

                # Model forward pass (model returns dictionary)
                outputs = model(past_episodes, recent_trajectory, masked_self_actions, current_state, 
                                oppo_states=oppo_states, oppo_actions=oppo_actions)
                action_logits = outputs["action_logits"]

                # Get predictions
                probabilities = F.softmax(action_logits, dim=1)
                _, predicted = torch.max(action_logits, 1)
                # Efficiently store predictions in pre-allocated arrays
                batch_end = sample_idx + batch_size
                predicted_numpy = predicted.cpu().numpy()
                targets_numpy = action_targets.cpu().numpy()
                probabilities_numpy = probabilities.cpu().numpy()
                
                all_predictions[sample_idx:batch_end] = predicted_numpy
                all_targets[sample_idx:batch_end] = targets_numpy
                all_probabilities[sample_idx:batch_end] = probabilities_numpy
                sample_idx = batch_end

        # Data is already in numpy arrays - no conversion needed
        predictions = all_predictions
        targets = all_targets
        probabilities = all_probabilities
        
        # Calculate metrics
        accuracy = accuracy_score(targets, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, predictions, average="weighted", zero_division=0
        )

        # Force confusion matrix to be for AchieverBlocker (actions vary by agent type)
        # Determine max action from data to handle both achiever and blocker actions
        max_action = max(np.max(targets), np.max(predictions)) + 1
        # Only include labels that exist in the data to avoid confusion matrix errors
        existing_labels = sorted(set(np.concatenate([targets, predictions])))
        
        conf_matrix = confusion_matrix(
            targets, predictions, labels=existing_labels
        )

        # Action-wise accuracy - AchieverBlocker has variable actions based on agent type
        action_accuracy = {}
        for action in existing_labels:  # Use existing labels instead of range(max_action)
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

    except Exception as e:
        print(f"Error during model evaluation: {str(e)}")
        raise


def evaluate_achieverblocker_model(
    config=None,
    model_path=None,
    test_data_dir=None,
    results_dir=None,
    plot_type="basic",
):
    """
    Perform evaluation on AchieverBlocker ToMnet model

    Args:
        config: Config object containing all evaluation parameters
        model_path: Path to trained model
        test_data_dir: Directory containing test data
        results_dir: Directory to save results
        plot_type: Type of evaluation to perform ("basic", "n_past", "char_embeddings", "mental_embeddings", "all")
    """
    if config is None:
        config = Config()

    # Use provided paths or default from config
    if model_path is None:
        # Find the best model in results directory
        model_path = os.path.join(config.model_dir, "best_model.pth")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No trained model found at {model_path}")

    if test_data_dir is None:
        # Get base data directory from config
        test_data_dir = os.path.join(config.save_dir, config.get_env_name())

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

    print(f"Evaluating AchieverBlocker ToMnet model on all combinations")
    print(f"Model: {model_path}")
    print(f"Test data: {test_data_dir}")
    print(f"Results directory: {results_dir}")
    print(f"Device: {device}")
    print(f"Achiever types: {list(config.achiever_types.keys())}")
    if config.is_single_agent_mode():
        print(f"Single-agent mode: No blockers")
        print(f"Total achiever types: {len(config.achiever_types)}")
    else:
        print(f"Blocker types: {list(config.blocker_types.keys())}")
        print(f"Total combinations: {len(config.achiever_types)} x {len(config.blocker_types)} = {len(config.achiever_types) * len(config.blocker_types)}")
    print("-" * 60)

    # Load model ONCE
    model = load_model(model_path, device, model_kwargs)
    print(model)
    print(f"Model loaded successfully")

    # Load test data for all combinations efficiently
    all_test_data = load_test_data_all_combinations(config, test_data_dir_base=test_data_dir)

    # Combine data from all combinations
    test_data = combine_all_combinations_data(all_test_data)
    
    total_test_samples = test_data["self_states"].shape[0]

    print(f"Test data usage:")
    print(f"  Using all test samples: {total_test_samples}")

    if total_test_samples == 0:
        raise ValueError(
            f"No test data found. Please generate test data first using --test_data flag."
        )

    # Log test data shapes for verification
    print(f"Test data shapes:")
    print(f"Self states: {test_data['self_states'].shape}")
    print(f"Actions: {test_data['actions'].shape}")
    print(f"Goals: {test_data['goals'].shape}")
    print(f"Goal ranks: {test_data['goal_ranks'].shape}")
    print(f"Consumption labels: {test_data['consumption_labels'].shape}")
    print(f"SR labels: {test_data['sr_labels'].shape}")

    # Convert numpy arrays to tensors for TensorDataset
    test_tensors = {
        key: torch.from_numpy(data) if isinstance(data, np.ndarray) else torch.tensor(data)
        for key, data in test_data.items()
    }
    
    # Create test dataset and loader with all required data including goal_ranks
    test_dataset = TensorDataset(
        test_tensors["self_states"],
        test_tensors["actions"],
        test_tensors["goals"],
        test_tensors["goal_ranks"],
        test_tensors["agents"],
        test_tensors["types"],
        test_tensors["consumption_labels"],
        test_tensors["sr_labels"],
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
        model_kwargs=model_kwargs,
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

    # Run N_past evaluation if requested (using same model and test_loader)
    if plot_type in ["n_past", "all"]:
        print("Running N_past evaluation...")
        evaluate_n_past_experiment(model, test_loader, results_dir, data_config, config)
        print("N_past evaluation completed!")

    # Create character embeddings if requested (using same model and test_loader)
    if plot_type in ["char_embeddings", "all"]:
        print("Creating character embedding visualizations...")
        # Import locally to avoid circular import
        import visualize

        visualize.plot_character_embeddings(
            model,
            test_loader,
            device,
            results_dir,
            experiment_no=config.experiment_no,
            n_samples=config.evaluation_config.get("n_samples", 1000),
        )
        print("Character embedding visualization completed!")

    # Create mental embeddings if requested (using same model and test_loader)
    if plot_type in ["mental_embeddings", "all"]:
        print("Creating mental embedding visualizations...")
        # Import locally to avoid circular import
        import visualize

        visualize.plot_mental_embeddings(
            model,
            test_loader,
            device,
            results_dir,
            config=config,
            experiment_no=config.experiment_no,
            n_samples=config.evaluation_config.get("n_samples", 1000),
        )
        print("Mental embedding visualization completed!")

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
    # Import locally to avoid circular import
    import visualize

    visualize.plot_accuracy_by_n_past(results_by_n_past, output_dir)
    visualize.plot_accuracy_heatmap_by_n_past(results_by_n_past, output_dir, config)

    # Create character embeddings visualization
    print("Creating character embeddings visualization...")
    visualize.plot_character_embeddings(
        model,
        test_loader,
        device,
        output_dir,
        experiment_no=config.experiment_no if config else 5,
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
        # Load test data - combine all combinations based on single-agent vs multi-agent mode
        if config.is_single_agent_mode():
            # For single-agent mode, use the first achiever type's test directory  
            achiever_type = list(config.achiever_types.keys())[0]
            test_data_dir_default = os.path.dirname(config.get_training_data_path(achiever_type, None, is_test=True))
        else:
            # For multi-agent mode, use environment base path
            env_name = config.get_env_name()
            test_data_dir_default = f"./data/{env_name}"

        # Load test data for all combinations efficiently
        all_test_data = load_test_data_all_combinations(config, test_data_dir_base=test_data_dir_default)

        # Combine data from all combinations
        test_data = combine_all_combinations_data(all_test_data)

        # Use all available test data (no sampling)
        total_test_samples = test_data["self_states"].shape[0]

        print(f"Test data usage:")
        print(f"  Using all available test samples: {total_test_samples}")

        if total_test_samples == 0:
            raise ValueError(
                f"No test data found. Please generate test data first using --test_data flag."
            )

        test_dataset = TensorDataset(
            test_data["self_states"],
            test_data["actions"],
            test_data["goals"],
            test_data["goal_ranks"],
            test_data["agents"],
            test_data["types"],
            test_data["consumption_labels"],
            test_data["sr_labels"],
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.evaluation_config["batch_size"],
            shuffle=False,
        )

    model.eval()
    # Determine max action space based on mode
    if config.is_single_agent_mode():
        max_action = 7  # KeyDoor single-agent actions (0-6)
    else:
        max_action = 8  # AchieverBlocker multi-agent (7 achiever + 6 blocker actions max)
    action_likelihoods = {i: [] for i in range(max_action)}

    sample_count = 0
    with torch.no_grad():
        for batch in test_loader:
            if sample_count >= n_samples:
                break

            # Unpack all data including goal_ranks and types
            # Handle both old format (8 fields) and new format (10 fields)
            if len(batch) == 10:
                # New format with opponent data
                (
                    self_states,
                    self_actions,
                    goals,
                    goal_ranks,
                    agents,
                    types,
                    consumption_labels,
                    sr_labels,
                    oppo_states,
                    oppo_actions,
                ) = batch
            elif len(batch) == 8:
                # Old format without opponent data
                (
                    self_states,
                    self_actions,
                    goals,
                    goal_ranks,
                    agents,
                    types,
                    consumption_labels,
                    sr_labels,
                ) = batch
                # Create dummy opponent data
                oppo_states = torch.zeros_like(self_states)
                oppo_actions = torch.zeros_like(self_actions)
            else:
                raise ValueError(f"Unexpected batch format with {len(batch)} fields. Expected 8 (old format) or 10 (new format).")
                
            self_states = self_states.to(device)
            self_actions = self_actions.to(device)
            goals = goals.to(device)
            goal_ranks = goal_ranks.to(device)
            agents = agents.to(device)
            oppo_states = oppo_states.to(device)
            oppo_actions = oppo_actions.to(device)

            batch_size = self_states.size(0)

            # Generate past episodes using goal_ranks
            past_episodes = generate_past_episodes_from_batch(
                self_states,
                goal_ranks,  # Use goal_ranks to match training
                agents,
                batch_size,
                n_past_min=data_config.get("n_past_min", 1),
                n_past_max=data_config.get("n_past_max", 1),
                max_n_past=data_config.get("max_n_past", 1),
                rank_threshold=data_config.get("rank_threshold", 4),
            )

            # Use dynamic trajectory slicing (same as training)
            # Find the effective length for each sample
            traj_sums = self_states.sum(dim=(2, 3, 4))
            non_zero_mask = traj_sums > 0
            seq_indices = (
                torch.arange(self_states.size(1), device=self_states.device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )
            masked_indices = torch.where(
                non_zero_mask,
                seq_indices,
                torch.tensor(-1, device=self_states.device),
            )
            effective_lengths = masked_indices.max(dim=1)[0].clamp(min=0)

            # Use full trajectory (all channels)
            recent_trajectory = self_states

            # Extract current state using advanced indexing
            batch_indices = torch.arange(batch_size, device=self_states.device)
            current_state = self_states[
                batch_indices, effective_lengths, :
            ]

            # Get action targets - use actions[:, -1] for trajectory slicing
            action_targets = self_actions[:, -1].clone()  # Target action for each sliced trajectory
            
            # Create masked actions for temporal masking (mask target action at last position)
            masked_self_actions = self_actions.clone()
            masked_self_actions[:, -1] = -1  # Mask the target action so model can't see it

            # Model forward pass (model returns dictionary)
            outputs = model(past_episodes, recent_trajectory, masked_self_actions, current_state,
                          oppo_states=oppo_states, oppo_actions=oppo_actions)
            action_logits = outputs["action_logits"]
            probabilities = F.softmax(action_logits, dim=1)

            for i in range(len(action_targets)):
                if sample_count >= n_samples:
                    break

                action = action_targets[i].item()
                if action < max_action:  # Ensure valid action
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
        description="Evaluate AchieverBlocker ToMnet model"
    )
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
    parser.add_argument("--n_past_min", type=int, help="Minimum N_past value")
    parser.add_argument("--n_past_max", type=int, help="Maximum N_past value")
    parser.add_argument(
        "--save_predictions", action="store_true", help="Save predictions to file"
    )
    parser.add_argument(
        "--plot_type",
        type=str,
        choices=["basic", "char_embeddings", "mental_embeddings", "n_past", "all"],
        default="all",
        help="Type of visualization to create",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    config = Config()

    # Override config with command line arguments if specified
    if args.config_override:
        config.update_from_args(args)

    # Set seed for reproducibility
    seed = args.seed if hasattr(args, "seed") else config.seed
    seed_worker = set_seed(seed)
    print(f"Set random seed to {seed} for reproducibility")

    # Run evaluation (now handles n_past and char_embeddings/mental_embeddings internally)
    results = evaluate_achieverblocker_model(
        config=config,
        model_path=args.model_path,
        test_data_dir=args.test_data_dir,
        results_dir=args.result_dir,
        plot_type=args.plot_type,
    )
    print("Evaluation completed successfully!")

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
