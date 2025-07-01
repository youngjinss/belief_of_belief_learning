#!/usr/bin/env python3
"""
Evaluation script for Figure 5 experiments with goal-directed agents
This script evaluates trained ToMnet models on goal-directed agent data for Figure 5b,d reproduction
"""

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import os
import pickle
import json
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Union
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import glob

from tomnet import ToMnet, create_tomnet
from data_generation import DataGenerator, ToMnetDataset, collate_fn
from evaluate import compute_kl_divergence, BayesOptimalBaseline
from train_figure5 import Figure5ExperimentConfig


def load_model(model_path: str, device: str = "cuda") -> Tuple[ToMnet, Dict]:
    """Load trained ToMnet model"""
    print(f"Loading model from: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    config_dict = checkpoint.get("config", {})
    
    # Recreate model architecture
    config = Figure5ExperimentConfig()
    
    # Update config with saved values if available
    for key, value in config_dict.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Assume standard dimensions for now (should be saved in checkpoint)
    state_dim = 600  # 10x10 grid * 6 channels
    
    model = create_tomnet(
        state_dim=state_dim,
        char_embedding_dim=config.char_embedding_dim,
        use_mental_state_net=config.use_mental_state_net,
        predictions=config.predictions,
        dropout_rate=config.dropout_rate,
    )
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Validation accuracy: {checkpoint.get('val_accuracy', 'unknown'):.4f}")
    
    return model, config


def evaluate_model_on_data(
    model: ToMnet, 
    dataloader: torch.utils.data.DataLoader, 
    device: str = "cuda",
    extract_embeddings: bool = True,
) -> Dict:
    """Evaluate model on dataset and extract embeddings"""
    model.eval()
    
    results = {
        "action_accuracies": [],
        "action_likelihoods": [],
        "n_past_values": [],
        "agent_ids": [],
        "preferred_objects": [],
        "character_embeddings": [],
        "step_indices": [],
    }
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move to device
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            
            # Forward pass
            predictions = model(
                batch["past_trajectories"],
                batch["current_trajectory"],
                batch["current_state"],
            )
            
            # Extract character embeddings if requested
            if extract_embeddings and hasattr(model, 'character_net'):
                # Get character embeddings from the model
                char_embeddings = model.character_net(
                    batch["past_trajectories"]
                ).cpu().numpy()
                results["character_embeddings"].extend(char_embeddings)
            
            # Action predictions
            action_probs = F.softmax(predictions["action"], dim=-1)
            predicted_actions = torch.argmax(action_probs, dim=-1)
            true_actions = batch["true_action"]
            
            # Calculate accuracies
            batch_accuracies = (predicted_actions == true_actions).float().cpu().numpy()
            results["action_accuracies"].extend(batch_accuracies)
            
            # Calculate action likelihoods (probability of true action)
            action_likelihoods = action_probs.gather(1, true_actions.unsqueeze(1)).squeeze(1).cpu().numpy()
            results["action_likelihoods"].extend(action_likelihoods)
            
            # Store metadata
            results["n_past_values"].extend(batch["n_past"])
            results["agent_ids"].extend(batch["agent_id"])
            results["step_indices"].extend(batch.get("step_idx", [0] * len(batch["agent_id"])))
            
            # Extract preferred objects from agent rewards
            if "rewards" in batch:
                rewards = batch["rewards"].cpu().numpy()
                preferred_objects = np.argmax(rewards, axis=1)
                results["preferred_objects"].extend(preferred_objects)
    
    return results


def compute_figure5b_data(
    evaluation_results: Dict,
    n_past_values: List[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
) -> Dict:
    """
    Compute data for Figure 5b: N_past vs average posterior probability assigned to true action
    """
    n_past_data = np.array(evaluation_results["n_past_values"])
    action_likelihoods = np.array(evaluation_results["action_likelihoods"])
    
    figure5b_data = {
        "n_past_values": [],
        "avg_action_likelihoods": [],
        "std_action_likelihoods": [],
        "n_samples": [],
    }
    
    for n_past in n_past_values:
        mask = n_past_data == n_past
        if np.any(mask):
            likelihoods = action_likelihoods[mask]
            
            figure5b_data["n_past_values"].append(n_past)
            figure5b_data["avg_action_likelihoods"].append(np.mean(likelihoods))
            figure5b_data["std_action_likelihoods"].append(np.std(likelihoods))
            figure5b_data["n_samples"].append(len(likelihoods))
        else:
            # No data for this n_past value
            figure5b_data["n_past_values"].append(n_past)
            figure5b_data["avg_action_likelihoods"].append(0.0)
            figure5b_data["std_action_likelihoods"].append(0.0)
            figure5b_data["n_samples"].append(0)
    
    return figure5b_data


def compute_figure5d_data(
    evaluation_results: Dict,
    n_past_filter: int = 0
) -> Dict:
    """
    Compute data for Figure 5d: 2D embedding space of ToMnet for preferred objects with N_past=0
    """
    n_past_data = np.array(evaluation_results["n_past_values"])
    character_embeddings = np.array(evaluation_results["character_embeddings"])
    preferred_objects = np.array(evaluation_results["preferred_objects"])
    agent_ids = np.array(evaluation_results["agent_ids"])
    
    # Filter for N_past=0
    mask = n_past_data == n_past_filter
    
    if not np.any(mask):
        print(f"Warning: No data found for N_past={n_past_filter}")
        return {"embeddings_2d": np.array([]), "preferred_objects": np.array([]), "agent_ids": np.array([])}
    
    filtered_embeddings = character_embeddings[mask]
    filtered_preferred_objects = preferred_objects[mask]
    filtered_agent_ids = agent_ids[mask]
    
    # Reduce to 2D using PCA if needed
    if filtered_embeddings.shape[1] > 2:
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(filtered_embeddings)
        print(f"Reduced embeddings from {filtered_embeddings.shape[1]}D to 2D using PCA")
        print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    else:
        embeddings_2d = filtered_embeddings
    
    # Normalize embeddings
    embeddings_2d = (embeddings_2d - embeddings_2d.mean(axis=0)) / embeddings_2d.std(axis=0)
    
    figure5d_data = {
        "embeddings_2d": embeddings_2d,
        "preferred_objects": filtered_preferred_objects,
        "agent_ids": filtered_agent_ids,
        "n_past_filter": n_past_filter,
        "n_samples": len(embeddings_2d),
    }
    
    return figure5d_data


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description="Evaluate ToMnet for Figure 5")
    
    # Model parameters
    parser.add_argument("--model_paths", nargs="+", help="Paths to trained model files")
    parser.add_argument("--model_dir", type=str, default="models/figure5", help="Directory containing models")
    parser.add_argument("--model_pattern", type=str, default="*_model.pth", help="Pattern to match model files")
    
    # Data parameters
    parser.add_argument("--data_dir", type=str, default="data/figure5", help="Data directory")
    parser.add_argument("--experiment_name", type=str, default="figure5_goal_directed", help="Experiment name")
    
    # Evaluation parameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--n_past_values", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], help="N_past values to analyze")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="result/figure5", help="Output directory")
    parser.add_argument("--output_file", type=str, default="figure5_results.pkl", help="Output file name")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Figure 5 ToMnet Evaluation")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Data directory: {args.data_dir}")
    print(f"Model directory: {args.model_dir}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)
    
    # Set device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"
    
    # Find model files
    if args.model_paths:
        model_paths = args.model_paths
    else:
        model_pattern = os.path.join(args.model_dir, args.model_pattern)
        model_paths = glob.glob(model_pattern)
        if not model_paths:
            raise ValueError(f"No model files found matching pattern: {model_pattern}")
    
    print(f"Found {len(model_paths)} model files:")
    for path in model_paths:
        print(f"  - {path}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load evaluation data
    data_path = os.path.join(args.data_dir, f"{args.experiment_name}_training_data.pkl")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    print(f"Loading evaluation data from: {data_path}")
    with open(data_path, "rb") as f:
        dataset = pickle.load(f)
    
    # Create dataset and dataloader
    tomnet_dataset = ToMnetDataset(dataset, experiment_type="figure5")
    dataloader = torch.utils.data.DataLoader(
        tomnet_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    print(f"Loaded {len(tomnet_dataset)} evaluation samples")
    
    # Evaluate each model
    all_results = {}
    
    for model_path in model_paths:
        model_name = os.path.basename(model_path).replace(".pth", "")
        print(f"\nEvaluating model: {model_name}")
        
        # Load model
        model, config = load_model(model_path, args.device)
        
        # Evaluate model
        evaluation_results = evaluate_model_on_data(
            model, dataloader, args.device, extract_embeddings=True
        )
        
        # Compute Figure 5b data
        figure5b_data = compute_figure5b_data(evaluation_results, args.n_past_values)
        
        # Compute Figure 5d data
        figure5d_data = compute_figure5d_data(evaluation_results, n_past_filter=0)
        
        # Store results
        all_results[model_name] = {
            "evaluation_results": evaluation_results,
            "figure5b": figure5b_data,
            "figure5d": figure5d_data,
            "model_path": model_path,
            "config": config.__dict__ if hasattr(config, '__dict__') else config,
        }
        
        # Print summary
        print(f"  Action accuracy: {np.mean(evaluation_results['action_accuracies']):.4f}")
        print(f"  Average action likelihood: {np.mean(evaluation_results['action_likelihoods']):.4f}")
        print(f"  Character embeddings: {len(evaluation_results['character_embeddings'])} samples")
        print(f"  Figure 5b data points: {len(figure5b_data['n_past_values'])}")
        print(f"  Figure 5d data points: {figure5d_data['n_samples']}")
    
    # Save results
    output_path = os.path.join(args.output_dir, args.output_file)
    with open(output_path, "wb") as f:
        pickle.dump(all_results, f)
    
    print(f"\nEvaluation completed!")
    print(f"Results saved to: {output_path}")
    print(f"Models evaluated: {len(all_results)}")
    
    # Save summary
    summary_path = os.path.join(args.output_dir, "figure5_summary.json")
    summary = {
        "n_models": len(all_results),
        "model_names": list(all_results.keys()),
        "n_samples": len(tomnet_dataset),
        "n_past_values": args.n_past_values,
        "data_path": data_path,
        "output_path": output_path,
    }
    
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()