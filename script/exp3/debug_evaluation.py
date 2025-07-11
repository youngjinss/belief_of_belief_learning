#!/usr/bin/env python3
"""
Debug script to test evaluation accuracy fixes
"""
import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.dirname(__file__))

from config import Config
from evaluate import load_model, evaluate_model
from train import prepare_data_for_training, generate_past_episodes_from_batch
from data_generation import DataReader


def debug_evaluation():
    """Debug the evaluation process step by step"""
    
    # Setup
    config = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Find model and test data
    results_dir = "results/exp3"
    model_path = None
    
    # Search for best_model.pth
    for root, dirs, files in os.walk(results_dir):
        if "best_model.pth" in files:
            model_path = os.path.join(root, "best_model.pth")
            break
    
    if not model_path:
        print("No model found! Please train a model first.")
        return
    
    print(f"Found model: {model_path}")
    
    # Load model
    model_kwargs = config.get_model_kwargs()
    model = load_model(model_path, device, model_kwargs)
    print(f"Model loaded successfully")
    print(f"Model config: use_mentalnet={model_kwargs.get('use_mentalnet', 'Not set')}")
    
    # Load test data
    test_data_dir = "data/exp3/test"
    if not os.path.exists(test_data_dir):
        print(f"Test data directory not found: {test_data_dir}")
        return
    
    print(f"\nLoading test data from: {test_data_dir}")
    data_config = config.get_data_config()
    
    # Create DataReader with correct dimensions
    data_reader = DataReader(
        time_step=data_config.get("time_step", 500),
        w=config.width,
        h=config.height,
        d=data_config.get("maze_depth", 9),
        experiment_no=config.experiment_no
    )
    
    test_games = data_reader.ReadAllGames(test_data_dir)
    print(f"Loaded {len(test_games)} test games")
    
    if len(test_games) == 0:
        print("No test games found!")
        return
    
    # Prepare test data
    print("\nPreparing test data...")
    test_data = prepare_data_for_training(
        test_games,
        min_timestep=6,
        max_trajectory_length=data_config["max_moves"],
    )
    
    print("\nTest data shapes:")
    for key, tensor in test_data.items():
        print(f"  {key}: {tensor.shape}")
    
    # Create test dataset with all 7 tensors
    test_dataset = TensorDataset(
        test_data["trajectories"],
        test_data["actions"],
        test_data["goals"],
        test_data["goal_ranks"],
        test_data["goal_rewards"],
        test_data["consumption_labels"],
        test_data["sr_labels"],
    )
    
    print(f"\nTest dataset size: {len(test_dataset)} samples")
    
    # Create a small batch for debugging
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    # Debug a single batch
    print("\n" + "="*60)
    print("DEBUGGING SINGLE BATCH")
    print("="*60)
    
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx > 0:  # Only process first batch
                break
            
            print(f"\nBatch {batch_idx}:")
            print(f"Number of tensors in batch: {len(batch)}")
            
            # Unpack all data
            (
                trajectories,
                actions,
                goals,
                goal_ranks,
                goal_rewards,
                consumption_labels,
                sr_labels,
            ) = batch
            
            print(f"\nBatch shapes:")
            print(f"  trajectories: {trajectories.shape}")
            print(f"  actions: {actions.shape}")
            print(f"  goals: {goals.shape}")
            print(f"  goal_ranks: {goal_ranks.shape}")
            
            # Move to device
            trajectories = trajectories.to(device)
            actions = actions.to(device)
            goals = goals.to(device)
            goal_ranks = goal_ranks.to(device)
            
            batch_size = trajectories.size(0)
            
            print(f"\nGenerating past episodes...")
            print(f"  Using goal_ranks: {goal_ranks}")
            print(f"  rank_threshold: {data_config.get('rank_threshold', 1)}")
            
            # Generate past episodes using goal_ranks
            past_episodes = generate_past_episodes_from_batch(
                trajectories,
                goal_ranks,  # Using goal_ranks instead of goals
                batch_size,
                n_past_min=data_config.get("n_past_min", 1),
                n_past_max=data_config.get("n_past_max", 1),
                max_n_past=data_config.get("max_n_past", 1),
                rank_threshold=data_config.get("rank_threshold", 1),
            )
            
            print(f"  past_episodes shape: {past_episodes.shape}")
            
            # Get action targets
            action_targets = actions[:, 0]
            print(f"\nAction targets: {action_targets}")
            
            # Find effective lengths
            traj_sums = trajectories.sum(dim=(2, 3, 4))
            non_zero_mask = traj_sums > 0
            seq_indices = torch.arange(trajectories.size(1), device=device).unsqueeze(0).expand(batch_size, -1)
            masked_indices = torch.where(non_zero_mask, seq_indices, torch.tensor(-1, device=device))
            effective_lengths = masked_indices.max(dim=1)[0].clamp(min=0).tolist()
            
            print(f"Effective lengths: {effective_lengths}")
            
            # Extract current state
            current_state_channels = data_config.get("current_state_channels", 8)
            recent_trajectory = trajectories[:, :, :current_state_channels]
            
            batch_indices = torch.arange(batch_size, device=device)
            last_timesteps = torch.tensor([max(0, length - 1) for length in effective_lengths], device=device)
            current_state = trajectories[batch_indices, last_timesteps, :current_state_channels]
            
            print(f"\nModel inputs:")
            print(f"  past_episodes: {past_episodes.shape}")
            print(f"  recent_trajectory: {recent_trajectory.shape}")
            print(f"  current_state: {current_state.shape}")
            
            # Model forward pass
            try:
                (
                    action_logits,
                    goal_logits,
                    consumption_logits,
                    sr_pred,
                    char_emb,
                    mental_state,
                ) = model(past_episodes, recent_trajectory, current_state)
                
                print(f"\nModel outputs:")
                print(f"  action_logits: {action_logits.shape}")
                
                # Get predictions
                _, predicted = torch.max(action_logits, 1)
                print(f"\nPredictions: {predicted}")
                print(f"Targets: {action_targets}")
                print(f"Correct: {(predicted == action_targets).sum().item()}/{batch_size}")
                
            except Exception as e:
                print(f"\nERROR in forward pass: {e}")
                import traceback
                traceback.print_exc()
                return
    
    # Now run full evaluation
    print("\n" + "="*60)
    print("RUNNING FULL EVALUATION")
    print("="*60)
    
    # Create full test loader
    eval_config = config.get_evaluation_config()
    test_loader_full = DataLoader(
        test_dataset, 
        batch_size=eval_config["batch_size"], 
        shuffle=False
    )
    
    # Run evaluation
    metrics = evaluate_model(
        model,
        test_loader_full,
        device,
        data_config=data_config,
        save_predictions=False,
        output_dir=None,
    )
    
    print(f"\nEvaluation Results:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1 Score: {metrics['f1_score']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    
    print(f"\nPer-action accuracy:")
    for action, acc in metrics['action_accuracy'].items():
        print(f"  {action}: {acc:.4f}")
    
    # Compare with training validation accuracy
    print("\n" + "="*60)
    print("COMPARISON WITH TRAINING")
    print("="*60)
    print("Expected validation accuracy (from training): ~52.8%")
    print(f"Actual evaluation accuracy: {metrics['accuracy']*100:.1f}%")
    
    if metrics['accuracy'] < 0.45:  # If still below 45%
        print("\nWARNING: Evaluation accuracy is still much lower than expected!")
        print("Possible issues:")
        print("- Data preprocessing still has differences")
        print("- Model architecture mismatch")
        print("- Test data distribution is different")
    else:
        print("\nSUCCESS: Evaluation accuracy is close to training validation!")


if __name__ == "__main__":
    debug_evaluation()