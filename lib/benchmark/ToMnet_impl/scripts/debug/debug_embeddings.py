#!/usr/bin/env python3
"""Debug script to analyze character embeddings for different N_past values"""

import torch
import pickle
import numpy as np
from scripts.evaluate import load_model_from_checkpoint
from scripts.data_generation import ToMnetDataset, collate_fn
from scripts.evaluate import ToMnetEvaluator

def debug_character_embeddings():
    """Analyze character embeddings for different N_past values"""
    
    # Load trained model
    model_path = "models/figure3_0.01_best.pth"
    state_dim = 11 * 11 * 6
    device = "cpu"
    
    try:
        model = load_model_from_checkpoint(model_path, "figure3", state_dim, device)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Load test dataset
    dataset_path = "data/figure3_alpha_0.01.pkl"
    try:
        with open(dataset_path, 'rb') as f:
            dataset_raw = pickle.load(f)
        dataset = ToMnetDataset(dataset_raw, experiment_type="figure3")
        print("✅ Dataset loaded successfully")
        print(f"Dataset size: {len(dataset)}")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return
    
    # Analyze embeddings for different N_past values
    n_past_values = [0, 1, 5]
    
    for n_past in n_past_values:
        print(f"\n=== Analyzing N_past = {n_past} ===")
        
        # Filter samples with specific N_past
        filtered_samples = []
        for i, sample in enumerate(dataset.data):
            if sample["n_past"] == n_past:
                filtered_samples.append(sample)
                if len(filtered_samples) >= 10:  # Take first 10 samples
                    break
        
        if not filtered_samples:
            print(f"❌ No samples found with N_past = {n_past}")
            continue
            
        print(f"Found {len(filtered_samples)} samples")
        
        # Create mini-dataset
        mini_dataset = ToMnetDataset(
            {"data": filtered_samples, "meta": dataset_raw.get("meta", {})},
            experiment_type="figure3"
        )
        
        # Get first batch
        batch_data = []
        for i in range(min(5, len(mini_dataset))):
            batch_data.append(mini_dataset[i])
        
        batch = collate_fn(batch_data)
        
        # Move to device
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        
        # Extract character embeddings
        model.eval()
        with torch.no_grad():
            # Get character embeddings directly from CharacterNet
            character_embeddings = model.character_net(batch["past_trajectories"])
            
            print(f"Batch size: {character_embeddings.shape[0]}")
            print(f"Character embedding shape: {character_embeddings.shape}")
            print(f"Character embeddings:")
            for i, emb in enumerate(character_embeddings):
                print(f"  Sample {i}: [{emb[0]:.4f}, {emb[1]:.4f}]")
            
            # Check if embeddings are all zeros (indicating no learning)
            all_zeros = torch.allclose(character_embeddings, torch.zeros_like(character_embeddings), atol=1e-6)
            print(f"All embeddings zero: {all_zeros}")
            
            # Check embedding statistics
            emb_mean = character_embeddings.mean(dim=0)
            emb_std = character_embeddings.std(dim=0)
            print(f"Embedding mean: [{emb_mean[0]:.4f}, {emb_mean[1]:.4f}]")
            print(f"Embedding std:  [{emb_std[0]:.4f}, {emb_std[1]:.4f}]")
            
            # Get full model predictions to see action probabilities
            predictions = model(
                batch["past_trajectories"],
                batch["current_trajectory"], 
                batch["current_state"]
            )
            
            action_probs = predictions["action_probs"]
            print(f"Action probabilities shape: {action_probs.shape}")
            print(f"Sample action probs:")
            for i, probs in enumerate(action_probs[:3]):  # Show first 3
                print(f"  Sample {i}: [{', '.join([f'{p:.3f}' for p in probs])}]")
            
            # Check true actions for comparison
            true_actions = batch["true_actions"]
            print(f"True actions: {true_actions[:3].tolist()}")
            
            # Calculate action likelihoods
            action_likelihoods = action_probs.gather(1, true_actions.unsqueeze(1)).squeeze(1)
            print(f"Action likelihoods: {action_likelihoods[:3].tolist()}")
            mean_likelihood = action_likelihoods.mean().item()
            print(f"Mean action likelihood: {mean_likelihood:.4f}")

if __name__ == "__main__":
    debug_character_embeddings()