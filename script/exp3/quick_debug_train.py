#!/usr/bin/env python3
"""
Quick training script for debugging evaluation
"""
import os
import sys
import torch

sys.path.append(os.path.dirname(__file__))

from config import Config
from train import train_keydoor_tomnet


def quick_train_for_debug():
    """Train a small model quickly for debugging"""
    
    # Override config for minimal training
    config = Config()
    
    # Use much smaller settings for quick training
    config.training_config["num_epochs"] = 2
    config.training_config["early_stopping_patience"] = 1
    config.n_games = 50  # Very few games
    config.validation_games = 20
    
    # Use existing training data if available
    train_data_dir = "data/exp3"
    if not os.path.exists(train_data_dir):
        print("No training data found. Please run data generation first.")
        return
    
    # Set up directories
    debug_results_dir = "results/exp3_debug_quick"
    os.makedirs(debug_results_dir, exist_ok=True)
    
    # Train with minimal settings
    print("Running quick training for debugging...")
    print(f"Training with {config.n_games} games, {config.training_config['num_epochs']} epochs")
    
    try:
        train_keydoor_tomnet(
            config=config,
            train_data_dir=train_data_dir,
            results_dir=debug_results_dir,
        )
        print(f"Model saved to: {debug_results_dir}")
        return debug_results_dir
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    quick_train_for_debug()