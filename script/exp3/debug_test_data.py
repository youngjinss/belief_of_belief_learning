#!/usr/bin/env python3
"""
Debug test data to see what actions are actually present
"""
import os
import sys
import numpy as np
import torch
from collections import Counter

sys.path.append(os.path.dirname(__file__))

from config import Config
from train import prepare_data_for_training
from data_generation import DataReader


def debug_test_data():
    """Debug the test data to see what actions are present"""
    
    print("="*60)
    print("DEBUGGING TEST DATA ACTION DISTRIBUTION")
    print("="*60)
    
    # Load config
    config = Config()
    
    # Check if test data exists
    test_data_dir = "data/exp3/test"
    if not os.path.exists(test_data_dir):
        print(f"Test data directory not found: {test_data_dir}")
        return
    
    print(f"Loading test data from: {test_data_dir}")
    
    # Load test data
    data_config = config.get_data_config()
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
    
    # Check raw actions in games before processing
    print("\n" + "="*40)
    print("RAW ACTION ANALYSIS FROM GAMES")
    print("="*40)
    
    all_raw_actions = []
    for i, game in enumerate(test_games[:5]):  # Check first 5 games
        actions = game.get("actions", [])
        print(f"Game {i}: {len(actions)} actions")
        print(f"  Actions: {actions[:20]}...")  # First 20 actions
        all_raw_actions.extend(actions)
    
    raw_action_counts = Counter(all_raw_actions)
    print(f"\nRaw action distribution (first 5 games):")
    for action in sorted(raw_action_counts.keys()):
        count = raw_action_counts[action]
        print(f"  Action {action}: {count} occurrences")
    
    # Prepare data for training/evaluation
    print("\n" + "="*40)
    print("PROCESSED DATA ANALYSIS")
    print("="*40)
    
    test_data = prepare_data_for_training(
        test_games,
        min_timestep=6,
        max_trajectory_length=data_config["max_moves"],
    )
    
    print(f"Test data shapes:")
    for key, tensor in test_data.items():
        print(f"  {key}: {tensor.shape}")
    
    # Analyze action distribution in processed data
    actions_tensor = test_data["actions"]
    print(f"\nActions tensor shape: {actions_tensor.shape}")
    print(f"Actions tensor dtype: {actions_tensor.dtype}")
    
    # Check actions at index 0 (target actions for trajectory slicing)
    target_actions = actions_tensor[:, 0]
    print(f"Target actions shape: {target_actions.shape}")
    print(f"Target actions range: {target_actions.min().item()} to {target_actions.max().item()}")
    
    # Count each action
    action_counts = {}
    for action in range(7):  # KeyDoor has 7 actions
        count = (target_actions == action).sum().item()
        action_counts[action] = count
        
    print(f"\nAction distribution in processed data:")
    action_names = ["up", "right", "down", "left", "stay", "pickup", "toggle"]
    total_actions = sum(action_counts.values())
    
    for action in range(7):
        count = action_counts[action]
        pct = (count / total_actions) * 100 if total_actions > 0 else 0
        print(f"  Action {action} ({action_names[action]}): {count} ({pct:.1f}%)")
    
    # Check if any actions are missing
    missing_actions = [action for action in range(7) if action_counts[action] == 0]
    if missing_actions:
        print(f"\n🚨 MISSING ACTIONS: {missing_actions}")
        missing_names = [action_names[a] for a in missing_actions]
        print(f"   Missing action names: {missing_names}")
    else:
        print(f"\n✅ ALL 7 ACTIONS PRESENT")
    
    # Compare with training data if available
    print(f"\n" + "="*40)
    print("COMPARISON WITH TRAINING DATA")
    print("="*40)
    
    train_data_dir = "data/exp3"
    if os.path.exists(train_data_dir):
        # Check a few training games
        train_games = data_reader.ReadAllGames(train_data_dir)
        if len(train_games) > 0:
            train_sample = train_games[:5]  # First 5 games
            
            all_train_actions = []
            for game in train_sample:
                actions = game.get("actions", [])
                all_train_actions.extend(actions)
            
            train_action_counts = Counter(all_train_actions)
            print(f"Training action distribution (first 5 games):")
            for action in sorted(train_action_counts.keys()):
                count = train_action_counts[action]
                print(f"  Action {action}: {count} occurrences")
            
            # Check if training and test have same action distributions
            train_actions_set = set(train_action_counts.keys())
            test_actions_set = set(raw_action_counts.keys())
            
            if train_actions_set != test_actions_set:
                print(f"\n🚨 MISMATCH: Training and test have different action sets!")
                print(f"  Training actions: {sorted(train_actions_set)}")
                print(f"  Test actions: {sorted(test_actions_set)}")
                print(f"  Missing in test: {sorted(train_actions_set - test_actions_set)}")
                print(f"  Extra in test: {sorted(test_actions_set - train_actions_set)}")
            else:
                print(f"\n✅ Training and test have same action sets")
    
    # Check goals vs actions confusion
    print(f"\n" + "="*40)
    print("GOALS VS ACTIONS CHECK")
    print("="*40)
    
    goals_tensor = test_data["goals"]
    print(f"Goals tensor shape: {goals_tensor.shape}")
    print(f"Goals range: {goals_tensor.min().item()} to {goals_tensor.max().item()}")
    
    goal_counts = {}
    for goal in range(4):  # KeyDoor has 4 goals
        count = (goals_tensor == goal).sum().item()
        goal_counts[goal] = count
    
    print(f"Goal distribution:")
    for goal in range(4):
        count = goal_counts[goal]
        pct = (count / len(goals_tensor)) * 100 if len(goals_tensor) > 0 else 0
        print(f"  Goal {goal}: {count} ({pct:.1f}%)")
    
    # Final diagnosis
    print(f"\n" + "="*60)
    print("DIAGNOSIS")
    print("="*60)
    
    if missing_actions:
        print(f"🚨 PROBLEM IDENTIFIED: Missing actions {missing_actions}")
        print(f"The test data doesn't contain all 7 actions that KeyDoor should have.")
        print(f"This explains why the confusion matrix is 4x4 instead of 7x7.")
        print(f"\nPossible causes:")
        print(f"  1. Test data generation didn't include all action types")
        print(f"  2. KeyDoor environment isn't generating pickup/toggle actions")
        print(f"  3. Data filtering removed some action types")
        print(f"  4. Bug in data generation script")
    else:
        print(f"✅ All 7 actions are present in test data")
        print(f"The confusion matrix issue must be in the evaluation code")


if __name__ == "__main__":
    debug_test_data()