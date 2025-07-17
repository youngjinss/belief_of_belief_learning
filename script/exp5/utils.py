"""
Common utility functions for exp5 scripts
Consolidates duplicated functions across train.py, evaluate.py, visualize.py
"""

import os
import pickle
import random
import numpy as np
import torch
import sys

# Add path for imports
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# All imports will be done dynamically to avoid circular imports


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility across all major libraries.

    Args:
        seed (int): Random seed value
    """
    # Python random module
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    # CUDA convolution determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Environment variables for additional reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)

    # For DataLoader workers
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return seed_worker


def load_data_mmap(filepath):
    """Load data using memory mapping for faster access"""
    data = np.load(filepath, mmap_mode="r")
    print(f"Loaded memory-mapped data from {filepath}")
    return data


def load_data_efficient(filepath):
    """Load data efficiently from pickle or chunk metadata"""
    # Load pickle file (now contains chunk metadata)
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        return data

    return None



def load_training_data_all_combinations(config, data_dir_base=None):
    """
    Load training data for all combinations following train.py pattern

    Args:
        config: Config object
        data_dir_base: Base directory for training data

    Returns:
        dict: Dictionary with (achiever_type, blocker_type) -> training_data mapping
    """
    if data_dir_base is None:
        env_name = config.get_env_name()
        data_dir_base = f"./data/{env_name}"

    # Get all combinations
    all_combinations = []
    for achiever_type in config.achiever_types.keys():
        for blocker_type in config.blocker_types.keys():
            all_combinations.append((achiever_type, blocker_type))

    existing_data = {}
    missing_combinations = []

    for combo_achiever, combo_blocker in all_combinations:
        # Construct training data directory path
        agent_pair = config.get_agent_pair_name(combo_achiever, combo_blocker)
        train_data_dir = os.path.join(data_dir_base, agent_pair)

        processed_data_path = os.path.join(
            train_data_dir,
            f"processed_data_exp{config.experiment_no}_{combo_achiever}_{combo_blocker}.pkl",
        )

        # Try efficient data loading
        data = load_data_efficient(processed_data_path)
        if data is not None:
            print(
                f"Loading existing processed training data for {combo_achiever}_{combo_blocker}..."
            )
            existing_data[(combo_achiever, combo_blocker)] = data
            print(f"  Successfully loaded from {processed_data_path}")
        else:
            missing_combinations.append((combo_achiever, combo_blocker))

    # Generate missing data if needed
    if missing_combinations:
        print(
            f"Processed training data not found for combinations: {missing_combinations}"
        )
        print("Generating training data for missing combinations...")

        for combo_achiever, combo_blocker in missing_combinations:
            # Construct training data directory path
            agent_pair = config.get_agent_pair_name(combo_achiever, combo_blocker)
            train_data_dir = os.path.join(data_dir_base, agent_pair)

            processed_data_path = os.path.join(
                train_data_dir,
                f"processed_data_exp{config.experiment_no}_{combo_achiever}_{combo_blocker}.pkl",
            )


            if not os.path.exists(train_data_dir):
                print(f"Training data directory not found: {train_data_dir}")
                print(f"Skipping combination {combo_achiever}_{combo_blocker}")
                continue

            # Create chunk directory for this combination
            chunk_dir = os.path.join(train_data_dir, f"chunks_{combo_achiever}_{combo_blocker}")
            
            # Check if processed data already exists
            if os.path.exists(processed_data_path):
                print(f"Loading existing processed data from: {processed_data_path}")
                train_data = load_data_efficient(processed_data_path)
                if train_data is not None:
                    # Check if this is chunk metadata (new format) or old format
                    if isinstance(train_data, dict) and "chunk_metadata" in train_data:
                        print(f"  Found chunk metadata with {train_data['num_chunks']} chunks")
                        # Verify chunks still exist
                        chunks_exist = all(os.path.exists(chunk['file_path']) for chunk in train_data['chunk_metadata'])
                        if chunks_exist:
                            print(f"  All chunks verified in {chunk_dir}")
                            existing_data[(combo_achiever, combo_blocker)] = train_data
                            continue
                        else:
                            print(f"  Some chunks are missing, regenerating...")
                    else:
                        print(f"  Found old format data, converting to chunked format...")
                        # Remove old pkl file to force regeneration
                        os.remove(processed_data_path)

            # Load and process raw training data
            from data_generation import DataGenerator as DataReader

            data_config = config.get_data_config()
            data_reader = DataReader(
                time_step=data_config.get("time_step", 500),
                w=config.width,
                h=config.height,
                d=data_config.get("maze_depth", 9),
                config=config,
            )

            train_games = data_reader.ReadAllGames(train_data_dir)
            if len(train_games) == 0:
                print(f"No training games found in {train_data_dir}")
                continue

            # Process training data using chunked approach
            from train import prepare_data_for_training
            
            train_data = prepare_data_for_training(
                train_games,
                min_timestep=data_config.get("min_time_steps", 6),
                max_trajectory_length=data_config.get("time_step", 500),
                chunk_size=data_config.get("chunk_size", 5000),  # Process in smaller chunks to avoid memory issues
                output_dir=chunk_dir,
            )

            # Save chunk metadata instead of full data
            print(f"Saving chunk metadata to: {processed_data_path}")
            with open(processed_data_path, "wb") as f:
                pickle.dump(train_data, f)
            print(f"  Successfully saved metadata to {processed_data_path}")
            print(f"  Data chunks saved in {chunk_dir}")

            existing_data[(combo_achiever, combo_blocker)] = train_data

    return existing_data


def load_test_data_all_combinations(config, test_data_dir_base=None):
    """
    Load test data for all combinations following train.py pattern

    Args:
        config: Config object
        test_data_dir_base: Base directory for test data

    Returns:
        dict: Dictionary with (achiever_type, blocker_type) -> test_data mapping
    """
    if test_data_dir_base is None:
        env_name = config.get_env_name()
        test_data_dir_base = f"./data/{env_name}"

    # Get all combinations
    all_combinations = []
    for achiever_type in config.achiever_types.keys():
        for blocker_type in config.blocker_types.keys():
            all_combinations.append((achiever_type, blocker_type))

    existing_data = {}
    missing_combinations = []

    for combo_achiever, combo_blocker in all_combinations:
        # Construct test data directory path
        agent_pair = config.get_agent_pair_name(combo_achiever, combo_blocker)
        test_data_dir = os.path.join(test_data_dir_base, agent_pair, "test")

        processed_test_data_path = os.path.join(
            test_data_dir,
            f"processed_test_data_exp{config.experiment_no}_{combo_achiever}_{combo_blocker}.pkl",
        )

        # Try efficient data loading
        data = load_data_efficient(processed_test_data_path)
        if data is not None:
            print(
                f"Loading existing processed test data for {combo_achiever}_{combo_blocker}..."
            )
            existing_data[(combo_achiever, combo_blocker)] = data
            print(f"  Successfully loaded from {processed_test_data_path}")
        else:
            missing_combinations.append((combo_achiever, combo_blocker))

    # Generate missing data if needed
    if missing_combinations:
        print(f"Processed test data not found for combinations: {missing_combinations}")
        print("Generating test data for missing combinations...")

        for combo_achiever, combo_blocker in missing_combinations:
            # Construct test data directory path
            agent_pair = config.get_agent_pair_name(combo_achiever, combo_blocker)
            test_data_dir = os.path.join(test_data_dir_base, agent_pair, "test")

            processed_test_data_path = os.path.join(
                test_data_dir,
                f"processed_test_data_exp{config.experiment_no}_{combo_achiever}_{combo_blocker}.pkl",
            )

            if not os.path.exists(test_data_dir):
                print(f"Test data directory not found: {test_data_dir}")
                print(f"Skipping combination {combo_achiever}_{combo_blocker}")
                continue

            # Load and process raw test data
            from data_generation import DataGenerator as DataReader

            data_config = config.get_data_config()
            data_reader = DataReader(
                time_step=data_config.get("time_step", 500),
                w=config.width,
                h=config.height,
                d=data_config.get("maze_depth", 9),
                config=config,
            )

            test_games = data_reader.ReadAllGames(test_data_dir)
            if len(test_games) == 0:
                print(f"No test games found in {test_data_dir}")
                continue

            # Process test data
            from train import prepare_data_for_training

            test_data = prepare_data_for_training(
                test_games,
                min_timestep=data_config.get("min_time_steps", 6),
                max_trajectory_length=data_config.get("time_step", 500),
            )

            # Save processed test data
            print(f"Saving processed test data to: {processed_test_data_path}")
            with open(processed_test_data_path, "wb") as f:
                pickle.dump(test_data, f)
            print(f"  Successfully saved to {processed_test_data_path}")

            existing_data[(combo_achiever, combo_blocker)] = test_data

    return existing_data


def get_data_for_combination(
    all_data, achiever_type, blocker_type, data_type="training"
):
    """
    Get data for a specific combination from loaded data dictionary

    Args:
        all_data: Dictionary with (achiever_type, blocker_type) -> data mapping
        achiever_type: Type of achiever agent
        blocker_type: Type of blocker agent
        data_type: Type of data (for error messages)

    Returns:
        data: Data for the specified combination
    """
    if (achiever_type, blocker_type) in all_data:
        data = all_data[(achiever_type, blocker_type)]
        print(f"Using {data_type} data for {achiever_type}_{blocker_type}")
        return data
    else:
        raise ValueError(
            f"No {data_type} data found for combination {achiever_type}_{blocker_type}. Please generate {data_type} data first."
        )


def load_chunked_data_for_training(chunk_metadata):
    """
    Load all chunks and combine them for training (for backward compatibility)
    
    Args:
        chunk_metadata: Dictionary containing chunk metadata
        
    Returns:
        Combined data dictionary with all tensors
    """
    if "chunk_metadata" not in chunk_metadata:
        # This is old format data, return as-is
        return chunk_metadata
        
    print(f"Loading {chunk_metadata['num_chunks']} chunks for training...")
    
    # Initialize lists to collect data
    all_trajectories = []
    all_actions = []
    all_goals = []
    all_goal_ranks = []
    all_agents = []
    all_types = []
    all_consumption_labels = []
    all_sr_labels = []
    
    # Load and combine all chunks
    for chunk_info in chunk_metadata['chunk_metadata']:
        print(f"Loading chunk {chunk_info['chunk_idx']} from {chunk_info['file_path']}")
        chunk_data = torch.load(chunk_info['file_path'])
        
        all_trajectories.append(chunk_data['trajectories'])
        all_actions.append(chunk_data['actions'])
        all_goals.append(chunk_data['goals'])
        all_goal_ranks.append(chunk_data['goal_ranks'])
        all_agents.append(chunk_data['agents'])
        all_types.append(chunk_data['types'])
        all_consumption_labels.append(chunk_data['consumption_labels'])
        all_sr_labels.append(chunk_data['sr_labels'])
        
        # Free memory
        del chunk_data
    
    # Combine all data
    print("Combining all chunks...")
    combined_data = {
        'trajectories': torch.cat(all_trajectories, dim=0).numpy(),
        'actions': torch.cat(all_actions, dim=0).numpy(),
        'goals': torch.cat(all_goals, dim=0).numpy(),
        'goal_ranks': torch.cat(all_goal_ranks, dim=0).numpy(),
        'agents': torch.cat(all_agents, dim=0).numpy(),
        'types': torch.cat(all_types, dim=0).numpy(),
        'consumption_labels': torch.cat(all_consumption_labels, dim=0).numpy(),
        'sr_labels': torch.cat(all_sr_labels, dim=0).numpy(),
    }
    
    print(f"Combined data shapes:")
    for key, value in combined_data.items():
        print(f"  {key}: {value.shape}")
    
    return combined_data


def validate_data_shape(data, data_type="data"):
    """
    Validate that data has the expected shape and structure

    Args:
        data: Data dictionary to validate
        data_type: Type of data (for error messages)

    Returns:
        bool: True if data is valid
    """
    if not isinstance(data, dict):
        raise ValueError(f"{data_type} must be a dictionary")

    required_keys = [
        "trajectories",
        "actions",
        "goals",
        "goal_ranks",
        "agents",
        "types",
        "consumption_labels",
        "sr_labels",
    ]

    for key in required_keys:
        if key not in data:
            raise ValueError(f"{data_type} missing required key: {key}")

    total_samples = data["trajectories"].shape[0]

    print(f"{data_type} validation:")
    print(f"  Total samples: {total_samples}")

    if total_samples == 0:
        raise ValueError(
            f"No {data_type} found. Please generate {data_type} first using appropriate flags."
        )

    return True


def log_data_shapes(data, data_type="data"):
    """
    Log the shapes of all data arrays for verification

    Args:
        data: Data dictionary
        data_type: Type of data (for logging)
    """
    print(f"{data_type} shapes:")
    for key, value in data.items():
        if hasattr(value, "shape"):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {type(value)}")
