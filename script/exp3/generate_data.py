import os
import sys
import numpy as np
import gym
import gym_minigrid
from gym_minigrid.minigrid import OBJECT_TO_IDX, COLOR_TO_IDX
import pickle
import argparse
from datetime import datetime
import multiprocessing as mp
from functools import partial

# Add the current directory to path to import our agent
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from astar_agent import MiniGridAStarAgent

"""
Data generation script for MiniGrid-LockedRoom-v0 environment with A* agent
Adapted from ToMnetF_impl experiment5 structure
"""


def run_single_episode(episode_id, config_dict, save_dir):
    """
    Run a single episode with A* agent in MiniGrid-LockedRoom-v0

    Args:
        episode_id: Unique identifier for this episode
        config_dict: Dictionary containing configuration parameters
        save_dir: Directory to save episode data

    Returns:
        episode_id: For tracking completion
    """
    # Set unique random seed for this episode
    np.random.seed(config_dict["base_random_seed"] + episode_id)

    # Create environment
    env = gym.make("MiniGrid-LockedRoom-v0", size=config_dict["size"])
    env.seed(config_dict["base_random_seed"] + episode_id)

    # Create agent
    agent = MiniGridAStarAgent(
        env, max_exploration_steps=config_dict["max_steps"], debug=config_dict["debug"]
    )

    # Reset environment and agent
    obs = env.reset()
    agent.reset()

    # Run episode
    done = False
    step_count = 0
    total_reward = 0

    while not done and step_count < config_dict["max_steps"]:
        # Agent chooses action
        action = agent.choose_action(obs)

        # Execute action
        obs, reward, done, info = env.step(action)

        # Track rewards
        agent.rewards.append(reward)
        total_reward += reward

        step_count += 1

        if config_dict["debug"] and step_count % 10 == 0:
            print(f"Episode {episode_id}, Step {step_count}, Reward: {total_reward}")

    # Get trajectory data
    trajectory_data = agent.get_trajectory_data()
    trajectory_data["total_reward"] = total_reward
    trajectory_data["episode_id"] = episode_id
    trajectory_data["success"] = done and total_reward > 0

    # Save episode data
    save_episode_data(trajectory_data, save_dir, episode_id)

    return episode_id


def save_episode_data(trajectory_data, save_dir, episode_id):
    """
    Save episode data in format compatible with ToMnetF data processing
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Save trajectory data as pickle file
    filepath = os.path.join(save_dir, f"episode_{episode_id}.pkl")
    with open(filepath, "wb") as f:
        pickle.dump(trajectory_data, f)

    # Also save in text format similar to ToMnetF experiment5
    save_trajectory_text(trajectory_data, save_dir, episode_id)


def save_trajectory_text(trajectory_data, save_dir, episode_id):
    """
    Save trajectory in text format similar to ToMnetF experiment5 format
    """
    filepath = os.path.join(save_dir, f"test{episode_id}.txt")

    with open(filepath, "w") as f:
        f.write("MiniGrid-LockedRoom-v0 Episode Data\n")
        f.write("=" * 40 + "\n")

        # Episode metadata
        f.write(f"Episode ID: {episode_id}\n")
        f.write(f"Total Steps: {trajectory_data['step_count']}\n")
        f.write(f"Total Reward: {trajectory_data['total_reward']}\n")
        f.write(f"Success: {trajectory_data['success']}\n")
        f.write(f"Trajectory Length: {len(trajectory_data['trajectory'])}\n")
        f.write("\n")

        # Trajectory data
        f.write("Trajectory Data:\n")
        f.write("Step | Position | Action | Reward\n")
        f.write("-" * 35 + "\n")

        for i, (pos, action, reward) in enumerate(
            zip(
                trajectory_data["position_trajectory"],
                trajectory_data["trajectory"],
                trajectory_data["rewards"],
            )
        ):
            f.write(f"{i:4d} | {pos} | {action:6d} | {reward:6.2f}\n")


def generate_dataset(
    n_episodes=1000,
    size=19,
    max_steps=1000,
    save_dir="./data/exp3",
    random_seed=42,
    n_processes=None,
    debug=False,
):
    """
    Generate dataset of MiniGrid-LockedRoom-v0 episodes with A* agent

    Args:
        n_episodes: Number of episodes to generate
        size: Size of the MiniGrid environment
        max_steps: Maximum steps per episode
        save_dir: Directory to save generated data
        random_seed: Base random seed
        n_processes: Number of parallel processes (default: CPU count - 1)
        debug: Whether to print debug information
    """
    print(f"Generating {n_episodes} episodes of MiniGrid-LockedRoom-v0 with A* agent")
    print(f"Environment size: {size}x{size}")
    print(f"Max steps per episode: {max_steps}")
    print(f"Save directory: {save_dir}")
    print(f"Random seed: {random_seed}")

    # Create output directory
    os.makedirs(save_dir, exist_ok=True)

    # Set number of processes
    if n_processes is None:
        n_processes = max(1, mp.cpu_count() - 1)

    print(f"Using {n_processes} processes for parallel generation")

    # Configuration dictionary
    config_dict = {
        "size": size,
        "max_steps": max_steps,
        "base_random_seed": random_seed,
        "debug": debug,
    }

    # Create partial function for multiprocessing
    episode_func = partial(
        run_single_episode, config_dict=config_dict, save_dir=save_dir
    )

    # Run episodes in parallel
    with mp.Pool(processes=n_processes) as pool:
        episode_ids = range(n_episodes)
        results = []

        for i, result in enumerate(pool.imap(episode_func, episode_ids)):
            results.append(result)
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{n_episodes} episodes")

        # Wait for all processes to complete
        pool.close()
        pool.join()

    print(f"Successfully generated {n_episodes} episodes!")

    # Save dataset metadata
    save_dataset_metadata(save_dir, n_episodes, config_dict)

    return save_dir


def save_dataset_metadata(save_dir, n_episodes, config_dict):
    """
    Save dataset metadata
    """
    metadata = {
        "n_episodes": n_episodes,
        "environment": "MiniGrid-LockedRoom-v0",
        "agent": "A*",
        "generation_time": datetime.now().isoformat(),
        "config": config_dict,
    }

    filepath = os.path.join(save_dir, "dataset_metadata.pkl")
    with open(filepath, "wb") as f:
        pickle.dump(metadata, f)

    # Also save as text
    filepath_txt = os.path.join(save_dir, "dataset_metadata.txt")
    with open(filepath_txt, "w") as f:
        f.write("Dataset Metadata\n")
        f.write("=" * 20 + "\n")
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")


def load_dataset(save_dir):
    """
    Load generated dataset

    Args:
        save_dir: Directory containing the dataset

    Returns:
        List of trajectory data dictionaries
    """
    # Load metadata
    metadata_path = os.path.join(save_dir, "dataset_metadata.pkl")
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    # Load all episodes
    trajectories = []
    for episode_id in range(metadata["n_episodes"]):
        episode_path = os.path.join(save_dir, f"episode_{episode_id}.pkl")
        if os.path.exists(episode_path):
            with open(episode_path, "rb") as f:
                trajectory = pickle.load(f)
                trajectories.append(trajectory)

    return trajectories, metadata


def analyze_dataset(save_dir):
    """
    Analyze the generated dataset
    """
    trajectories, metadata = load_dataset(save_dir)

    print(f"\nDataset Analysis")
    print(f"=" * 20)
    print(f"Number of episodes: {len(trajectories)}")

    # Success rate
    successes = sum(1 for traj in trajectories if traj["success"])
    success_rate = successes / len(trajectories) * 100
    print(f"Success rate: {success_rate:.1f}% ({successes}/{len(trajectories)})")

    # Trajectory length statistics
    lengths = [traj["step_count"] for traj in trajectories]
    print(
        f"Trajectory lengths - Mean: {np.mean(lengths):.1f}, Std: {np.std(lengths):.1f}"
    )
    print(f"                   - Min: {np.min(lengths)}, Max: {np.max(lengths)}")

    # Reward statistics
    rewards = [traj["total_reward"] for traj in trajectories]
    print(f"Total rewards - Mean: {np.mean(rewards):.3f}, Std: {np.std(rewards):.3f}")
    print(f"              - Min: {np.min(rewards):.3f}, Max: {np.max(rewards):.3f}")


if __name__ == "__main__":
    # Set multiprocessing start method
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass  # Already set

    parser = argparse.ArgumentParser(
        description="Generate MiniGrid-LockedRoom-v0 dataset with A* agent"
    )
    parser.add_argument(
        "--n_episodes", type=int, default=1000, help="Number of episodes to generate"
    )
    parser.add_argument(
        "--size", type=int, default=19, help="Size of the MiniGrid environment"
    )
    parser.add_argument(
        "--max_steps", type=int, default=1000, help="Maximum steps per episode"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./data/exp3",
        help="Directory to save generated data",
    )
    parser.add_argument("--random_seed", type=int, default=42, help="Base random seed")
    parser.add_argument(
        "--n_processes", type=int, default=None, help="Number of parallel processes"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze existing dataset instead of generating new one",
    )

    args = parser.parse_args()

    if args.analyze:
        analyze_dataset(args.save_dir)
    else:
        generate_dataset(
            n_episodes=args.n_episodes,
            size=args.size,
            max_steps=args.max_steps,
            save_dir=args.save_dir,
            random_seed=args.random_seed,
            n_processes=args.n_processes,
            debug=args.debug,
        )

        # Run analysis after generation
        analyze_dataset(args.save_dir)
