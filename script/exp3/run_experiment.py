#!/usr/bin/env python3
"""
Main runner script for exp3 - MiniGrid-LockedRoom-v0 with A* agent
"""

import os
import sys
import argparse
import numpy as np
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from generate_data import generate_dataset, analyze_dataset, load_dataset
from astar_agent import MiniGridAStarAgent


def run_single_test(config):
    """
    Run a single test episode to verify the agent works
    """
    print("Running single test episode...")

    import gym
    import gym_minigrid

    # Create environment
    env = gym.make(config.env_name, size=config.size)
    env.seed(config.random_seed)

    # Create agent
    agent = MiniGridAStarAgent(
        env, max_exploration_steps=config.max_exploration_steps, debug=True
    )

    # Reset environment and agent
    obs = env.reset()
    agent.reset()

    print(f"Initial mission: {obs['mission']}")
    print(f"Initial position: {env.agent_pos}")
    print(f"Initial direction: {env.agent_dir}")

    # Run episode
    done = False
    step_count = 0
    total_reward = 0

    while not done and step_count < config.max_steps:
        # Agent chooses action
        action = agent.choose_action(obs)

        # Execute action
        obs, reward, done, info = env.step(action)

        # Track rewards
        agent.rewards.append(reward)
        total_reward += reward

        step_count += 1

        print(f"Step {step_count}: Action {action}, Reward {reward}, Done {done}")
        print(f"  Position: {env.agent_pos}, Direction: {env.agent_dir}")
        print(f"  Total reward: {total_reward}")

        if done:
            print(f"Episode completed! Success: {total_reward > 0}")
            break

    return total_reward > 0


def run_data_generation(config):
    """
    Run data generation
    """
    print("Starting data generation...")
    print(config)

    # Generate dataset
    save_dir = generate_dataset(
        n_episodes=config.n_episodes,
        size=config.size,
        max_steps=config.max_steps,
        save_dir=config.save_dir,
        random_seed=config.random_seed,
        n_processes=config.n_processes,
        debug=config.debug,
    )

    print(f"\nData generation completed!")
    print(f"Data saved to: {save_dir}")

    # Analyze the generated dataset
    analyze_dataset(save_dir)


def run_evaluation(config):
    """
    Run evaluation on existing dataset
    """
    print("Running evaluation...")

    if not os.path.exists(config.save_dir):
        print(f"Error: Dataset directory {config.save_dir} does not exist!")
        return

    # Load and analyze dataset
    trajectories, metadata = load_dataset(config.save_dir)

    print(f"\nEvaluation Results")
    print(f"=" * 20)
    print(f"Dataset: {metadata['environment']}")
    print(f"Agent: {metadata['agent']}")
    print(f"Total episodes: {len(trajectories)}")

    # Success rate
    successes = sum(1 for traj in trajectories if traj["success"])
    success_rate = successes / len(trajectories) * 100
    print(f"Success rate: {success_rate:.1f}% ({successes}/{len(trajectories)})")

    # Trajectory statistics
    lengths = [traj["step_count"] for traj in trajectories]
    rewards = [traj["total_reward"] for traj in trajectories]

    print(f"\nTrajectory Statistics:")
    print(f"  Steps - Mean: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
    print(f"        - Range: {np.min(lengths)} - {np.max(lengths)}")
    print(f"  Rewards - Mean: {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")
    print(f"          - Range: {np.min(rewards):.3f} - {np.max(rewards):.3f}")

    # Success vs failure analysis
    successful_episodes = [traj for traj in trajectories if traj["success"]]
    failed_episodes = [traj for traj in trajectories if not traj["success"]]

    if successful_episodes:
        success_lengths = [traj["step_count"] for traj in successful_episodes]
        print(f"\nSuccessful episodes:")
        print(
            f"  Steps - Mean: {np.mean(success_lengths):.1f} ± {np.std(success_lengths):.1f}"
        )
        print(f"        - Range: {np.min(success_lengths)} - {np.max(success_lengths)}")

    if failed_episodes:
        fail_lengths = [traj["step_count"] for traj in failed_episodes]
        print(f"\nFailed episodes:")
        print(
            f"  Steps - Mean: {np.mean(fail_lengths):.1f} ± {np.std(fail_lengths):.1f}"
        )
        print(f"        - Range: {np.min(fail_lengths)} - {np.max(fail_lengths)}")


def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(
        description="Run MiniGrid-LockedRoom-v0 A* agent experiments"
    )
    parser.add_argument(
        "mode",
        choices=["test", "generate", "evaluate", "all"],
        help="Mode to run: test (single episode), generate (dataset), evaluate (analyze), or all",
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
        help="Directory to save/load data",
    )
    parser.add_argument("--random_seed", type=int, default=42, help="Base random seed")
    parser.add_argument(
        "--n_processes", type=int, default=None, help="Number of parallel processes"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug output")

    args = parser.parse_args()

    # Create configuration
    config = Config()
    config.update_from_args(args)

    # Run based on mode
    if args.mode == "test":
        success = run_single_test(config)
        print(f"\nTest completed. Success: {success}")

    elif args.mode == "generate":
        run_data_generation(config)

    elif args.mode == "evaluate":
        run_evaluation(config)

    elif args.mode == "all":
        print("Running all modes...")

        # First run a test
        print("\n1. Running single test...")
        success = run_single_test(config)
        print(f"Test completed. Success: {success}")

        if success:
            # Then generate data
            print("\n2. Generating dataset...")
            run_data_generation(config)

            # Finally evaluate
            print("\n3. Running evaluation...")
            run_evaluation(config)
        else:
            print("Test failed, skipping data generation and evaluation.")


if __name__ == "__main__":
    main()
