import argparse
import numpy as np
import time
import sys
import os

# Add the lib directory to the path
lib_path = os.path.join(os.path.dirname(__file__), "..", "..", "lib")
sys.path.insert(0, lib_path)

# Add the env directory to the path
env_path = os.path.join(lib_path, "env")
sys.path.insert(0, env_path)

# Add the rl-starter-files utils to the path
rl_utils_path = os.path.join(env_path, "rl-starter-files")
sys.path.insert(0, rl_utils_path)

import gymnasium as gym

# Import utils from rl-starter-files
import utils

# Add gym_minigrid to Python path (same as train.py)
gym_minigrid_path = os.path.join(os.path.dirname(__file__), "../../")
if gym_minigrid_path not in sys.path:
    sys.path.insert(0, gym_minigrid_path)

# Import the gym_minigrid environments
try:
    import gym_minigrid
except Exception as e:
    print(f"Warning: Could not import gym_minigrid: {e}")
    # Continue anyway as environments might be registered elsewhere


class RandomAgent:
    """A simple random agent that selects actions uniformly at random."""

    def __init__(self, action_space, avoid_done_prob=0.9):
        self.action_space = action_space
        self.avoid_done_prob = avoid_done_prob

    def get_action(self, obs):
        """Return a random action, focusing on movement actions."""
        if np.random.random() < self.avoid_done_prob:
            # Focus on movement actions: left, right, forward
            return np.random.choice([0, 1, 2])
        else:
            # Sometimes try other actions including done
            return self.action_space.sample()

    def analyze_feedback(self, reward, done):
        """Random agent doesn't learn from feedback."""
        pass


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Visualize random agent exploring MiniGrid-GoToDoor-8x8-v0"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed (default: 42)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="number of episodes to visualize (default: 5)",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=0.5,
        help="pause duration between actions in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--gif",
        type=str,
        default=None,
        help="save output as gif with given filename (without .gif extension)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=200,
        help="maximum steps per episode (default: 200)",
    )

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Create environment using utils.make_env (same as train.py)
    env_name = "MiniGrid-GoToDoor-5x5-v0"
    try:
        env = utils.make_env(env_name, args.seed)
    except Exception as e:
        return

    # Create random agent
    agent = RandomAgent(env.action_space)

    total_steps = 0
    total_rewards = 0
    successful_episodes = 0

    try:
        for episode in range(args.episodes):
            # Reset environment
            obs, info = env.reset(seed=args.seed + episode)
            episode_reward = 0
            step_count = 0

            # Display initial state
            env.render()

            # Run episode
            while step_count < args.max_steps:
                # Get action from random agent
                action = agent.get_action(obs)

                # Take step
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Update counters
                episode_reward += reward
                step_count += 1

                # Render environment
                env.render()

                # Let agent analyze feedback (no-op for random agent)
                # agent.analyze_feedback(reward, done)

                # Check if episode is done
                if done:
                    if reward > 0:
                        successful_episodes += 1
                    break

                # Pause between actions
                if args.pause > 0:
                    time.sleep(args.pause)

            total_steps += step_count
            total_rewards += episode_reward

    except KeyboardInterrupt:
        pass

    # Close environment
    env.close()


if __name__ == "__main__":
    main()
