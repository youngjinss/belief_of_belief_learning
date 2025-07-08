#!/usr/bin/env python3

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

# Add gym_minigrid to Python path (same as visualize_gotodoor.py)
gym_minigrid_path = os.path.join(os.path.dirname(__file__), "../../")
if gym_minigrid_path not in sys.path:
    sys.path.insert(0, gym_minigrid_path)

# Import the gym_minigrid environments
try:
    import gym_minigrid

    print("Successfully imported gym_minigrid")
except Exception as e:
    print(f"Warning: Could not import gym_minigrid: {e}")
    # Continue anyway as environments might be registered elsewhere


def test_keydoor_env():
    """Test the custom KeyDoor environment"""

    # Test different sizes
    env_ids = [
        "MiniGrid-KeyDoor-3x3-v0",
        "MiniGrid-KeyDoor-5x5-v0",
        "MiniGrid-KeyDoor-9x9-v0",
        "MiniGrid-KeyDoor-11x11-v0",
    ]

    for env_id in env_ids:
        print(f"\n=== Testing {env_id} ===")

        try:
            # Create environment
            env = gym.make(env_id)

            # Reset environment
            obs, info = env.reset(seed=42)

            print(f"Grid size: {env.grid.width}x{env.grid.height}")
            print(f"Action space: {env.action_space}")
            print(f"Max keys: {env.max_keys}")
            print(f"Target door: {env.target_door_color}")
            print(f"Mission: {env.mission}")

            # Test a few steps
            for step in range(10):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)

                if terminated or truncated:
                    print(f"Episode ended at step {step}")
                    break

            env.close()
            print(f"✓ {env_id} works correctly")

        except Exception as e:
            print(f"✗ {env_id} failed: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    test_keydoor_env()
