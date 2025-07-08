import argparse
import numpy as np
import time
import sys
import os

# Add the lib directory to the path
lib_path = os.path.join(os.path.dirname(__file__), '..', '..', 'lib')
sys.path.insert(0, lib_path)

# Add the env directory to the path
env_path = os.path.join(lib_path, 'env')
sys.path.insert(0, env_path)

# Add the rl-starter-files utils to the path
rl_utils_path = os.path.join(env_path, 'rl-starter-files')
sys.path.insert(0, rl_utils_path)

import gymnasium as gym

# Import utils from rl-starter-files
try:
    import utils
except ImportError:
    print("Warning: Could not import utils from rl-starter-files")
    utils = None

# Add gym_minigrid to Python path
gym_minigrid_path = os.path.join(os.path.dirname(__file__), '../../lib/env')
sys.path.insert(0, gym_minigrid_path)

# Import the gym_minigrid environments
try:
    import gym_minigrid
    print("Successfully imported gym_minigrid")
except Exception as e:
    print(f"Warning: Could not import gym_minigrid: {e}")
    # Continue anyway as environments might be registered elsewhere


class RandomAgent:
    """A random agent that explores the KeyDoor environment."""
    
    def __init__(self, action_space, pickup_prob=0.3, movement_prob=0.6):
        self.action_space = action_space
        self.pickup_prob = pickup_prob
        self.movement_prob = movement_prob
    
    def get_action(self, obs):
        """Return a random action with bias towards movement and pickup."""
        rand = np.random.random()
        
        if rand < self.pickup_prob:
            # Try pickup action
            return 5  # pickup
        elif rand < self.pickup_prob + self.movement_prob:
            # Focus on movement actions: up, down, left, right
            return np.random.choice([0, 1, 2, 3])
        else:
            # Stay action
            return 4  # stay
    
    def analyze_feedback(self, reward, done):
        """Random agent doesn't learn from feedback."""
        pass


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Visualize random agent exploring MiniGrid-KeyDoor-9x9-v0")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed (default: 42)")
    parser.add_argument("--episodes", type=int, default=3,
                        help="number of episodes to visualize (default: 3)")
    parser.add_argument("--pause", type=float, default=0.3,
                        help="pause duration between actions in seconds (default: 0.3)")
    parser.add_argument("--gif", type=str, default=None,
                        help="save output as gif with given filename (without .gif extension)")
    parser.add_argument("--max_steps", type=int, default=500,
                        help="maximum steps per episode (default: 500)")
    parser.add_argument("--env_size", type=str, default="9x9", 
                        choices=["3x3", "5x5", "9x9", "11x11"],
                        help="environment size (default: 9x9)")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Create environment
    env_name = f"MiniGrid-KeyDoor-{args.env_size}-v0"
    print(f"Creating environment: {env_name}")
    
    try:
        if utils:
            env = utils.make_env(env_name, args.seed)
        else:
            env = gym.make(env_name)
        print(f"✓ Environment {env_name} created successfully")
    except Exception as e:
        print(f"✗ Failed to create environment {env_name}: {e}")
        return
    
    # Create random agent
    agent = RandomAgent(env.action_space)
    
    total_steps = 0
    total_rewards = 0
    successful_episodes = 0
    
    try:
        for episode in range(args.episodes):
            print(f"\n=== Episode {episode + 1}/{args.episodes} ===")
            
            # Reset environment
            reset_result = env.reset(seed=args.seed + episode)
            if isinstance(reset_result, tuple):
                obs, info = reset_result
            else:
                obs = reset_result
                info = {}
            
            episode_reward = 0
            step_count = 0
            
            # Print episode information
            print(f"Mission: {env.mission}")
            print(f"Target door: {env.target_door_color}")
            print(f"Preference: {env.preference}")
            print(f"Cost: {env.cost}")
            print(f"Max keys: {env.max_keys}")
            
            # Display initial state
            try:
                env.render()
            except Exception as e:
                print(f"Warning: Could not render: {e}")
            
            # Run episode
            while step_count < args.max_steps:
                # Get action from random agent
                action = agent.get_action(obs)
                
                # Action names for display
                action_names = ['up', 'down', 'left', 'right', 'stay', 'pickup']
                action_name = action_names[action] if action < len(action_names) else f'action_{action}'
                
                # Take step
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Update counters
                episode_reward += reward
                step_count += 1
                
                # Print step information
                if reward != 0 or action == 5:  # Show pickup attempts and rewards
                    print(f"Step {step_count}: {action_name} -> reward: {reward:.2f}, keys: {len(env.agent_keys)}/{env.max_keys}")
                    if len(env.agent_keys) > 0:
                        print(f"  Agent keys: {env.agent_keys}")
                
                # Render environment
                try:
                    env.render()
                except Exception as e:
                    print(f"Warning: Could not render: {e}")
                
                # Let agent analyze feedback (no-op for random agent)
                agent.analyze_feedback(reward, done)
                
                # Check if episode is done
                if done:
                    if reward > 0:
                        successful_episodes += 1
                        print(f"✓ SUCCESS! Episode completed with reward: {episode_reward:.2f}")
                    else:
                        print(f"Episode ended with reward: {episode_reward:.2f}")
                    break
                
                # Pause between actions
                if args.pause > 0:
                    time.sleep(args.pause)
            
            total_steps += step_count
            total_rewards += episode_reward
            
            print(f"Episode {episode + 1} summary:")
            print(f"  Steps: {step_count}")
            print(f"  Reward: {episode_reward:.2f}")
            print(f"  Keys collected: {len(env.agent_keys)}")
            print(f"  Success: {'Yes' if episode_reward > 0 else 'No'}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    # Print final statistics
    print(f"\n=== Final Statistics ===")
    print(f"Total episodes: {args.episodes}")
    print(f"Successful episodes: {successful_episodes}")
    print(f"Success rate: {successful_episodes/args.episodes*100:.1f}%")
    print(f"Average steps per episode: {total_steps/args.episodes:.1f}")
    print(f"Average reward per episode: {total_rewards/args.episodes:.2f}")
    
    # Close environment
    env.close()


if __name__ == "__main__":
    main()