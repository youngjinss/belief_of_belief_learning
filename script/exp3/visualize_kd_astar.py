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

import gymnasium as gym
from gymnasium.wrappers import TransformObservation

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

# Import our A* agent
from astar_agent import AStarAgent


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Visualize A* agent solving MiniGrid-KeyDoor-9x9-v0")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed (default: 42)")
    parser.add_argument("--episodes", type=int, default=3,
                        help="number of episodes to visualize (default: 3)")
    parser.add_argument("--pause", type=float, default=0.5,
                        help="pause duration between actions in seconds (default: 0.5)")
    parser.add_argument("--gif", type=str, default=None,
                        help="save output as gif with given filename (without .gif extension)")
    parser.add_argument("--max_steps", type=int, default=500,
                        help="maximum steps per episode (default: 500)")
    parser.add_argument("--env_size", type=str, default="9x9", 
                        choices=["3x3", "5x5", "9x9", "11x11"],
                        help="environment size (default: 9x9)")
    parser.add_argument("--observability", type=str, default="full",
                        choices=["full", "partial"],
                        help="observability type (default: full)")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Create environment
    env_name = f"MiniGrid-KeyDoor-{args.env_size}-v0"
    print(f"Creating environment: {env_name}")
    
    try:
        env = gym.make(env_name, max_steps=args.max_steps)
        # Disable gym wrappers that cause issues
        env = env.unwrapped if hasattr(env, 'unwrapped') else env
        print(f"✓ Environment {env_name} created successfully")
    except Exception as e:
        print(f"✗ Failed to create environment {env_name}: {e}")
        return
    
    # Create A* agent
    agent = AStarAgent(env, observability=args.observability)
    
    total_steps = 0
    total_rewards = 0
    successful_episodes = 0
    
    try:
        for episode in range(args.episodes):
            print(f"\n=== Episode {episode + 1}/{args.episodes} ===")
            
            # Reset environment
            env.seed(args.seed + episode)
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                obs, info = reset_result
            else:
                obs = reset_result
                info = {}
            
            # Reset agent
            agent.reset()
            
            # Handle observation if it's a dict
            if isinstance(obs, dict):
                obs = obs.get('image', obs)
            
            episode_reward = 0
            step_count = 0
            
            # Print episode information
            print(f"Mission: {env.mission}")
            print(f"Target door color: {env.target_door_color}")
            print(f"Agent preference: {env.preference}")
            print(f"Agent costs: {env.cost}")
            
            # Display initial state
            try:
                env.render()
            except Exception as e:
                print(f"Warning: Could not render: {e}")
            
            # Run episode
            while step_count < args.max_steps:
                # Get action from A* agent
                action = agent.get_action(obs)
                
                # Action names for display
                action_names = ['up', 'down', 'left', 'right', 'stay', 'pickup']
                action_name = action_names[action] if action < len(action_names) else f'action_{action}'
                
                # Take step
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Handle observation if it's a dict
                if isinstance(obs, dict):
                    obs = obs.get('image', obs)
                
                # Update counters
                episode_reward += reward
                step_count += 1
                
                # Print step information
                if reward != 0 or action == 5:  # Show pickup attempts and rewards
                    print(f"Step {step_count}: {action_name} -> reward: {reward:.2f}")
                    if action == 5:  # Pickup action
                        print(f"  Agent keys: {list(agent.collected_keys)}")
                        print(f"  Agent phase: {agent.strategy_phase}")
                
                # Show agent position and strategy
                if step_count % 10 == 0:  # Print every 10 steps
                    print(f"Step {step_count}: Agent at {agent.agent_pos}, Phase: {agent.strategy_phase}")
                
                # Render environment
                try:
                    env.render()
                except Exception as e:
                    print(f"Warning: Could not render: {e}")
                
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
            print(f"  Success: {'Yes' if episode_reward > 0 else 'No'}")
            print(f"  Final keys collected: {list(agent.collected_keys)}")
            print(f"  Final phase: {agent.strategy_phase}")
    
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