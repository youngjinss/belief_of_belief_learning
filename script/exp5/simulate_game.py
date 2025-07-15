import argparse
import numpy as np
import time
import sys
import os
from PIL import Image

# Add the lib directory to the path
lib_path = os.path.join(os.path.dirname(__file__), "..", "..", "lib")
sys.path.insert(0, lib_path)

# Import seed utility
from lib.utils.seed import set_seed

# Add the env directory to the path
env_path = os.path.join(lib_path, "env")
sys.path.insert(0, env_path)

import gymnasium as gym
from gymnasium.wrappers import TransformObservation

# Add gym_minigrid to Python path
gym_minigrid_path = os.path.join(os.path.dirname(__file__), "../../lib/env")
sys.path.insert(0, gym_minigrid_path)

# Import the gym_minigrid environments
try:
    import gym_minigrid

    print("Successfully imported gym_minigrid")
except Exception as e:
    print(f"Warning: Could not import gym_minigrid: {e}")
    # Continue anyway as environments might be registered elsewhere

# Import our modules
from config import Config
from achievers import AStarAgent, RandomAgent, Level0ValueAchiever, Level1ValueAchiever

# Set seed using Config default value
config = Config()
set_seed(config.seed)


def create_agent(agent_type, env, config):
    """Create agent based on type"""
    if agent_type == "astar":
        return AStarAgent(observability=config.observability)
    elif agent_type == "random":
        return RandomAgent(env.action_space, movement_prob=config.movement_prob)
    elif agent_type == "lv0va":
        agent_config = config.achiever_configs.get("lv0va", {})
        return Level0ValueAchiever(
            observability=agent_config.get("observability", "full"),
            movement_cost=agent_config.get("movement_cost", 0.01),
            wall_penalty=agent_config.get("wall_penalty", 2.0),
            gamma=agent_config.get("gamma", 0.99),
            temperature=agent_config.get("temperature", 0.1),
        )
    elif agent_type == "lv1va":
        agent_config = config.achiever_configs.get("lv1va", {})
        return Level1ValueAchiever(
            observability=agent_config.get("observability", "full"),
            movement_cost=agent_config.get("movement_cost", 0.01),
            wall_penalty=agent_config.get("wall_penalty", 2.0),
            gamma=agent_config.get("gamma", 0.99),
            temperature=agent_config.get("temperature", 0.1),
        )
    elif agent_type == "value":
        agent_config = config.achiever_configs.get("value", {})
        return Level0ValueAchiever(
            observability=agent_config.get("observability", "full"),
            movement_cost=agent_config.get("movement_cost", 0.01),
            wall_penalty=agent_config.get("wall_penalty", 2.0),
            gamma=agent_config.get("gamma", 0.99),
            temperature=agent_config.get("temperature", 0.1),
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def print_episode_info(env, episode, total_episodes):
    """Print episode information"""
    print(f"\n=== Episode {episode + 1}/{total_episodes} ===")
    print(f"Mission: {env.mission}")
    print(f"Target door color: {env.target_door_color}")
    if hasattr(env, "preference"):
        print(f"Agent preference: {env.preference}")
    if hasattr(env, "cost"):
        print(f"Agent costs: {env.cost}")


def print_step_info(step_count, action, action_name, reward, agent, config):
    """Print step information"""
    if config.log_rewards and reward != 0:
        print(f"Step {step_count}: {action_name} -> reward: {reward:.2f}")
        if hasattr(agent, "collected_keys"):
            print(f"  Agent keys: {list(agent.collected_keys)}")
        if hasattr(agent, "strategy_phase"):
            print(f"  Agent phase: {agent.strategy_phase}")

    # Show agent position and strategy periodically
    if config.debug and step_count % 10 == 0 and hasattr(agent, "agent_pos"):
        print(f"Step {step_count}: Agent at {agent.agent_pos}")
        if hasattr(agent, "strategy_phase"):
            print(f"  Phase: {agent.strategy_phase}")


def print_episode_summary(episode, step_count, episode_reward, agent, config):
    """Print episode summary"""
    print(f"Episode {episode + 1} summary:")
    print(f"  Steps: {step_count}")
    print(f"  Reward: {episode_reward:.2f}")
    print(f"  Success: {'Yes' if episode_reward > 0 else 'No'}")

    if hasattr(agent, "collected_keys"):
        print(f"  Final keys collected: {list(agent.collected_keys)}")
    if hasattr(agent, "strategy_phase"):
        print(f"  Final phase: {agent.strategy_phase}")


def render_to_image(env):
    """Render environment to PIL Image using native MiniGrid rendering"""
    # Get the rendered image as RGB array
    renderer = env.render()

    # If renderer is a MiniGrid Renderer object, get the array
    if hasattr(renderer, "getArray"):
        img = renderer.getArray()
        if isinstance(img, np.ndarray):
            return Image.fromarray(img)

    # If img is already a PIL Image, return it
    if isinstance(renderer, Image.Image):
        return renderer

    # If it's a numpy array, convert to PIL Image
    if isinstance(renderer, np.ndarray):
        if renderer.dtype != np.uint8:
            renderer = (renderer * 255).astype(np.uint8)
        return Image.fromarray(renderer)

    # If it's something else, try to handle it
    return None


def run_episode(env, agent, config, episode, args):
    """Run a single episode"""
    # Reset environment
    env.seed(config.seed + episode)
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
        obs = obs.get("image", obs)

    episode_reward = 0
    step_count = 0
    frames = []

    # Print episode information
    print_episode_info(env, episode, config.episodes)

    # Display initial state and capture frame for GIF
    if config.render:
        try:
            env.render()
        except Exception as e:
            print(f"Warning: Could not render: {e}")

    # Capture initial frame for GIF if saving
    if hasattr(args, "gif") and args.gif:
        initial_frame = render_to_image(env)
        if initial_frame:
            frames.append(initial_frame)

    # Run episode
    while step_count < config.max_steps:
        # Get action from agent
        action = agent.get_action(obs)

        # Action names for display
        action_names = [
            "up",
            "right",
            "down",
            "left",
            "stay",
            "pickup",
            "toggle",
        ]
        action_name = (
            action_names[action] if action < len(action_names) else f"action_{action}"
        )

        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Handle observation if it's a dict
        if isinstance(obs, dict):
            obs = obs.get("image", obs)

        # Update counters
        episode_reward += reward
        step_count += 1

        # Print step information
        print_step_info(step_count, action, action_name, reward, agent, config)

        # Render environment
        if config.render:
            try:
                env.render()
            except Exception as e:
                print(f"Warning: Could not render: {e}")

        # Capture frame for GIF if saving
        if hasattr(args, "gif") and args.gif:
            frame = render_to_image(env)
            if frame:
                frames.append(frame)

        # Let agent analyze feedback (for learning agents)
        if hasattr(agent, "analyze_feedback"):
            agent.analyze_feedback(reward, done)

        # Check if episode is done
        if done:
            # Save GIF if requested
            if hasattr(args, "gif") and args.gif and frames:
                gif_path = (
                    f"{args.gif}_ep{episode+1}.gif"
                    if args.gif
                    else f"keydoor_ep{episode+1}.gif"
                )
                print(f"Saving {len(frames)} frames to {gif_path}")
                frames[0].save(
                    gif_path,
                    save_all=True,
                    append_images=frames[1:],
                    duration=500,  # 500ms per frame
                    loop=0,
                )
                print(f"GIF saved to {gif_path}")

            if reward > 0:
                print(f"✓ SUCCESS! Episode completed with reward: {episode_reward:.2f}")
                return step_count, episode_reward, True
            else:
                print(f"Episode ended with reward: {episode_reward:.2f}")
                return step_count, episode_reward, False

        # Pause between actions
        if config.pause > 0:
            time.sleep(config.pause)

    # Episode ended due to max steps
    # Save GIF if requested
    if hasattr(args, "gif") and args.gif and frames:
        gif_path = (
            f"{args.gif}_ep{episode+1}.gif"
            if args.gif
            else f"keydoor_ep{episode+1}.gif"
        )
        print(f"Saving {len(frames)} frames to {gif_path}")
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=500,  # 500ms per frame
            loop=0,
        )
        print(f"GIF saved to {gif_path}")

    print(f"Episode ended after {step_count} steps (max reached)")
    return step_count, episode_reward, False


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Render agent solving MiniGrid-KeyDoor environment"
    )
    parser.add_argument(
        "--agent_type",
        type=str,
        default="astar",
        choices=["lv0va", "lv1va", "astar", "random", "value"],
        help="agent type to use (default: astar)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed (default: 42)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="number of episodes to visualize (default: 3)",
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
        default=500,
        help="maximum steps per episode (default: 500)",
    )
    parser.add_argument(
        "--env_size",
        type=str,
        default="9x9",
        choices=["3x3", "5x5", "9x9", "11x11"],
        help="environment size (default: 9x9)",
    )
    parser.add_argument(
        "--observability",
        type=str,
        default="full",
        choices=["full", "partial"],
        help="observability type (default: full)",
    )
    parser.add_argument("--debug", action="store_true", help="enable debug output")

    args = parser.parse_args()

    # Create configuration
    config = Config()
    config.update_from_args(args)
    config.validate()

    print(f"Configuration:")
    print(config)

    # Set random seed
    set_seed(config.seed)

    # Create environment
    env_name = config.get_env_name()
    print(f"Creating environment: {env_name}")

    try:
        env = gym.make(env_name, max_steps=config.max_steps)
        # Disable gym wrappers that cause issues
        env = env.unwrapped if hasattr(env, "unwrapped") else env

        # Set render mode for GIF saving
        if args.gif:
            env.render_mode = "rgb_array"

        print(f"✓ Environment {env_name} created successfully")
    except Exception as e:
        print(f"✗ Failed to create environment {env_name}: {e}")
        return

    # Create agent
    try:
        # Use first achiever type as default
        achiever_type = config.achiever_types[0]
        agent = create_agent(achiever_type, env, config)
        print(f"✓ Agent {achiever_type} created successfully")
    except Exception as e:
        # Use first achiever type as default
        achiever_type = config.achiever_types[0]
        print(f"✗ Failed to create agent {achiever_type}: {e}")
        return

    # Statistics tracking
    total_steps = 0
    total_rewards = 0
    successful_episodes = 0

    try:
        # Run episodes
        for episode in range(config.episodes):
            step_count, episode_reward, success = run_episode(
                env, agent, config, episode, args
            )

            total_steps += step_count
            total_rewards += episode_reward
            if success:
                successful_episodes += 1

            # Print episode summary
            print_episode_summary(episode, step_count, episode_reward, agent, config)

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    # Print final statistics
    print(f"\n=== Final Statistics ===")
    # Use first combination as default
    achiever_type = config.achiever_types[0]
    blocker_type = config.blocker_types[0]
    print(f"Achiever Type: {achiever_type}")
    print(f"Blocker Type: {blocker_type}")
    print(f"Total episodes: {config.episodes}")
    print(f"Successful episodes: {successful_episodes}")
    print(f"Success rate: {successful_episodes/config.episodes*100:.1f}%")
    print(f"Average steps per episode: {total_steps/config.episodes:.1f}")
    print(f"Average reward per episode: {total_rewards/config.episodes:.2f}")

    # Close environment
    env.close()


if __name__ == "__main__":
    main()
