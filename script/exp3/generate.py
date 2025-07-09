import os
import sys
import numpy as np
import multiprocessing as mp
from functools import partial
import warnings

# Suppress gymnasium registration warnings
warnings.filterwarnings(
    "ignore", message=".*Overriding environment.*already in registry.*"
)
warnings.filterwarnings("ignore", message=".*gym_minigrid has been deprecated.*")
warnings.filterwarnings(
    "ignore", message=".*environment creator metadata doesn't include `render_modes`.*"
)

# Add parent directory to path for imports
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from lib.env.gym_minigrid.envs.keydoor import KeyDoor5x5Env, KeyDoor9x9Env
from script.exp3.agents import AStarAgent, RandomAgent, ValueAgent
from script.exp3.config import Config

"""
Data generation for KeyDoor environment in ToMnet format
Adapted from experiment 5 for KeyDoor environment with 4 keys and 4 doors
"""


def calculate_successor_representation(
    positions, query_time_t, grid_size=9, gammas=None, num_rollouts=1
):
    """
    Calculate successor representation from query time t onwards using the correct formula
    SRγ(s) = 1/Z × Σ(from Δt=0 to T-t) γ^Δt × I(s_{t+Δt} = s)

    Args:
        positions: List of positions visited in the episode
        query_time_t: Current timestep t to calculate SR from
        grid_size: Size of the grid (9x9 for KeyDoor)
        gammas: List of discount factors (defaults to [0.5, 0.9, 0.99])
        num_rollouts: Number of rollouts (for stochastic agents, default 1)

    Returns:
        sr_maps_sparse: List of sparse representations [(position, value)] for each gamma
    """
    if gammas is None:
        gammas = [0.5, 0.9, 0.99]

    sr_maps_sparse = []

    # Calculate SR for each discount factor
    for gamma in gammas:
        sr_map = np.zeros((grid_size, grid_size))

        # Only consider future states from query_time_t onwards
        if query_time_t < len(positions):
            T_minus_t = len(positions) - query_time_t

            # Count discounted future state visitations
            for delta_t in range(T_minus_t):
                future_timestep = query_time_t + delta_t
                if future_timestep < len(positions):
                    discount = gamma**delta_t
                    future_pos = positions[future_timestep]
                    sr_map[future_pos[0], future_pos[1]] += discount

            # Normalize the SR map (Z normalization)
            if sr_map.sum() > 0:
                sr_map = sr_map / sr_map.sum()

        # Convert to sparse format: [(position, value)] for non-zero values
        sparse_sr = []
        for i in range(grid_size):
            for j in range(grid_size):
                if sr_map[i, j] > 0:
                    sparse_sr.append(((i, j), sr_map[i, j]))

        sr_maps_sparse.append(sparse_sr)

    return sr_maps_sparse


def calculate_sr_labels_for_trajectory(positions, grid_size=9, gammas=None):
    """
    Calculate SR labels for each timestep in the trajectory

    Args:
        positions: List of positions visited in the episode
        grid_size: Size of the grid (9x9 for KeyDoor)
        gammas: List of discount factors (defaults to [0.5, 0.9, 0.99])

    Returns:
        sr_labels_per_timestep: List of SR labels for each timestep
    """
    if gammas is None:
        gammas = [0.5, 0.9, 0.99]

    sr_labels_per_timestep = []

    # Calculate SR for each timestep in the trajectory
    for t in range(len(positions)):
        sr_sparse = calculate_successor_representation(
            positions, query_time_t=t, grid_size=grid_size, gammas=gammas
        )
        sr_labels_per_timestep.append(sr_sparse)

    return sr_labels_per_timestep


def calculate_key_door_rank(keys_collected, doors_opened, target_door_color):
    """
    Calculate the rank of keys and doors based on collection order and target

    Args:
        keys_collected: List of key colors collected in order
        doors_opened: List of door colors opened in order
        target_door_color: The target door color

    Returns:
        key_door_rank: List of ranks [rank_key0, rank_key1, rank_key2, rank_key3]
                      where 1 is the most important (target key), 4 is least important
    """
    colors = ["red", "green", "blue", "yellow"]
    key_door_rank = [4, 4, 4, 4]  # Default lowest rank

    # Target key gets rank 1
    if target_door_color in colors:
        target_idx = colors.index(target_door_color)
        key_door_rank[target_idx] = 1

    # Other collected keys get ranks 2, 3 based on collection order
    rank = 2
    for key_color in keys_collected:
        if key_color != target_door_color and key_color in colors:
            idx = colors.index(key_color)
            if key_door_rank[idx] == 4:  # Not yet ranked
                key_door_rank[idx] = rank
                rank += 1

    return key_door_rank


def calculate_consumption_labels(keys_collected, doors_opened):
    """
    Calculate consumption labels for keys and doors

    Args:
        keys_collected: List of key colors collected
        doors_opened: List of door colors opened

    Returns:
        consumption_vector: Binary vector of length 8 (4 keys + 4 doors)
                          [key_red, key_green, key_blue, key_yellow,
                           door_red, door_green, door_blue, door_yellow]
    """
    consumption_vector = np.zeros(8, dtype=np.float32)
    colors = ["red", "green", "blue", "yellow"]

    # Mark collected keys
    for key_color in keys_collected:
        if key_color in colors:
            idx = colors.index(key_color)
            consumption_vector[idx] = 1.0

    # Mark opened doors
    for door_color in doors_opened:
        if door_color in colors:
            idx = colors.index(door_color)
            consumption_vector[4 + idx] = 1.0

    return consumption_vector


def env_to_maze_format(env, agent_pos):
    """
    Convert KeyDoor environment to maze format without outer walls

    Args:
        env: KeyDoor environment
        agent_pos: Current agent position

    Returns:
        maze_str: String representation of the maze (9x9 internal grid)
    """
    maze_lines = []
    width, height = env.width, env.height

    # Process each row without adding outer walls
    for j in range(height):
        row = ""
        for i in range(width):
            cell = env.grid.get(i, j)

            if agent_pos == (i, j):
                row += "O"  # Agent
            elif cell is None:
                row += "-"  # Empty space
            elif cell.type == "wall":
                row += "#"  # Wall
            elif cell.type == "key":
                # Map key colors to letters A, B, C, D
                color_map = {"red": "A", "green": "B", "blue": "C", "yellow": "D"}
                row += color_map.get(cell.color, "?")
            elif cell.type == "door":
                # Use lowercase for doors
                color_map = {"red": "a", "green": "b", "blue": "c", "yellow": "d"}
                row += color_map.get(cell.color, "?")
            else:
                row += "?"  # Unknown

        maze_lines.append(row)

    return maze_lines


def save_game_with_labels(
    agent,
    env,
    sr_labels_per_timestep,
    consumption_labels,
    key_door_rank,
    name="",
    base_dir="data/exp3",
    game_id=None,
    initial_maze_lines=None,
    gammas=None,
    goal_rewards=None,
):
    """
    Save game data with SR and consumption labels in experiment 5 format

    Args:
        agent: The agent that played the game
        env: The KeyDoor environment
        sr_labels_per_timestep: SR labels for each timestep (sparse format)
        consumption_labels: Consumption labels for keys and doors
        key_door_rank: Rank of keys based on importance
        name: Output directory name
        base_dir: Base directory for saving
        game_id: Unique game ID for file naming
    """
    import uuid

    # Get the path to folder
    gf = base_dir if name == "" else os.path.join(base_dir, name)
    os.makedirs(gf, exist_ok=True)

    # Use game_id if provided, otherwise generate a unique filename
    if game_id is not None:
        new_file_path = os.path.join(gf, f"test{game_id}.txt")
    else:
        # Fallback to unique timestamp-based naming
        unique_id = str(uuid.uuid4())[:8]
        timestamp = int(np.random.rand() * 1e9)
        new_file_path = os.path.join(gf, f"test_{timestamp}_{unique_id}.txt")

    with open(new_file_path, "w") as f:
        f.write("Maze:\n")

        # Save the maze in experiment 5 format (use initial state if provided, otherwise final state)
        if initial_maze_lines is not None:
            maze_lines = initial_maze_lines
        else:
            # Fallback to final state (for backward compatibility)
            maze_lines = env_to_maze_format(
                env,
                (
                    agent.position_history[-1]
                    if hasattr(agent, "position_history") and agent.position_history
                    else env.agent_pos
                ),
            )
        for line in maze_lines:
            f.write(line + "\n")

        # Save key/door rank (similar to goal rank)
        f.write("Goal Consumed Rank : " + str(key_door_rank) + "\n")
        f.write("Trajectory length: " + str(len(agent.action_history)) + "\n")

        # Save goal rewards (sum constrained to configured total)
        if goal_rewards is not None:
            # Convert to list format: [red, green, blue, yellow]
            colors = ["red", "green", "blue", "yellow"]
            reward_list = [goal_rewards.get(color, 0.0) for color in colors]
            f.write("Goal Rewards: " + ",".join(map(str, reward_list)) + "\n")
            f.write("Goal Rewards Sum: " + str(sum(reward_list)) + "\n")

        # Save consumption labels
        f.write(
            "Consumption Labels: "
            + ",".join(map(str, consumption_labels.tolist()))
            + "\n"
        )

        # Save SR data per timestep in sparse format
        if gammas is None:
            gammas = [0.5, 0.9, 0.99]

        f.write("SR_Data_Per_Timestep:\n")
        for t, sr_data_at_t in enumerate(sr_labels_per_timestep):
            f.write(f"Timestep_{t}:\n")
            for gamma_idx, gamma in enumerate(gammas):
                sparse_sr = (
                    sr_data_at_t[gamma_idx] if gamma_idx < len(sr_data_at_t) else []
                )
                # Convert sparse format [(position, value)] to string
                sparse_str = ";".join(
                    [f"{pos[0]},{pos[1]}:{val}" for pos, val in sparse_sr]
                )
                f.write(f"SR_gamma_{gamma}: {sparse_str}\n")

        # Save trajectory moves
        for i in range(len(agent.action_history)):
            pos = (
                agent.position_history[i]
                if hasattr(agent, "position_history")
                and i < len(agent.position_history)
                else [0, 0]
            )
            action = agent.action_history[i]

            # Check if key or door was interacted at this step
            interaction = "X"
            if hasattr(agent, "keys_collected_steps"):
                for step, key_color in agent.keys_collected_steps:
                    if step == i:
                        color_map = {
                            "red": "A",
                            "green": "B",
                            "blue": "C",
                            "yellow": "D",
                        }
                        interaction = color_map.get(key_color, "X")
                        break

            if interaction == "X" and hasattr(agent, "doors_opened_steps"):
                for step, door_color in agent.doors_opened_steps:
                    if step == i:
                        color_map = {
                            "red": "a",
                            "green": "b",
                            "blue": "c",
                            "yellow": "d",
                        }
                        interaction = color_map.get(door_color, "X")
                        break

            msg = f"[{pos[0]}, {pos[1]}] : {action} : {interaction}"
            f.write(msg + "\n")


def generate_game_rewards(config_dict, game_id):
    """
    Generate goal rewards for this game based on config settings

    Args:
        config_dict: Dictionary containing configuration
        game_id: Game ID for seeding

    Returns:
        dict: Mapping of goal colors to reward values
    """
    # Set seed for consistent reward generation for this game
    np.random.seed(config_dict["base_random_seed"] + game_id + 1000)

    reward_settings = config_dict.get("goal_reward_settings", {})

    if reward_settings.get("use_random_rewards", True):
        # Generate random rewards that sum to total
        total_reward = reward_settings.get("total_reward_sum", 4)
        min_reward = reward_settings.get("min_reward", 0.1)
        max_reward = reward_settings.get("max_reward", 3.0)

        # Generate 3 random split points
        splits = np.random.uniform(0, 1, 3)
        splits = np.sort(splits)

        # Create 4 proportions
        proportions = [
            splits[0],
            splits[1] - splits[0],
            splits[2] - splits[1],
            1 - splits[2],
        ]

        # Scale to total reward
        rewards = [prop * total_reward for prop in proportions]

        # Ensure minimum reward constraint
        for i in range(len(rewards)):
            if rewards[i] < min_reward:
                rewards[i] = min_reward

        # Rescale to maintain sum constraint
        current_sum = sum(rewards)
        if current_sum > 0:
            scale_factor = total_reward / current_sum
            rewards = [r * scale_factor for r in rewards]

        # Ensure no reward exceeds maximum
        for i in range(len(rewards)):
            if rewards[i] > max_reward:
                rewards[i] = max_reward

        # Final rescaling to maintain exact sum
        current_sum = sum(rewards)
        if current_sum > 0:
            scale_factor = total_reward / current_sum
            rewards = [r * scale_factor for r in rewards]
    else:
        # Use default rewards
        rewards = reward_settings.get("default_rewards", [0.5, 1.0, 1.5, 1.0])

    # Map to color names
    colors = ["red", "green", "blue", "yellow"]
    return {colors[i]: rewards[i] for i in range(len(colors))}


def run_single_game(game_id, config_dict, save_dir):
    """
    Run a single KeyDoor game simulation

    Args:
        game_id: Unique identifier for this game
        config_dict: Dictionary containing config parameters
        save_dir: Directory to save game data

    Returns:
        game_id: For tracking completion
    """
    # Set unique random seed for this game
    np.random.seed(config_dict["base_random_seed"] + game_id)

    # Generate goal rewards for this game
    goal_rewards = generate_game_rewards(config_dict, game_id)

    # Create KeyDoor environment using gym.make (same as render_kd.py)
    env_size = config_dict["env_size"]
    if env_size == "5x5":
        env_name = "MiniGrid-KeyDoor-5x5-v0"
    elif env_size == "9x9":
        env_name = "MiniGrid-KeyDoor-9x9-v0"
    else:
        raise ValueError(f"Unknown environment size: {env_size}")

    import gymnasium as gym

    # Create environment same way as render_kd.py
    env = gym.make(env_name, max_steps=config_dict["max_steps"])
    env = env.unwrapped if hasattr(env, "unwrapped") else env

    # Reset environment with seed
    env.seed(config_dict["base_random_seed"] + game_id)
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        obs, _ = reset_result
    else:
        obs = reset_result

    # Handle observation if it's a dict (same as render_kd.py)
    if isinstance(obs, dict):
        obs = obs.get("image", obs)

    # Create agent based on config (same as render_kd.py)
    agent_type = config_dict["agent_type"]
    if agent_type == "astar":
        agent = AStarAgent(env, observability=config_dict["observability"])
    elif agent_type == "value":
        agent_config = config_dict.get("agent_configs", {}).get("value", {})
        agent = ValueAgent(
            env,
            observability=agent_config.get("observability", "full"),
            movement_cost=agent_config.get("movement_cost", 0.01),
            wall_penalty=agent_config.get("wall_penalty", 2.0),
            gamma=agent_config.get("gamma", 0.99),
            temperature=agent_config.get("temperature", 0.1),
        )
    elif agent_type == "random":
        agent = RandomAgent(
            env.action_space, movement_prob=config_dict.get("random_movement_prob", 0.8)
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    # Reset agent
    agent.reset()

    # Initialize tracking
    position_history = []
    action_history = []
    keys_collected = []
    doors_opened = []
    keys_collected_steps = []
    doors_opened_steps = []

    # Save initial maze state (before agent starts moving)
    initial_agent_pos = tuple(env.agent_pos)
    initial_maze_lines = env_to_maze_format(env, initial_agent_pos)

    # Run game simulation
    step_count = 0
    max_steps = config_dict["max_steps"]
    episode_reward = 0

    # Log goal rewards
    total_reward = sum(goal_rewards.values())

    while step_count < max_steps:
        # Record current position
        current_position = tuple(env.agent_pos)
        position_history.append(current_position)

        # Get action from agent
        action = agent.get_action(obs)
        action_history.append(action)

        # Execute action (same as render_kd.py)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Handle observation if it's a dict
        if isinstance(obs, dict):
            obs = obs.get("image", obs)

        # Update reward
        episode_reward += reward
        step_count += 1

        # Track key collection and door opening
        if hasattr(agent, "collected_keys"):
            current_keys = list(agent.collected_keys)
            if len(current_keys) > len(keys_collected):
                new_key = [k for k in current_keys if k not in keys_collected][0]
                keys_collected.append(new_key)
                keys_collected_steps.append((step_count - 1, new_key))

        # Check for door opening by reward
        if reward > 0 and step_count > 1:
            if reward >= 1.0:  # Door opening reward
                # Find which door was opened
                for x in range(env.grid.width):
                    for y in range(env.grid.height):
                        obj = env.grid.get(x, y)
                        if obj and obj.type == "door" and obj.is_open:
                            if obj.color not in doors_opened:
                                doors_opened.append(obj.color)
                                doors_opened_steps.append((step_count - 1, obj.color))
                                break

        # Let agent analyze feedback (for learning agents)
        if hasattr(agent, "analyze_feedback"):
            agent.analyze_feedback(reward, done)

        # Check if episode is done
        if done:
            break

    # Store tracking data in agent for save function
    agent.position_history = position_history
    agent.action_history = action_history
    agent.keys_collected_steps = keys_collected_steps
    agent.doors_opened_steps = doors_opened_steps

    # Calculate SR labels for each timestep
    sr_gammas = config_dict.get("sr_gammas", [0.5, 0.9, 0.99])
    sr_labels_per_timestep = calculate_sr_labels_for_trajectory(
        position_history, grid_size=env.width, gammas=sr_gammas
    )

    # Calculate consumption labels (keys and doors)
    consumption_labels = calculate_consumption_labels(keys_collected, doors_opened)

    # Calculate key/door rank based on target
    target_door_color = (
        env.target_door_color if hasattr(env, "target_door_color") else "yellow"
    )
    key_door_rank = calculate_key_door_rank(
        keys_collected, doors_opened, target_door_color
    )

    # Save game with labels
    save_game_with_labels(
        agent=agent,
        env=env,
        sr_labels_per_timestep=sr_labels_per_timestep,
        consumption_labels=consumption_labels,
        key_door_rank=key_door_rank,
        name="",
        base_dir=save_dir,
        game_id=game_id,
        initial_maze_lines=initial_maze_lines,
        gammas=sr_gammas,
        goal_rewards=goal_rewards,
    )

    return game_id


def generate_trajectories(config=None, random_seed=42, n_processes=None):
    """
    Generate trajectories for KeyDoor environment in ToMnet format

    Args:
        config: Config object containing all parameters. If None, uses default values.
        random_seed: Random seed for environment generation
        n_processes: Number of parallel processes
    """
    if config is None:
        config = Config()

    n_games = config.n_games

    print(f"Generating {n_games} KeyDoor trajectories with random seed: {random_seed}")
    print(f"Agent type: {config.agent_type}")
    print(f"Environment size: {config.env_size}")

    # Create output directory
    os.makedirs(config.save_dir, exist_ok=True)

    # Set number of processes (default to CPU count - 1, min 1)
    if n_processes is None:
        n_processes = max(1, mp.cpu_count() - 1)

    print(f"Using {n_processes} processes for parallel game generation")

    # Convert config to dictionary for multiprocessing
    config_dict = {
        "env_size": config.env_size,
        "max_steps": config.max_steps,
        "observability": config.observability,
        "agent_type": config.agent_type,
        "base_random_seed": random_seed,
        # Agent-specific configs (complete configurations)
        "agent_configs": config.agent_configs,
        # SR settings
        "sr_gammas": config.sr_settings["gammas"],
        "sr_grid_size": config.sr_settings["grid_size"],
        # Goal reward settings
        "goal_reward_settings": config.goal_reward_settings,
        # Backward compatibility
        "random_movement_prob": config.agent_configs.get("random", {}).get(
            "movement_prob", 0.8
        ),
    }

    # Create partial function with fixed arguments
    game_func = partial(
        run_single_game, config_dict=config_dict, save_dir=config.save_dir
    )

    # Run games in parallel
    with mp.Pool(processes=n_processes) as pool:
        # Use imap for progress tracking
        game_ids = range(n_games)
        results = []

        for i, result in enumerate(pool.imap(game_func, game_ids)):
            results.append(result)
            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{n_games} games")

        # Wait for all processes to complete
        pool.close()
        pool.join()

    print(f"Generated {n_games} games successfully using {n_processes} processes!")
    print(f"Data saved to: {config.save_dir}")


if __name__ == "__main__":
    # Set multiprocessing start method to avoid issues
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass  # Already set

    import argparse

    parser = argparse.ArgumentParser(
        description="Generate trajectories for KeyDoor environment in ToMnet format"
    )
    parser.add_argument(
        "--config_override",
        action="store_true",
        help="Override config with command line arguments",
    )
    parser.add_argument("--n_games", type=int, help="Number of games to generate")
    parser.add_argument(
        "--agent_type",
        type=str,
        choices=["astar", "value", "random"],
        default="astar",
        help="Type of agent to use",
    )
    parser.add_argument(
        "--env_size",
        type=str,
        choices=["5x5", "9x9", "11x11"],
        default="9x9",
        help="KeyDoor environment size",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="data/exp3",
        help="Directory to save generated data",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for environment generation",
    )
    parser.add_argument(
        "--n_processes",
        type=int,
        default=None,
        help="Number of parallel processes (default: CPU count - 1)",
    )

    args = parser.parse_args()

    config = Config()

    # Override config with command line arguments if specified
    if args.config_override:
        config.update_from_args(args)

    generate_trajectories(
        config, random_seed=args.random_seed, n_processes=args.n_processes
    )
