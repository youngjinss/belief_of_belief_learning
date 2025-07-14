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

from lib.env.gym_minigrid.envs.achiever_blocker import (
    AchieverBlocker5x5Env,
    AchieverBlocker9x9Env,
    AchieverBlocker11x11Env,
)
from script.exp5.achievers import (
    AStarAgent,
    RandomAgent as AchieverRandomAgent,
    ValueAgent,
)
from script.exp5.blockers import (
    RandomAgent as BlockerRandomAgent,
    GoalDirectAgent as BlockerGoalDirectAgent,
)
from script.exp5.config import Config

"""
Data generation for AchieverBlocker environment in ToMnet format
2-agent environment with achiever and blocker agents
"""


def calculate_successor_representation_vectorized(positions, grid_size=9, gammas=None):
    """
    Vectorized calculation of successor representation for all timesteps and gammas.

    Optimized version that processes all timesteps and discount factors simultaneously
    using NumPy vectorization for significant performance improvement.

    Args:
        positions: List of positions visited in the episode
        grid_size: Size of the grid (9x9 for AchieverBlocker)
        gammas: List of discount factors (defaults to [0.5, 0.9, 0.99])

    Returns:
        sr_labels_per_timestep: List of SR labels for each timestep
                               Each element contains sparse representations for all gammas
    """
    if gammas is None:
        gammas = [0.5, 0.9, 0.99]

    T = len(positions)
    n_gammas = len(gammas)

    if T == 0:
        return []

    # Convert positions to numpy array for vectorized operations
    pos_array = np.array(positions)  # Shape: (T, 2)

    # Precompute all discount factors: gamma^delta_t for all gammas and delta_t
    max_delta_t = T
    gamma_powers = np.zeros((n_gammas, max_delta_t))  # Shape: (n_gammas, max_delta_t)
    for g_idx, gamma in enumerate(gammas):
        gamma_powers[g_idx] = gamma ** np.arange(max_delta_t)

    sr_labels_per_timestep = []

    # Process each query timestep
    for query_t in range(T):
        # Get future positions from query_t onwards
        future_positions = pos_array[query_t:]  # Shape: (T-query_t, 2)
        remaining_steps = T - query_t

        if remaining_steps == 0:
            # No future steps, return empty sparse representations
            sr_labels_per_timestep.append([[] for _ in gammas])
            continue

        # Initialize SR maps for all gammas: (n_gammas, grid_size, grid_size)
        sr_maps = np.zeros((n_gammas, grid_size, grid_size))

        # Vectorized accumulation of discounted visits
        for delta_t in range(remaining_steps):
            pos = future_positions[delta_t]
            x, y = pos[0], pos[1]

            # Add discounted visit for all gammas simultaneously
            discounts = gamma_powers[:, delta_t]  # Shape: (n_gammas,)
            sr_maps[:, x, y] += discounts

        # Normalize each SR map
        sr_sums = sr_maps.sum(axis=(1, 2), keepdims=True)  # Shape: (n_gammas, 1, 1)
        sr_sums = np.where(sr_sums > 0, sr_sums, 1)  # Avoid division by zero
        sr_maps = sr_maps / sr_sums

        # Convert to sparse format for each gamma
        sparse_representations = []
        for g_idx in range(n_gammas):
            sparse_sr = []
            nonzero_coords = np.nonzero(sr_maps[g_idx])
            for i, j in zip(nonzero_coords[0], nonzero_coords[1]):
                value = sr_maps[g_idx, i, j]
                sparse_sr.append(((i, j), value))
            sparse_representations.append(sparse_sr)

        sr_labels_per_timestep.append(sparse_representations)

    return sr_labels_per_timestep


def calculate_successor_representation(
    positions, query_time_t, grid_size=9, gammas=None, num_rollouts=1
):
    """
    Calculate successor representation from query time t onwards using the correct formula
    SRγ(s) = 1/Z × Σ(from Δt=0 to T-t) γ^Δt × I(s_{t+Δt} = s)

    Args:
        positions: List of positions visited in the episode
        query_time_t: Current timestep t to calculate SR from
        grid_size: Size of the grid (9x9 for AchieverBlocker)
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
        grid_size: Size of the grid (9x9 for AchieverBlocker)
        gammas: List of discount factors (defaults to [0.5, 0.9, 0.99])

    Returns:
        sr_labels_per_timestep: List of SR labels for each timestep
    """
    # Use the vectorized implementation for better performance
    return calculate_successor_representation_vectorized(positions, grid_size, gammas)


def calculate_key_door_rank(
    keys_collected, doors_opened, target_door_color, goal_rewards
):
    """
    Calculate the rank of keys and doors based on reward values

    Args:
        keys_collected: List of key colors collected in order
        doors_opened: List of door colors opened in order
        target_door_color: The target door color
        goal_rewards: Dictionary of color -> reward value

    Returns:
        key_door_rank: List of ranks [rank_key0, rank_key1, rank_key2, rank_key3]
                      where 1 is the highest reward, 4 is lowest reward
    """
    colors = ["red", "green", "blue", "yellow"]

    # Get reward values for each color
    reward_values = [goal_rewards.get(color, 0) for color in colors]

    # Create ranking: highest reward gets rank 1, lowest gets rank 4
    # Sort indices by reward value (descending order)
    sorted_indices = sorted(
        range(len(reward_values)), key=lambda i: reward_values[i], reverse=True
    )

    # Create ranking array
    key_door_rank = [0] * 4
    for rank, idx in enumerate(sorted_indices):
        key_door_rank[idx] = rank + 1

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


def env_to_maze_format(env, achiever_pos, blocker_pos):
    """
    Convert AchieverBlocker environment to maze format with both agents

    Args:
        env: AchieverBlocker environment
        achiever_pos: Current achiever position
        blocker_pos: Current blocker position

    Returns:
        maze_str: String representation of the maze with O (achiever) and X (blocker)
    """
    maze_lines = []
    width, height = env.width, env.height

    # Process each row
    for j in range(height):
        row = ""
        for i in range(width):
            cell = env.grid.get(i, j)

            if achiever_pos == (i, j):
                row += "O"  # Achiever
            elif blocker_pos == (i, j):
                row += "X"  # Blocker
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
    achiever_agent,
    blocker_agent,
    env,
    achiever_sr_labels_per_timestep,
    blocker_sr_labels_per_timestep,
    consumption_labels,
    key_door_rank,
    name="",
    base_dir="data/exp5",
    game_id=None,
    initial_maze_lines=None,
    gammas=None,
    goal_rewards=None,
    game_costs=None,
    trajectory_data=None,
):
    """
    Save game data with SR and consumption labels for 2-agent environment

    Args:
        achiever_agent: The achiever agent that played the game
        blocker_agent: The blocker agent that played the game
        env: The AchieverBlocker environment
        achiever_sr_labels_per_timestep: SR labels for achiever at each timestep
        blocker_sr_labels_per_timestep: SR labels for blocker at each timestep
        consumption_labels: Consumption labels for keys and doors
        key_door_rank: Rank of keys based on importance
        name: Output directory name
        base_dir: Base directory for saving
        game_id: Unique game ID for file naming
        trajectory_data: Dictionary containing trajectory information
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
        # Section 1: Environment
        f.write("MAZE:\n")
        if initial_maze_lines is not None:
            for line in initial_maze_lines:
                f.write(line + "\n")

        f.write(
            f"Trajectory length: {len(trajectory_data['achiever_positions']) - 1}\n"
        )

        # Pre-calculate blocker interaction result for final step
        blocker_inferred_goal = getattr(blocker_agent, "target_door_color", None)
        actual_target_door = getattr(env, "target_door_color", None)

        # Determine blocker interaction result
        if blocker_inferred_goal and actual_target_door:
            if blocker_inferred_goal == actual_target_door:
                final_blocker_interaction = "1"  # Success
            else:
                final_blocker_interaction = "0"  # Fail
        else:
            final_blocker_interaction = "X"  # No interaction

        # Write trajectory with 2-agent actions and interactions
        for i in range(len(trajectory_data["achiever_actions"])):
            achiever_pos = trajectory_data["achiever_positions"][i]
            blocker_pos = trajectory_data["blocker_positions"][i]
            achiever_action = trajectory_data["achiever_actions"][i]
            blocker_action = trajectory_data["blocker_actions"][i]

            # Determine interactions
            achiever_interaction = "X"  # Default no interaction
            blocker_interaction = "X"  # Default no interaction

            # Check for key pickup by achiever
            if hasattr(achiever_agent, "keys_collected_steps"):
                for step, key_color in achiever_agent.keys_collected_steps:
                    if step == i:
                        color_map = {
                            "red": "A",
                            "green": "B",
                            "blue": "C",
                            "yellow": "D",
                        }
                        achiever_interaction = color_map.get(key_color, "X")
                        break

            # Check for door opening by achiever
            if achiever_interaction == "X" and hasattr(
                achiever_agent, "doors_opened_steps"
            ):
                for step, door_color in achiever_agent.doors_opened_steps:
                    if step == i:
                        color_map = {
                            "red": "a",
                            "green": "b",
                            "blue": "c",
                            "yellow": "d",
                        }
                        achiever_interaction = color_map.get(door_color, "X")
                        break

            # For blocker: show interaction result only in the final step
            if i == len(trajectory_data["achiever_actions"]) - 1:
                blocker_interaction = final_blocker_interaction
            else:
                blocker_interaction = "X"

            f.write(
                f"[{achiever_pos[0]}, {achiever_pos[1]}][{blocker_pos[0]}, {blocker_pos[1]}] : {achiever_action},{blocker_action} : {achiever_interaction},{blocker_interaction}\n"
            )

        f.write("\n")

        # Section 2: Achiever
        f.write("Achiever:\n")
        f.write("Goal Consumed Rank : " + str(key_door_rank) + "\n")

        if goal_rewards is not None:
            colors = ["red", "green", "blue", "yellow"]
            reward_list = [goal_rewards.get(color, 0.0) for color in colors]
            f.write("Goal Rewards: " + ",".join(map(str, reward_list)) + "\n")
            f.write("Goal Rewards Sum: " + str(sum(reward_list)) + "\n")

        if game_costs is not None:
            colors = ["red", "green", "blue", "yellow"]
            cost_list = [game_costs.get(color, 0.0) for color in colors]
            f.write("Goal Costs: " + ",".join(map(str, cost_list)) + "\n")
            f.write("Goal Costs Sum: " + str(sum(cost_list)) + "\n")

        f.write(
            "Consumption Labels: "
            + ",".join(map(str, consumption_labels.tolist()))
            + "\n"
        )

        # Save achiever SR data per timestep
        if gammas is None:
            gammas = [0.5, 0.9, 0.99]

        f.write("SR_Data_Per_Timestep:\n")
        for t, sr_data_at_t in enumerate(achiever_sr_labels_per_timestep):
            f.write(f"Timestep_{t}:\n")
            for gamma_idx, gamma in enumerate(gammas):
                sparse_sr = (
                    sr_data_at_t[gamma_idx] if gamma_idx < len(sr_data_at_t) else []
                )
                sparse_str = ";".join(
                    [f"{pos[0]},{pos[1]}:{val}" for pos, val in sparse_sr]
                )
                f.write(f"SR_gamma_{gamma}: {sparse_str}\n")

        f.write("\n")

        # Section 3: Blocker
        f.write("Blocker:\n")

        # Get blocker's inferred goal and actual target door
        blocker_inferred_goal = getattr(blocker_agent, "target_door_color", None)
        actual_target_door = getattr(env, "target_door_color", None)

        # Write inferred goal
        if blocker_inferred_goal:
            f.write("Infer Goal: " + str(blocker_inferred_goal) + "\n")
        else:
            f.write("Infer Goal: X\n")  # No inference made

        # Determine interaction result
        # 1 = Success (correct door inference)
        # 0 = Fail (incorrect door inference)
        # X = No interaction (no inference made)
        if blocker_inferred_goal and actual_target_door:
            if blocker_inferred_goal == actual_target_door:
                interaction_result = "1"  # Success - correct inference
            else:
                interaction_result = "0"  # Fail - incorrect inference
        else:
            interaction_result = "X"  # No interaction

        f.write("Interaction: " + interaction_result + "\n")

        # Save blocker SR data per timestep
        f.write("SR_Data_Per_Timestep:\n")
        for t, sr_data_at_t in enumerate(blocker_sr_labels_per_timestep):
            f.write(f"Timestep_{t}:\n")
            for gamma_idx, gamma in enumerate(gammas):
                sparse_sr = (
                    sr_data_at_t[gamma_idx] if gamma_idx < len(sr_data_at_t) else []
                )
                sparse_str = ";".join(
                    [f"{pos[0]},{pos[1]}:{val}" for pos, val in sparse_sr]
                )
                f.write(f"SR_gamma_{gamma}: {sparse_str}\n")


def generate_game_rewards(config_dict, game_id):
    """
    Generate goal rewards for this game based on config settings
    """
    # Set seed for consistent reward generation for this game
    np.random.seed(config_dict["base_random_seed"] + game_id + 1000)

    reward_settings = config_dict.get("goal_reward_settings", {})

    if reward_settings.get("use_random_rewards", True):
        # Generate 4 random rewards from uniform [0,1]
        rewards = np.random.uniform(0, 1, 4).tolist()

        # Find the maximum value
        max_value = max(rewards)

        # Find all indices with the maximum value
        max_indices = [i for i, val in enumerate(rewards) if val == max_value]

        # If there are ties, randomly select one
        if len(max_indices) > 1:
            selected_index = np.random.choice(max_indices)
        else:
            selected_index = max_indices[0]

        # Set only the selected maximum to 1.0
        rewards[selected_index] = 1.0
    else:
        # Use default rewards
        rewards = reward_settings.get("default_rewards", [0.5, 1.0, 1.5, 1.0])

    # Map to color names
    colors = ["red", "green", "blue", "yellow"]
    return {colors[i]: rewards[i] for i in range(len(colors))}


def generate_game_costs(config_dict, game_id):
    """
    Generate random costs for this game based on config settings
    """
    # Set seed for consistent cost generation for this game
    np.random.seed(config_dict["base_random_seed"] + game_id + 2000)

    cost_settings = config_dict.get("cost_settings", {})

    if cost_settings.get("use_random_costs", True):
        # Generate random costs that sum to 1.0
        min_cost = cost_settings.get("min_cost", 0.05)
        max_cost = cost_settings.get("max_cost", 0.7)
        total_cost_sum = cost_settings.get("total_cost_sum", 1.0)

        # Generate 3 random split points between 0 and 1
        splits = np.random.uniform(0, 1, 3)
        splits = np.sort(splits)

        # Create 4 proportions from splits
        proportions = [
            splits[0],
            splits[1] - splits[0],
            splits[2] - splits[1],
            1 - splits[2],
        ]

        # Scale to total cost sum (1.0)
        costs = [prop * total_cost_sum for prop in proportions]

        # Ensure minimum cost constraint
        for i in range(len(costs)):
            if costs[i] < min_cost:
                costs[i] = min_cost

        # Rescale to maintain sum constraint
        current_sum = sum(costs)
        if current_sum != total_cost_sum:
            scale_factor = total_cost_sum / current_sum
            costs = [c * scale_factor for c in costs]

        # Ensure no cost exceeds maximum
        for i in range(len(costs)):
            if costs[i] > max_cost:
                costs[i] = max_cost

        # Final rescaling to maintain exact sum
        current_sum = sum(costs)
        if current_sum != total_cost_sum:
            scale_factor = total_cost_sum / current_sum
            costs = [c * scale_factor for c in costs]
    else:
        # Use default costs
        costs = cost_settings.get("default_costs", [0.1, 0.2, 0.3, 0.4])

    # Map to color names
    colors = ["red", "green", "blue", "yellow"]
    return {colors[i]: costs[i] for i in range(len(colors))}


def run_single_game(game_id, config_dict, save_dir):
    """
    Run a single AchieverBlocker game simulation

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

    # Generate random costs for this game
    game_costs = generate_game_costs(config_dict, game_id)

    # Create AchieverBlocker environment with custom preferences and costs
    env_size = config_dict["env_size"]
    if env_size == "5x5":
        env = AchieverBlocker5x5Env(
            preference=goal_rewards, cost=game_costs, max_steps=config_dict["max_steps"]
        )
    elif env_size == "9x9":
        env = AchieverBlocker9x9Env(
            preference=goal_rewards, cost=game_costs, max_steps=config_dict["max_steps"]
        )
    elif env_size == "11x11":
        env = AchieverBlocker11x11Env(
            preference=goal_rewards, cost=game_costs, max_steps=config_dict["max_steps"]
        )
    else:
        raise ValueError(f"Unknown environment size: {env_size}")

    # Seed environment before reset (following exp3 pattern)
    env.seed(config_dict["base_random_seed"] + game_id)

    # Reset environment
    obs, info = env.reset()

    # Create achiever agent
    achiever_type = config_dict["achiever_type"]
    if achiever_type == "astar":
        achiever_agent = AStarAgent(observability=config_dict["observability"])
    elif achiever_type == "value":
        agent_config = config_dict.get("achiever_configs", {}).get("value", {})
        achiever_agent = ValueAgent(
            observability=agent_config.get("observability", "full"),
            movement_cost=agent_config.get("movement_cost", 0.01),
            wall_penalty=agent_config.get("wall_penalty", 2.0),
            gamma=agent_config.get("gamma", 0.99),
            temperature=agent_config.get("temperature", 0.1),
        )
    elif achiever_type == "random":
        achiever_agent = AchieverRandomAgent(
            action_space=config_dict.get("achiever_action_space", 7),
            movement_prob=config_dict.get("random_movement_prob", 0.8),
        )
    else:
        raise ValueError(f"Unknown achiever type: {achiever_type}")

    # Create blocker agent
    blocker_type = config_dict["blocker_type"]
    if blocker_type == "random":
        blocker_agent = BlockerRandomAgent()
    elif blocker_type == "goal_direct":
        blocker_agent = BlockerGoalDirectAgent()
    else:
        raise ValueError(f"Unknown blocker type: {blocker_type}")

    # Reset agents
    achiever_agent.reset()
    blocker_agent.reset()

    # Set environment reference for blocker agent
    blocker_agent.set_env(env)

    # Initialize tracking
    achiever_positions = []
    blocker_positions = []
    achiever_actions = []
    blocker_actions = []
    keys_collected = []
    doors_opened = []
    keys_collected_steps = []
    doors_opened_steps = []

    # Save initial maze state (before agents start moving)
    initial_achiever_pos = tuple(env.achiever_pos)
    initial_blocker_pos = tuple(env.blocker_pos)
    initial_maze_lines = env_to_maze_format(
        env, initial_achiever_pos, initial_blocker_pos
    )

    # Record initial positions before any actions
    achiever_positions.append(initial_achiever_pos)
    blocker_positions.append(initial_blocker_pos)

    # Run game simulation
    step_count = 0
    max_steps = config_dict["max_steps"]
    total_achiever_reward = 0
    total_blocker_reward = 0

    while step_count < max_steps:
        # Get actions from both agents
        achiever_action = achiever_agent.get_action(obs)
        blocker_action = blocker_agent.get_action(obs)

        achiever_actions.append(achiever_action)
        blocker_actions.append(blocker_action)

        action_pair = (achiever_action, blocker_action)

        # Execute action
        obs, rewards, terminated, truncated, info = env.step(action_pair)

        # Record positions AFTER action execution
        current_achiever_pos = tuple(env.achiever_pos)
        current_blocker_pos = tuple(env.blocker_pos)
        achiever_positions.append(current_achiever_pos)
        blocker_positions.append(current_blocker_pos)

        done = terminated or truncated

        # Update rewards
        total_achiever_reward += rewards["achiever"]
        total_blocker_reward += rewards["blocker"]
        step_count += 1

        # Track key collection
        if hasattr(env, "achiever_keys"):
            current_keys = list(env.achiever_keys)
            if len(current_keys) > len(keys_collected):
                new_key = [k for k in current_keys if k not in keys_collected][0]
                keys_collected.append(new_key)
                keys_collected_steps.append((step_count - 1, new_key))

        # Check for door opening by reward
        if rewards["achiever"] >= 1.0:  # Door opening reward
            # Find which door was opened
            for x in range(env.grid.width):
                for y in range(env.grid.height):
                    obj = env.grid.get(x, y)
                    if obj and obj.type == "door" and obj.is_open:
                        if obj.color not in doors_opened:
                            doors_opened.append(obj.color)
                            doors_opened_steps.append((step_count - 1, obj.color))
                            break

        # Check if episode is done
        if done:
            break

    # Store tracking data in agents for save function
    achiever_agent.keys_collected_steps = keys_collected_steps
    achiever_agent.doors_opened_steps = doors_opened_steps

    # Calculate SR labels for each agent and timestep
    sr_gammas = config_dict.get("sr_gammas", [0.5, 0.9, 0.99])

    achiever_sr_labels_per_timestep = calculate_sr_labels_for_trajectory(
        achiever_positions, grid_size=env.width, gammas=sr_gammas
    )

    blocker_sr_labels_per_timestep = calculate_sr_labels_for_trajectory(
        blocker_positions, grid_size=env.width, gammas=sr_gammas
    )

    # Calculate consumption labels (keys and doors)
    consumption_labels = calculate_consumption_labels(keys_collected, doors_opened)

    # Calculate key/door rank based on target
    target_door_color = (
        env.target_door_color if hasattr(env, "target_door_color") else "yellow"
    )
    key_door_rank = calculate_key_door_rank(
        keys_collected, doors_opened, target_door_color, goal_rewards
    )

    # Prepare trajectory data
    trajectory_data = {
        "achiever_positions": achiever_positions,
        "blocker_positions": blocker_positions,
        "achiever_actions": achiever_actions,
        "blocker_actions": blocker_actions,
    }

    # Save game with labels
    save_game_with_labels(
        achiever_agent=achiever_agent,
        blocker_agent=blocker_agent,
        env=env,
        achiever_sr_labels_per_timestep=achiever_sr_labels_per_timestep,
        blocker_sr_labels_per_timestep=blocker_sr_labels_per_timestep,
        consumption_labels=consumption_labels,
        key_door_rank=key_door_rank,
        name="",
        base_dir=save_dir,
        game_id=game_id,
        initial_maze_lines=initial_maze_lines,
        gammas=sr_gammas,
        goal_rewards=goal_rewards,
        game_costs=game_costs,
        trajectory_data=trajectory_data,
    )

    return game_id


def generate_trajectories(
    config=None, random_seed=42, n_processes=None, test_data=False
):
    """
    Generate trajectories for AchieverBlocker environment in ToMnet format

    Args:
        config: Config object containing all parameters. If None, uses default values.
        random_seed: Random seed for environment generation
        n_processes: Number of parallel processes
        test_data: If True, saves data to test subdirectory
    """
    if config is None:
        config = Config()

    n_games = config.n_games

    print(
        f"Generating {n_games} AchieverBlocker trajectories with random seed: {random_seed}"
    )
    print(f"Achiever type: {config.achiever_type}")
    print(f"Blocker type: {config.blocker_type}")
    print(f"Environment size: {config.env_size}")

    # Create save directory based on environment name and agent types
    save_dir = config.get_data_path(is_test=test_data)

    # Override config save_dir with the new path
    config.save_dir = save_dir

    # Check if data already exists in the save directory
    if os.path.exists(config.save_dir):
        existing_files = [
            f
            for f in os.listdir(config.save_dir)
            if f.startswith("test") and f.endswith(".txt")
        ]
        if existing_files:
            print(f"Data already exists in {config.save_dir}")
            print(f"Found {len(existing_files)} existing trajectory files")
            print("Exiting to avoid overwriting existing data")
            print(
                "If you want to regenerate data, please delete the existing directory first"
            )
            return

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
        "achiever_type": config.achiever_type,
        "blocker_type": config.blocker_type,
        "base_random_seed": random_seed,
        # Agent-specific configs (complete configurations)
        "achiever_configs": config.achiever_configs,
        "blocker_configs": config.blocker_configs,
        # SR settings
        "sr_gammas": config.sr_settings["gammas"],
        "sr_grid_size": config.sr_settings["grid_size"],
        # Goal reward settings
        "goal_reward_settings": config.goal_reward_settings,
        # Cost settings
        "cost_settings": config.cost_settings,
        # Backward compatibility
        "random_movement_prob": config.achiever_configs.get("random", {}).get(
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
            if (i + 1) % 5000 == 0 or (i + 1) == n_games:
                print(f"Generated {i + 1}/{n_games} games")

        # Wait for all processes to complete
        pool.close()
        pool.join()

    print(f"Generated {n_games} games successfully using {n_processes} processes!")
    print(f"Data saved to: {config.save_dir}")
    print(
        f"Environment: {config.get_env_name()}, Achiever: {config.achiever_type}, Blocker: {config.blocker_type}"
    )


if __name__ == "__main__":
    # Set multiprocessing start method to avoid issues
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass  # Already set

    import argparse

    parser = argparse.ArgumentParser(
        description="Generate trajectories for AchieverBlocker environment in ToMnet format"
    )
    parser.add_argument(
        "--config_override",
        action="store_true",
        help="Override config with command line arguments",
    )
    parser.add_argument("--n_games", type=int, help="Number of games to generate")
    parser.add_argument(
        "--achiever_type",
        type=str,
        choices=["astar", "value", "random"],
        default=None,
        help="Type of achiever agent to use",
    )
    parser.add_argument(
        "--blocker_type",
        type=str,
        choices=["random", "goal_direct"],
        default=None,
        help="Type of blocker agent to use",
    )
    parser.add_argument(
        "--env_size",
        type=str,
        choices=["5x5", "9x9", "11x11"],
        default=None,
        help="AchieverBlocker environment size",
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
    parser.add_argument(
        "--test_data",
        action="store_true",
        help="Generate test data (saves to test subdirectory)",
    )

    args = parser.parse_args()

    config = Config()

    # Override config with command line arguments if specified
    if args.config_override:
        config.update_from_args(args)

    generate_trajectories(
        config,
        random_seed=args.random_seed,
        n_processes=args.n_processes,
        test_data=args.test_data,
    )
