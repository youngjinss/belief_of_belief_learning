import os
import sys
import numpy as np
import multiprocessing as mp
from functools import partial
import warnings
import gc

# Suppress gymnasium registration warnings
warnings.filterwarnings(
    "ignore", message=".*Overriding environment.*already in registry.*"
)
warnings.filterwarnings("ignore", message=".*gym_minigrid has been deprecated.*")
warnings.filterwarnings(
    "ignore", message=".*environment creator metadata doesn't include `render_modes`.*"
)

# Add the lib directory to the path
lib_path = os.path.join(os.path.dirname(__file__), "..", "..", "lib")
sys.path.insert(0, lib_path)

# Add parent directory to path for imports
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Import seed utility
from script.exp6.utils import set_seed

from lib.env.gym_minigrid.envs.achiever_blocker import (
    AchieverBlocker5x5Env,
    AchieverBlocker9x9Env,
    AchieverBlocker11x11Env,
)
from lib.env.gym_minigrid.envs.keydoor import KeyDoorEnv
from script.exp6.achievers import (
    AStarAgent,
    RandomAgent as AchieverRandomAgent,
    Level0ValueAchiever,
    Level1ValueAchiever,
)
from script.exp6.blockers import (
    RandomAgent as BlockerRandomAgent,
    GoalDirectAgent as BlockerGoalDirectAgent,
    RandomlySelectedAgent as BlockerRandomlySelectedAgent,
    RuleBasedAgent as BlockerRuleBasedAgent,
    Level0ValueBlocker,
    Level1ValueBlocker,
)
from script.exp6.config import Config
from beliefrl.data.generation import (
    calculate_consumption_labels,
    calculate_key_door_rank,
    calculate_successor_representation,
    calculate_successor_representation_vectorized,
    env_to_maze_format,
    generate_game_costs,
    generate_game_rewards,
)

# Set seed using Config default value
config = Config()
set_seed(config.seed)

"""
Data generation for AchieverBlocker environment in ToMnet format
2-agent environment with achiever and blocker agents
"""


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


def env_to_maze_format_single_agent(env, agent_pos):
    """
    Convert KeyDoor environment to maze format with single agent

    Args:
        env: KeyDoor environment
        agent_pos: Current agent position

    Returns:
        maze_str: String representation of the maze with O (agent)
    """
    maze_lines = []
    width, height = env.grid.width, env.grid.height

    # Process each row
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
    achiever_agent,
    blocker_agent,
    env,
    achiever_sr_labels_per_timestep,
    blocker_sr_labels_per_timestep,
    consumption_labels,
    key_door_rank,
    name="",
    base_dir="data/exp6",
    game_id=None,
    initial_maze_lines=None,
    gammas=None,
    goal_rewards=None,
    game_costs=None,
    trajectory_data=None,
    achiever_type=None,
    blocker_type=None,
    config_dict=None,
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

        # Get environment and agent information for multi-attempt processing
        actual_target_door = getattr(env, "target_door_color", None)

        # Process trajectory to find all blocker break attempts
        blocker_break_attempts = []

        # Get all door positions from environment using direct storage access
        door_positions = getattr(env, "door_positions", [])

        # Only process blocker actions if blocker exists
        if blocker_agent is not None and "blocker_actions" in trajectory_data:
            for i in range(len(trajectory_data["blocker_actions"])):
                blocker_action = trajectory_data["blocker_actions"][i]
                blocker_pos = trajectory_data["blocker_positions"][i]

                # Check if blocker used break action (5) at a door position
                if blocker_action == 5:
                    # Check which door (if any) the blocker is breaking
                    blocker_pos_tuple = tuple(blocker_pos)
                    for door_color, door_pos in door_positions.items():
                        if door_pos and blocker_pos_tuple == tuple(door_pos):
                            blocker_break_attempts.append(
                                {
                                    "step": i,
                                    "door_color": door_color,
                                    "is_target_door": door_color == actual_target_door,
                                }
                            )
                            break

        # Determine overall blocker interaction result based on attempts
        if blocker_agent is None:
            # Single-agent mode - no blocker
            blocker_interaction_result = "N/A"
        elif not blocker_break_attempts:
            # No break attempts made
            blocker_interaction_result = "X"
        else:
            # Check if any attempt was successful (broke actual target door)
            successful_attempts = [
                attempt
                for attempt in blocker_break_attempts
                if attempt["is_target_door"]
            ]
            if successful_attempts:
                # Use the color of the target door that was successfully broken
                target_door_color = successful_attempts[0]["door_color"]
                color_map = {
                    "red": "a",
                    "green": "b",
                    "blue": "c",
                    "yellow": "d",
                }
                blocker_interaction_result = color_map.get(target_door_color, "X")
            else:
                # Use the color of the last door that was attempted (wrong door)
                last_attempt = blocker_break_attempts[-1]
                last_door_color = last_attempt["door_color"]
                color_map = {
                    "red": "a",
                    "green": "b",
                    "blue": "c",
                    "yellow": "d",
                }
                blocker_interaction_result = color_map.get(last_door_color, "X")

        # Write trajectory with agent actions and interactions
        for i in range(len(trajectory_data["achiever_actions"])):
            achiever_pos = trajectory_data["achiever_positions"][i]
            achiever_action = trajectory_data["achiever_actions"][i]
            
            if blocker_agent is not None:
                blocker_pos = trajectory_data["blocker_positions"][i]
                blocker_action = trajectory_data["blocker_actions"][i]
            else:
                blocker_pos = None
                blocker_action = None

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

            # For blocker: check if this step is a break attempt
            blocker_interaction = "X"  # Default no interaction
            for attempt in blocker_break_attempts:
                if attempt["step"] == i:
                    # This step is a break attempt - use door color
                    door_color = attempt["door_color"]
                    color_map = {
                        "red": "a",
                        "green": "b",
                        "blue": "c",
                        "yellow": "d",
                    }
                    blocker_interaction = color_map.get(door_color, "X")
                    break

            if blocker_agent is not None:
                f.write(
                    f"[{achiever_pos[0]}, {achiever_pos[1]}][{blocker_pos[0]}, {blocker_pos[1]}] : {achiever_action},{blocker_action} : {achiever_interaction},{blocker_interaction}\n"
                )
            else:
                # Single-agent format
                f.write(
                    f"[{achiever_pos[0]}, {achiever_pos[1]}] : {achiever_action} : {achiever_interaction}\n"
                )

        f.write("\n")

        # Section 2: Achiever
        f.write("Achiever:\n")

        # Write achiever type
        if achiever_type:
            # Get achiever type map from config_dict if available
            achiever_type_map = config_dict.get("achiever_type_map", {})
            achiever_type_id = achiever_type_map.get(achiever_type, -1)
            f.write(f"Type: {achiever_type_id}\n")
        else:
            f.write("Type: -1\n")  # Unknown type

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

        # Section 3: Blocker (only for multi-agent mode)
        if blocker_agent is not None:
            f.write("Blocker:\n")

            # Write blocker type
            if blocker_type:
                # Get blocker type map from config_dict if available
                blocker_type_map = config_dict.get("blocker_type_map", {})
                blocker_type_id = blocker_type_map.get(blocker_type, -1)
                f.write(f"Type: {blocker_type_id}\n")
            else:
                f.write("Type: -1\n")  # Unknown type

            # Get blocker's inferred goal and actual target door
            blocker_inferred_goal = getattr(blocker_agent, "target_door_color", None)
            actual_target_door = getattr(env, "target_door_color", None)

            # Write inferred goal as multi-hot vector based on break attempts and blocker's inferred goal
            # Format: [key_red, key_green, key_blue, key_yellow, door_red, door_green, door_blue, door_yellow]
            infer_goal_vector = [0, 0, 0, 0, 0, 0, 0, 0]

            # Mark doors that were attempted to be broken
            color_to_door_idx = {"red": 4, "green": 5, "blue": 6, "yellow": 7}
            for attempt in blocker_break_attempts:
                door_color = attempt["door_color"]
                if door_color in color_to_door_idx:
                    infer_goal_vector[color_to_door_idx[door_color]] = 1

            # Also mark the blocker's inferred goal if available
            if blocker_inferred_goal and blocker_inferred_goal in color_to_door_idx:
                infer_goal_vector[color_to_door_idx[blocker_inferred_goal]] = 1

            f.write("Infer Goal: " + ",".join(map(str, infer_goal_vector)) + "\n")

            # Use the pre-calculated blocker interaction result
            f.write("Interaction: " + blocker_interaction_result + "\n")

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


def run_single_game(game_id, config_dict, save_dir, blocker_type=None):
    """
    Run a single AchieverBlocker game simulation

    Args:
        game_id: Unique identifier for this game
        config_dict: Dictionary containing config parameters
        save_dir: Directory to save game data
        blocker_type: Specific blocker type to use for this game

    Returns:
        game_id: For tracking completion
    """
    # Set unique random seed for this game
    np.random.seed(config_dict["base_random_seed"] + game_id)

    # Generate goal rewards for this game
    goal_rewards = generate_game_rewards(config_dict, game_id)

    # Generate random costs for this game
    game_costs = generate_game_costs(config_dict, game_id)

    # Create environment based on single-agent or multi-agent mode
    env_size = config_dict["env_size"]
    
    # Check if this is single-agent mode (no blocker or blocker_type is None)
    is_single_agent = blocker_type is None or len(config_dict.get("blocker_types", {})) == 0
    
    if is_single_agent:
        # Create KeyDoor environment for single-agent mode
        size_map = {"5x5": 5, "9x9": 9, "11x11": 11}
        if env_size not in size_map:
            raise ValueError(f"Unknown environment size: {env_size}")
        
        env = KeyDoorEnv(
            size=size_map[env_size],
            preference=goal_rewards, 
            cost=game_costs, 
            max_steps=config_dict["max_steps"]
        )
    else:
        # Create AchieverBlocker environment for multi-agent mode
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
    reset_result = env.reset()
    if isinstance(reset_result, tuple) and len(reset_result) >= 2:
        obs, info = reset_result[0], reset_result[1]
    else:
        obs = reset_result
        info = {}

    # Create achiever agent
    achiever_type = config_dict["achiever_type"]
    if achiever_type == "astar":
        achiever_agent = AStarAgent(observability=config_dict["observability"])
    elif achiever_type == "lv0va":
        agent_config = config_dict.get("achiever_configs", {}).get("lv0va", {})
        achiever_agent = Level0ValueAchiever(
            observability=agent_config.get("observability", "full"),
            movement_cost=agent_config.get("movement_cost", 0.01),
            wall_penalty=agent_config.get("wall_penalty", 2.0),
            conflict_penalty=agent_config.get("conflict_penalty", 2.0),
            consumption_penalty=agent_config.get("consumption_penalty", 1.0),
            gamma=agent_config.get("gamma", 0.99),
            temperature=agent_config.get("temperature", 0.1),
            q_value_clip=agent_config.get("q_value_clip", 100),
        )
    elif achiever_type == "lv1va":
        agent_config = config_dict.get("achiever_configs", {}).get("lv1va", {})
        achiever_agent = Level1ValueAchiever(
            observability=agent_config.get("observability", "full"),
            movement_cost=agent_config.get("movement_cost", 0.01),
            wall_penalty=agent_config.get("wall_penalty", 2.0),
            conflict_penalty=agent_config.get("conflict_penalty", 2.0),
            consumption_penalty=agent_config.get("consumption_penalty", 1.0),
            gamma=agent_config.get("gamma", 0.99),
            temperature=agent_config.get("temperature", 0.1),
            q_value_clip=agent_config.get("q_value_clip", 100),
        )
    elif achiever_type == "value":
        agent_config = config_dict.get("achiever_configs", {}).get("value", {})
        achiever_agent = Level0ValueAchiever(
            observability=agent_config.get("observability", "full"),
            movement_cost=agent_config.get("movement_cost", 0.01),
            wall_penalty=agent_config.get("wall_penalty", 2.0),
            consumption_penalty=agent_config.get("consumption_penalty", 1.0),
            gamma=agent_config.get("gamma", 0.99),
            temperature=agent_config.get("temperature", 0.1),
            q_value_clip=agent_config.get("q_value_clip", 100),
        )
    elif achiever_type == "random":
        achiever_agent = AchieverRandomAgent(
            action_space=config_dict.get("achiever_action_space", 7),
            movement_prob=config_dict.get("random_movement_prob", 0.8),
        )
    else:
        raise ValueError(f"Unknown achiever type: {achiever_type}")

    # Create blocker agent (only for multi-agent mode)
    blocker_agent = None
    if not is_single_agent:
        # Use passed blocker_type if provided, otherwise use config default
        current_blocker_type = (
            blocker_type if blocker_type else list(config_dict["blocker_types"].keys())[0]
        )
        if current_blocker_type == "random":
            blocker_agent = BlockerRandomAgent()
        elif current_blocker_type == "goal_direct":
            blocker_agent = BlockerGoalDirectAgent()
        elif current_blocker_type == "lv0vb":
            blocker_config = config_dict.get("blocker_configs", {}).get("lv0vb", {})
            blocker_agent = Level0ValueBlocker(
                movement_cost=blocker_config.get("movement_cost", 0.01),
                wall_penalty=blocker_config.get("wall_penalty", 2.0),
                conflict_penalty=blocker_config.get("conflict_penalty", 2.0),
                gamma=blocker_config.get("gamma", 0.99),
                temperature=blocker_config.get("temperature", 0.1),
                q_value_clip=blocker_config.get("q_value_clip", 100),
            )
        elif current_blocker_type == "lv1vb":
            blocker_config = config_dict.get("blocker_configs", {}).get("lv1vb", {})
            blocker_agent = Level1ValueBlocker(
                movement_cost=blocker_config.get("movement_cost", 0.01),
                wall_penalty=blocker_config.get("wall_penalty", 2.0),
                conflict_penalty=blocker_config.get("conflict_penalty", 2.0),
                gamma=blocker_config.get("gamma", 0.99),
                temperature=blocker_config.get("temperature", 0.1),
                q_value_clip=blocker_config.get("q_value_clip", 100),
            )
        elif current_blocker_type == "randomly_selected":
            blocker_config = config_dict.get("blocker_configs", {}).get(
                "randomly_selected", {}
            )
            stay_probability = blocker_config.get("stay_probability", 0.7)
            blocker_agent = BlockerRandomlySelectedAgent(stay_probability=stay_probability)
        elif current_blocker_type == "rule_based":
            blocker_config = config_dict.get("blocker_configs", {}).get("rule_based", {})
            stay_probability = blocker_config.get("stay_probability", 0.7)
            blocker_agent = BlockerRuleBasedAgent(stay_probability=stay_probability)
        else:
            raise ValueError(f"Unknown blocker type: {current_blocker_type}")

    # Reset agents
    achiever_agent.reset()
    if blocker_agent is not None:
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
    if is_single_agent:
        # KeyDoor environment has only agent_pos
        initial_achiever_pos = tuple(env.agent_pos)
        initial_blocker_pos = None
        initial_maze_lines = env_to_maze_format_single_agent(env, initial_achiever_pos)
    else:
        # AchieverBlocker environment has achiever_pos and blocker_pos
        initial_achiever_pos = tuple(env.achiever_pos)
        initial_blocker_pos = tuple(env.blocker_pos)
        initial_maze_lines = env_to_maze_format(
            env, initial_achiever_pos, initial_blocker_pos
        )

    # Record initial positions before any actions
    achiever_positions.append(initial_achiever_pos)
    if not is_single_agent:
        blocker_positions.append(initial_blocker_pos)

    # Run game simulation
    step_count = 0
    max_steps = config_dict["max_steps"]
    total_achiever_reward = 0
    total_blocker_reward = 0

    while step_count < max_steps:
        # Get actions from agents
        achiever_action = achiever_agent.get_action(obs)
        
        if is_single_agent:
            # Single-agent mode: only achiever acts
            obs, reward, terminated, truncated, info = env.step(achiever_action)
            blocker_action = None
            rewards = reward
        else:
            # Multi-agent mode: both agents act
            blocker_action = blocker_agent.get_action(obs)
            action_pair = (achiever_action, blocker_action)
            obs, rewards, terminated, truncated, info = env.step(action_pair)

        done = terminated or truncated

        # Always record actions and positions - this ensures final actions are captured
        achiever_actions.append(achiever_action)
        if not is_single_agent:
            blocker_actions.append(blocker_action)

        # Record positions AFTER action execution
        if is_single_agent:
            current_achiever_pos = tuple(env.agent_pos)
            current_blocker_pos = None
            achiever_positions.append(current_achiever_pos)
        else:
            current_achiever_pos = tuple(env.achiever_pos)
            current_blocker_pos = tuple(env.blocker_pos)
            achiever_positions.append(current_achiever_pos)
            blocker_positions.append(current_blocker_pos)

        # Update rewards
        if is_single_agent:
            total_achiever_reward += rewards
            total_blocker_reward = 0
        else:
            total_achiever_reward += rewards["achiever"]
            total_blocker_reward += rewards["blocker"]

        # Track key collection
        if is_single_agent:
            current_keys = list(env.agent_keys) if hasattr(env, "agent_keys") else []
        else:
            current_keys = list(env.achiever_keys) if hasattr(env, "achiever_keys") else []
        
        # Check if new keys were collected (for both single and multi-agent)
        if len(current_keys) > len(keys_collected):
            new_key = [k for k in current_keys if k not in keys_collected][0]
            keys_collected.append(new_key)
            keys_collected_steps.append((step_count, new_key))

        # Check for door opening by reward
        # Handle both single-agent (scalar) and multi-agent (dict) reward formats
        achiever_reward = rewards["achiever"] if isinstance(rewards, dict) else rewards
        if achiever_reward >= 1.0:  # Door opening reward
            # Get the door that was just opened using environment tracking
            if env.last_door_opened and env.last_door_opened not in doors_opened:
                doors_opened.append(env.last_door_opened)
                doors_opened_steps.append((step_count, env.last_door_opened))

        step_count += 1

        # Check if episode is done
        if done:
            break

    # Store tracking data in agents for save function
    achiever_agent.keys_collected_steps = keys_collected_steps
    achiever_agent.doors_opened_steps = doors_opened_steps

    # Calculate SR labels for agents
    sr_gammas = config_dict.get("sr_gammas", [0.5, 0.9, 0.99])
    
    # Get grid size from environment (handle different env types)
    grid_size = env.grid.width if hasattr(env, 'grid') else env.width

    # Batch SR calculation for achiever
    achiever_sr_labels_per_timestep = calculate_sr_labels_for_trajectory(
        achiever_positions, grid_size=grid_size, gammas=sr_gammas
    )

    # Calculate for blocker only if it exists
    if blocker_agent is not None:
        blocker_sr_labels_per_timestep = calculate_sr_labels_for_trajectory(
            blocker_positions, grid_size=grid_size, gammas=sr_gammas
        )
    else:
        blocker_sr_labels_per_timestep = None

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
    if is_single_agent:
        trajectory_data = {
            "achiever_positions": achiever_positions,
            "achiever_actions": achiever_actions,
        }
    else:
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
        blocker_sr_labels_per_timestep=blocker_sr_labels_per_timestep if not is_single_agent else None,
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
        achiever_type=achiever_type,
        blocker_type=blocker_type if not is_single_agent else None,
        config_dict=config_dict,
    )

    # Memory cleanup after each game
    del env, achiever_agent, trajectory_data
    del achiever_positions, achiever_actions, achiever_sr_labels_per_timestep
    if blocker_agent is not None:
        del blocker_agent, blocker_positions, blocker_actions, blocker_sr_labels_per_timestep
    gc.collect()

    return game_id


def run_single_game_with_blocker(game_assignment, config_dict, save_dir):
    """
    Wrapper function for multiprocessing that unpacks game assignment

    Args:
        game_assignment: Tuple of (game_id, blocker_type)
        config_dict: Dictionary containing config parameters
        save_dir: Directory to save game data

    Returns:
        game_id: For tracking completion
    """
    game_id, blocker_type = game_assignment
    return run_single_game(game_id, config_dict, save_dir, blocker_type)


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

    # Check if single-agent mode
    is_single_agent = config.is_single_agent_mode()
    
    if is_single_agent:
        print(f"Generating KeyDoor (single-agent) trajectories with random seed: {random_seed}")
        print(f"Achiever types: {', '.join(config.achiever_types.keys())}")
        print(f"Environment size: {config.env_size}")
        
        # Generate data for each achiever type (no blockers)
        for achiever_type in config.achiever_types.keys():
            print(f"\nGenerating data for {achiever_type} achiever (single-agent mode)...")
            
            # Create save directory for this achiever type
            save_dir = config.get_data_path(
                achiever_type, None, is_test=test_data
            )
            os.makedirs(save_dir, exist_ok=True)
            
            # Generate trajectories for single-agent mode
            num_games = config.achiever_types[achiever_type]
            
            # If generating test data, use test proportion of games
            if test_data:
                num_games = int(num_games * config.get_test_data_proportion())
            
            generate_trajectories_for_combination(
                config=config,
                achiever_type=achiever_type,
                blocker_type=None,  # No blocker in single-agent mode
                num_games=num_games,
                save_dir=save_dir,
                random_seed=random_seed,
                n_processes=n_processes,
            )
    else:
        print(f"Generating AchieverBlocker (multi-agent) trajectories with random seed: {random_seed}")
        print(f"Achiever types: {', '.join(config.achiever_types.keys())}")
        print(f"Blocker types: {', '.join(config.blocker_types.keys())}")
        print(f"Environment size: {config.env_size}")

        # Generate data for each achiever-blocker combination
        for achiever_type in config.achiever_types.keys():
            for blocker_type in config.blocker_types.keys():
                print(
                    f"\nGenerating data for {achiever_type} achiever with {blocker_type} blocker..."
                )

                # Create save directory for this combination
                save_dir = config.get_data_path(
                    achiever_type, blocker_type, is_test=test_data
                )

                # Check if data already exists in the save directory
                if os.path.exists(save_dir):
                    existing_files = [
                        f
                        for f in os.listdir(save_dir)
                        if f.startswith("test") and f.endswith(".txt")
                    ]
                    if existing_files:
                        print(f"Data already exists in {save_dir}")
                        print(f"Found {len(existing_files)} existing trajectory files")
                        print(
                            "Skipping this combination to avoid overwriting existing data"
                        )
                        continue

                # Create output directory
                os.makedirs(save_dir, exist_ok=True)

                # Generate trajectories for this combination
                # Get games count for this specific combination
                achiever_games = config.achiever_types[achiever_type]
                blocker_games = config.blocker_types[blocker_type]
                combination_games = min(achiever_games, blocker_games)

                # If generating test data, use test proportion of games
                if test_data:
                    combination_games = int(combination_games * config.get_test_data_proportion())

                generate_trajectories_for_combination(
                    config=config,
                    achiever_type=achiever_type,
                    blocker_type=blocker_type,
                    num_games=combination_games,
                    save_dir=save_dir,
                    random_seed=random_seed,
                    n_processes=n_processes,
                )

    # Final summary
    if is_single_agent:
        print(f"Generated data for all {len(config.achiever_types)} achiever types (single-agent mode)")
        print(f"Total combinations: {len(config.achiever_types)}")
    else:
        print(
            f"Generated data for all {len(config.achiever_types)} achiever types and {len(config.blocker_types)} blocker types"
        )
        print(
            f"Total combinations: {len(config.achiever_types) * len(config.blocker_types)}"
        )


def generate_trajectories_for_combination(
    config, achiever_type, blocker_type, save_dir, random_seed, n_processes, num_games
):
    """
    Generate trajectories for a specific achiever-blocker combination
    """
    # Set number of processes (default to CPU count - 1, min 1)
    if n_processes is None:
        n_processes = max(1, mp.cpu_count() - 1)

    print(f"Using {n_processes} processes for parallel game generation")

    # Convert config to dictionary for multiprocessing
    config_dict = {
        "env_size": config.env_size,
        "max_steps": config.max_steps,
        "observability": config.observability,
        "achiever_type": achiever_type,
        "blocker_types": [blocker_type],  # Only use the specific blocker type
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
        # Type mappings
        "achiever_type_map": config.achiever_type_map,
        "blocker_type_map": config.blocker_type_map,
    }

    # Create game assignments (all games use the same blocker type)
    game_assignments = []
    for i in range(num_games):
        game_assignments.append((i, blocker_type))

    print(
        f"Generating {num_games} games for {achiever_type} achiever with {blocker_type} blocker"
    )

    # Create partial function with fixed arguments
    game_func = partial(
        run_single_game_with_blocker, config_dict=config_dict, save_dir=save_dir
    )

    # Run games in parallel with optimized process management
    with mp.Pool(processes=n_processes, maxtasksperchild=100) as pool:
        # Use imap for progress tracking with chunking for better performance
        chunk_size = max(1, num_games // (n_processes * 10))
        results = []

        for i, result in enumerate(
            pool.imap(game_func, game_assignments, chunksize=chunk_size)
        ):
            results.append(result)
            if (i + 1) % 5000 == 0 or (i + 1) == num_games:
                print(f"Generated {i + 1}/{num_games} games")

        # Wait for all processes to complete
        pool.close()
        pool.join()

    print(f"Generated {num_games} games successfully using {n_processes} processes!")
    print(f"Data saved to: {save_dir}")
    print(
        f"Environment: {config.get_env_name()}, Achiever: {achiever_type}, Blocker: {blocker_type}"
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
    parser.add_argument(
        "--achiever_type",
        type=str,
        choices=["lv0va", "lv1va", "astar", "value", "random"],
        default=None,
        help="Type of achiever agent to use (will be converted to list internally)",
    )
    parser.add_argument(
        "--blocker_type",
        type=str,
        choices=[
            "lv0vb",
            "lv1vb",
            "random",
            "goal_direct",
            "randomly_selected",
            "rule_based",
            "none",  # For single-agent mode
        ],
        default=None,
        help="Type of blocker agent to use (use 'none' for single-agent mode)",
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

    # Override config with command line arguments if specified
    if args.config_override:
        config.update_from_args(args)

    # Set main seed for reproducibility
    set_seed(args.random_seed)
    print(f"Set main random seed to {args.random_seed} for reproducibility")

    generate_trajectories(
        config,
        random_seed=args.random_seed,
        n_processes=args.n_processes,
        test_data=args.test_data,
    )
