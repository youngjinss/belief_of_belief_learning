import os
import sys
sys.path.append('..')

from environment import World
import agents as Agent
import numpy as np
from config import Config

"""
Experiment-specific trajectory generation for Experiment 2
Original code from https://github.com/Nik-Kras/ToMnet-N
@Author Nikita Krasnytskyi
@Modified by Filip Borowiak
Modified for SR and consumption prediction
"""


def calculate_successor_representation(trajectory, positions, grid_size=13, gammas=[0.5, 0.9, 0.99]):
    """
    Calculate successor representation for different discount factors
    
    Args:
        trajectory: List of actions taken
        positions: List of positions visited
        grid_size: Size of the grid
        gammas: List of discount factors
        
    Returns:
        sr_maps: Array of shape (len(gammas), grid_size, grid_size) with normalized SR
    """
    sr_maps = np.zeros((len(gammas), grid_size, grid_size))
    
    # Calculate SR for each discount factor
    for gamma_idx, gamma in enumerate(gammas):
        sr_map = np.zeros((grid_size, grid_size))
        
        # Count discounted future state visitations
        for t, pos in enumerate(positions):
            for future_t in range(t, len(positions)):
                discount = gamma ** (future_t - t)
                future_pos = positions[future_t]
                sr_map[future_pos[0], future_pos[1]] += discount
        
        # Normalize the SR map
        if sr_map.sum() > 0:
            sr_map = sr_map / sr_map.sum()
            
        sr_maps[gamma_idx] = sr_map
    
    return sr_maps


def calculate_consumption_labels(consumed_goals):
    """
    Calculate consumption labels for goals A, B, C, D
    
    Args:
        consumed_goals: String of consumed goals (e.g., "AB" or "D")
        
    Returns:
        consumption_vector: Binary vector of length 4 indicating which goals were consumed
    """
    consumption_vector = np.zeros(4, dtype=np.float32)
    
    if 'A' in consumed_goals:
        consumption_vector[0] = 1.0
    if 'B' in consumed_goals:
        consumption_vector[1] = 1.0
    if 'C' in consumed_goals:
        consumption_vector[2] = 1.0
    if 'D' in consumed_goals:
        consumption_vector[3] = 1.0
        
    return consumption_vector


def save_game_with_labels(agent, env, sr_maps, consumption_labels, name="experiment2", base_dir="../../data"):
    """
    Save game data with SR and consumption labels
    
    Args:
        agent: The agent that played the game
        env: The environment
        sr_maps: Successor representation maps
        consumption_labels: Consumption labels
        name: Output directory name
        base_dir: Base directory for saving
    """
    import re
    
    # Shouldn't be here, but it fixes the save of last goal :(
    if env.goal_picked != 0:
        agent.step_picked_goal.append(len(agent.trajectory) - 1)

    # Get the path to folder
    gf = os.path.join(base_dir, name)
    os.makedirs(gf, exist_ok=True)

    files = os.listdir(gf)
    r = re.compile(".*.txt")
    files = list(filter(r.match, files))

    # Choose the number for the new name
    max_number = 0
    for file in files:
        if max_number < int(file[4:-4]):
            max_number = int(file[4:-4])

    new_name_number = max_number + 1

    # Save the Game line by line
    new_file_path = os.path.join(gf, "test" + str(new_name_number) + ".txt")

    with open(new_file_path, "w") as f:
        f.write("Maze:\n")

        # Save the Maze (Walls, Goals, Player)
        wall_line = "#" * (env.height + 2)
        f.write(wall_line + "\n")
        for i in range(env.width):
            f.write(env.init_map[i] + "\n")
        f.write(wall_line + "\n")
        for i, goal in enumerate(env.consumed_goal):
            f.write("Goal Consumed #" + str(i + 1) + " : " + goal + "\n")
        f.write("Trajectory length: " + str(len(agent.trajectory)) + "\n")
        
        # Save consumption labels
        f.write("Consumption Labels: " + ",".join(map(str, consumption_labels.tolist())) + "\n")
        
        # Save SR maps (as flattened arrays for each gamma)
        for gamma_idx, gamma in enumerate([0.5, 0.9, 0.99]):
            sr_flat = sr_maps[gamma_idx].flatten()
            f.write(f"SR_gamma_{gamma}: " + ",".join(map(str, sr_flat.tolist())) + "\n")

        # Save moves
        for i in range(len(agent.trajectory)):
            msg = (
                str(agent.position_trajectory[i])
                + " : "
                + str(agent.trajectory[i])
                + " : "
            )
            picked_bool = False
            for j in range(len(agent.step_picked_goal)):
                if agent.step_picked_goal[j] == i:
                    msg = (
                        msg + env.consumed_goal[j]
                    )  # If consumed First goal here - mention which
                    picked_bool = True
            if not picked_bool:
                msg = msg + "X"  # If didn't consume - put X
            f.write(msg + "\n")


def generate_trajectories(config=None):
    """
    Generate trajectories for Experiment 2 using A* agents with SR and consumption labels
    
    Args:
        config: Config object containing all parameters. If None, uses default values.
    """
    if config is None:
        config = Config()
    
    n_games = config.n_games
    rows = config.rows
    cols = config.cols
    sight = config.sight
    max_moves = config.max_moves
    observability = config.observability
    output_dir = config.output_dir
    shuffle = config.shuffle
    no_walls = config.no_walls
    save_dir = config.data_dir

    env = World(
        row_size=rows,
        col_size=cols,
        max_moves_per_episode=max_moves,
        shuffle=shuffle,
        no_walls=no_walls,
    )

    # Create output directory
    full_output_dir = os.path.join(save_dir, output_dir)
    os.makedirs(full_output_dir, exist_ok=True)

    for i in range(n_games):
        if i % 1000 == 0:
            print(f"Generated {i}/{n_games} games")

        env.reset()
        # env.render()
        agent = Agent.AgentStar(env, sight, observability=observability)

        while True:
            agent.update_world_observation()
            # agent.render()

            action = agent.chose_action(observability=observability)
            # print(action)

            observe, terminate, goal_picked, reward = env.execute(action)

            if goal_picked:
                # print("You have picked a goal, reward = {}".format(reward))
                agent.on_pickup(reward)
                # added to terminate after picking one goal
                # terminate = True

            if terminate:
                # print("Game result: ", reward)
                break

            # input("Press the <Enter> key to continue...")

        # Calculate SR and consumption labels
        sr_maps = calculate_successor_representation(
            agent.trajectory, 
            agent.position_trajectory,
            grid_size=rows
        )
        
        consumption_labels = calculate_consumption_labels(env.consumed_goal)
        
        # Save game with additional SR and consumption data
        save_game_with_labels(
            agent=agent,
            env=env,
            sr_maps=sr_maps,
            consumption_labels=consumption_labels,
            name=output_dir,
            base_dir=save_dir
        )

    print(f"Generated {n_games} games successfully!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate trajectories for Experiment 2 using A* agents with SR and consumption labels"
    )
    parser.add_argument(
        "--config_override", action="store_true", 
        help="Override config with command line arguments"
    )
    parser.add_argument(
        "--n_games", type=int, help="Number of games to generate"
    )
    parser.add_argument("--rows", type=int, help="Grid height")
    parser.add_argument("--cols", type=int, help="Grid width")
    parser.add_argument("--sight", type=int, help="Agent sight radius")
    parser.add_argument(
        "--max_moves", type=int, help="Maximum moves per episode"
    )
    parser.add_argument(
        "--observability",
        type=str,
        choices=["full", "partial"],
        help="Observability type: full or partial",
    )
    parser.add_argument(
        "--output_dir", type=str, help="Directory to save games"
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Whether to shuffle goals after pickup",
    )
    parser.add_argument(
        "--no_walls",
        action="store_true",
        help="Whether to use empty maze",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        help="Base directory for saving data",
    )

    args = parser.parse_args()

    config = Config()
    
    # Override config with command line arguments if specified
    if args.config_override:
        if args.n_games is not None:
            config.n_games = args.n_games
        if args.rows is not None:
            config.rows = args.rows
        if args.cols is not None:
            config.cols = args.cols
        if args.sight is not None:
            config.sight = args.sight
        if args.max_moves is not None:
            config.max_moves = args.max_moves
        if args.observability is not None:
            config.observability = args.observability
        if args.output_dir is not None:
            config.output_dir = args.output_dir
        if args.shuffle:
            config.shuffle = args.shuffle
        if args.no_walls:
            config.no_walls = args.no_walls
        if args.save_dir is not None:
            config.data_dir = args.save_dir

    generate_trajectories(config)
