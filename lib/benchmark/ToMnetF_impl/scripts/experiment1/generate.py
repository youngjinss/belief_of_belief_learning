import os
import sys
sys.path.append('..')

from environment import World
import agents as Agent
from config import Config

"""
Experiment-specific trajectory generation for Experiment 1
Original code from https://github.com/Nik-Kras/ToMnet-N
@Author Nikita Krasnytskyi
@Modified by Filip Borowiak
"""


def generate_trajectories(config=None):
    """
    Generate trajectories for Experiment 1 using A* agents
    
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

        agent.save_game(name=output_dir, base_dir=save_dir)

    print(f"Generated {n_games} games successfully!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate trajectories for Experiment 1 using A* agents"
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
