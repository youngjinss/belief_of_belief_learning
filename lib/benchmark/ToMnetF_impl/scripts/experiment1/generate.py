import os
import sys
sys.path.append('..')

from environment import World
import agents as Agent

"""
Experiment-specific trajectory generation for Experiment 1
Original code from https://github.com/Nik-Kras/ToMnet-N
@Author Nikita Krasnytskyi
@Modified by Filip Borowiak
"""


def generate_trajectories(
    n_games=10000,
    rows=13,
    cols=13,
    sight=3,
    max_moves=50,
    observability="full",
    output_dir="experiment1",
    shuffle=False,
    no_walls=False,
    save_dir="../../data",
):
    """
    Generate trajectories for Experiment 1 using A* agents

    Args:
        n_games: Number of games to generate
        rows: Grid height
        cols: Grid width
        sight: Agent sight radius
        max_moves: Maximum moves per episode
        observability: "full" or "partial"
        output_dir: Directory to save games
        shuffle: Whether to shuffle goals after pickup
        no_walls: Whether to use empty maze
        save_dir: Base directory for saving data
    """

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
        "--n_games", type=int, default=100000, help="Number of games to generate"
    )
    parser.add_argument("--rows", type=int, default=13, help="Grid height")
    parser.add_argument("--cols", type=int, default=13, help="Grid width")
    parser.add_argument("--sight", type=int, default=3, help="Agent sight radius")
    parser.add_argument(
        "--max_moves", type=int, default=50, help="Maximum moves per episode"
    )
    parser.add_argument(
        "--observability",
        type=str,
        default="full",
        choices=["full", "partial"],
        help="Observability type: full or partial",
    )
    parser.add_argument(
        "--output_dir", type=str, default="experiment1", help="Directory to save games"
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        default=False,
        help="Whether to shuffle goals after pickup",
    )
    parser.add_argument(
        "--no_walls",
        action="store_true",
        default=False,
        help="Whether to use empty maze",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="../../data",
        help="Base directory for saving data",
    )

    args = parser.parse_args()

    generate_trajectories(
        n_games=args.n_games,
        rows=args.rows,
        cols=args.cols,
        sight=args.sight,
        max_moves=args.max_moves,
        observability=args.observability,
        output_dir=args.output_dir,
        shuffle=args.shuffle,
        no_walls=args.no_walls,
        save_dir=args.save_dir,
    )
