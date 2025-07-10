import os
import re
import pickle
import matplotlib.pyplot as plt
import torch
import numpy as np

"""
Data processing and loading utilities for KeyDoor ToMnet implementation
Adapted from ToMnetF experiment5 for KeyDoor environment
@author: Based on ToMnetF implementation, adapted for KeyDoor
"""


class DataReader:
    def __init__(self, time_step=500, w=9, h=9, d=9, experiment_no=3):
        """
        Initialize DataReader for KeyDoor environment

        Args:
            time_step: Maximum trajectory length
            w: Maze width (9x9 for KeyDoor)
            h: Maze height (9x9 for KeyDoor)
            d: Maze depth (9 layers: 8 original + 1 heading direction)
            experiment_no: Experiment number (3 for KeyDoor)
        """
        self.EXPERIMENT_NO = experiment_no
        self.MAX_TRAJECTORY_SIZE = time_step
        self.MAZE_WIDTH = w
        self.MAZE_HEIGHT = h
        self.MAZE_DEPTH_TRAJECTORY = d

        # KeyDoor action space: [left, right, forward, pickup, drop, toggle, done]
        self.ACTION_SPACE = 7

        # Key and door mapping
        self.KEY_MAPPING = {"A": 0, "B": 1, "C": 2, "D": 3}  # red, green, blue, yellow
        self.DOOR_MAPPING = {"a": 0, "b": 1, "c": 2, "d": 3}  # red, green, blue, yellow

        # Object encoding for maze representation
        self.OBJECT_ENCODING = {
            "#": 0,  # Wall
            "-": 1,  # Empty space
            "A": 2,  # Red key
            "B": 3,  # Green key
            "C": 4,  # Blue key
            "D": 5,  # Yellow key
            "a": 6,  # Red door
            "b": 7,  # Green door
            "c": 8,  # Blue door
            "d": 9,  # Yellow door
            "O": 10,  # Agent
        }

    def check_ifCollected(self, interaction):
        """Check if an interaction represents key collection"""
        return interaction in ["A", "B", "C", "D"]

    def check_ifDoorOpened(self, interaction):
        """Check if an interaction represents door opening"""
        return interaction in ["a", "b", "c", "d"]

    def ReadOneGame(self, filename):
        """
        Read and parse one game file from KeyDoor environment

        Args:
            filename: Path to the game file

        Returns:
            Dictionary containing parsed game data
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Game file not found: {filename}")

        with open(filename, "r") as f:
            lines = f.readlines()

        # Initialize data structures
        maze = []
        goal_rank = None
        trajectory_length = 0
        goal_rewards = None
        goal_rewards_sum = None
        consumption_labels = None
        sr_data_per_timestep = {}
        actions = []
        positions = []
        interactions = []

        # Parse data
        parsing_sr_data = False
        current_timestep = None
        maze_section = True

        for idx, line in enumerate(lines):
            line = line.strip()

            # Parse maze section
            if line == "Maze:":
                maze_section = True
                continue

            if maze_section and line.startswith("Goal Consumed Rank"):
                maze_section = False
                # Parse goal rank
                rank_str = line.split(":")[1].strip()
                rank_str = rank_str.strip("[]")
                goal_rank = [int(r.strip()) for r in rank_str.split(",")]
                continue

            if maze_section and line and not line.startswith("Goal Consumed Rank"):
                # Parse maze row
                maze_row = []
                for char in line:
                    if char in self.OBJECT_ENCODING:
                        maze_row.append(self.OBJECT_ENCODING[char])
                    else:
                        maze_row.append(1)  # Default to empty space
                if maze_row:  # Only add non-empty rows
                    maze.append(maze_row)
                continue

            # Parse trajectory length
            if line.startswith("Trajectory length:"):
                trajectory_length = int(line.split(":")[1].strip())
                continue

            # Parse goal rewards
            if line.startswith("Goal Rewards:"):
                rewards_str = line.split(":")[1].strip()
                goal_rewards = [float(r) for r in rewards_str.split(",")]
                continue

            if line.startswith("Goal Rewards Sum:"):
                goal_rewards_sum = float(line.split(":")[1].strip())
                continue

            # Parse consumption labels
            if line.startswith("Consumption Labels:"):
                consumption_str = line.split(":")[1].strip()
                consumption_labels = np.array(
                    [float(val) for val in consumption_str.split(",")], dtype=np.float32
                )
                continue

            # Parse SR data
            if line.startswith("SR_Data_Per_Timestep"):
                parsing_sr_data = True
                current_timestep = None
                continue

            if parsing_sr_data and line.startswith("Timestep_"):
                current_timestep = int(line.replace("Timestep_", "").replace(":", ""))
                if current_timestep not in sr_data_per_timestep:
                    sr_data_per_timestep[current_timestep] = {}
                continue

            if current_timestep is not None and line.startswith("SR_gamma_"):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    gamma_str = parts[0].split("_")[-1]
                    sparse_str = parts[1].strip()

                    # Parse sparse format "x,y:value;x,y:value;..."
                    sparse_sr = []
                    if sparse_str:
                        for entry in sparse_str.split(";"):
                            if entry.strip():
                                pos_val = entry.split(":")
                                if len(pos_val) == 2:
                                    pos_coords = [int(x) for x in pos_val[0].split(",")]
                                    pos = tuple(pos_coords)
                                    val = float(pos_val[1])
                                    sparse_sr.append((pos, val))

                    sr_data_per_timestep[current_timestep][gamma_str] = sparse_sr
                continue

            # Parse trajectory data
            if line.startswith("["):
                parsing_sr_data = False
                # Parse trajectory lines: [x, y] : action : interaction
                pattern = r"\[(\d+),\s*(\d+)\]\s*:\s*(\d+)\s*:\s*(\w+)"
                match = re.match(pattern, line)

                if match:
                    pos_x = int(match.group(1))
                    pos_y = int(match.group(2))
                    action = int(match.group(3))
                    interaction = match.group(4)

                    positions.append((pos_x, pos_y))
                    actions.append(action)
                    interactions.append(interaction)

        # Convert maze to numpy array
        maze = np.array(maze, dtype=np.int32)

        # Determine consumed goal from goal rank
        consumed_goal = None
        if goal_rank:
            try:
                highest_rank_idx = goal_rank.index(1)  # Find index of rank 1 (highest)
                goal_symbols = ["A", "B", "C", "D"]
                consumed_goal = goal_symbols[highest_rank_idx]
            except ValueError:
                consumed_goal = "A"  # Default fallback

        # Create trajectory tensor
        trajectory_tensor = self._create_trajectory_tensor(
            maze, positions, actions, interactions, trajectory_length
        )

        # Create goal tensor
        goal_tensor = self._create_goal_tensor(consumed_goal)

        return {
            "maze": maze,
            "goal_rank": goal_rank,
            "trajectory_length": trajectory_length,
            "goal_rewards": goal_rewards,
            "goal_rewards_sum": goal_rewards_sum,
            "consumption_labels": consumption_labels,
            "sr_data_per_timestep": sr_data_per_timestep,
            "actions": actions,
            "positions": positions,
            "interactions": interactions,
            "consumed_goal": consumed_goal,
            "trajectory_tensor": trajectory_tensor,
            "goal_tensor": goal_tensor,
            "filename": filename,
        }

    def _create_trajectory_tensor(
        self, maze, positions, actions, interactions, trajectory_length
    ):
        """
        Create trajectory tensor from parsed data

        Args:
            maze: Maze layout
            positions: List of agent positions
            actions: List of actions taken
            interactions: List of interactions
            trajectory_length: Length of trajectory

        Returns:
            Trajectory tensor of shape (seq_len, channels, height, width)
        """
        seq_len = min(trajectory_length, self.MAX_TRAJECTORY_SIZE)
        trajectory = np.zeros(
            (seq_len, self.MAZE_DEPTH_TRAJECTORY, self.MAZE_HEIGHT, self.MAZE_WIDTH)
        )

        # Static layers (same for all timesteps)
        for t in range(seq_len):
            # Layer 0: Walls
            trajectory[t, 0] = (maze == 0).astype(np.float32)

            # Layer 1: Empty spaces
            trajectory[t, 1] = (maze == 1).astype(np.float32)

            # Layer 2-3: Keys (combine all keys in 2 layers - collected/uncollected)
            # For simplicity, put all keys in layer 2
            key_mask = np.isin(maze, [2, 3, 4, 5])
            trajectory[t, 2] = key_mask.astype(np.float32)

            # Layer 3-4: Doors (combine all doors in 2 layers - opened/closed)
            # For simplicity, put all doors in layer 3
            door_mask = np.isin(maze, [6, 7, 8, 9])
            trajectory[t, 3] = door_mask.astype(np.float32)

            # Layer 4-7: Can be used for other features or specific key/door types
            # Layer 4: Red objects (keys and doors)
            red_mask = np.isin(maze, [2, 6])
            trajectory[t, 4] = red_mask.astype(np.float32)

            # Layer 5: Green objects
            green_mask = np.isin(maze, [3, 7])
            trajectory[t, 5] = green_mask.astype(np.float32)

            # Layer 6: Blue objects
            blue_mask = np.isin(maze, [4, 8])
            trajectory[t, 6] = blue_mask.astype(np.float32)

            # Layer 7: Yellow objects
            yellow_mask = np.isin(maze, [5, 9])
            trajectory[t, 7] = yellow_mask.astype(np.float32)

        # Dynamic layers (agent position and actions)
        for t in range(min(len(positions), seq_len)):
            pos_x, pos_y = positions[t]
            if 0 <= pos_x < self.MAZE_WIDTH and 0 <= pos_y < self.MAZE_HEIGHT:
                # Agent position (overwrite static layers)
                trajectory[t, :, pos_y, pos_x] = 0  # Clear other objects
                trajectory[t, 1, pos_y, pos_x] = 1  # Agent in empty space layer

        return trajectory.astype(np.float32)

    def _create_goal_tensor(self, consumed_goal):
        """
        Create goal tensor from consumed goal

        Args:
            consumed_goal: Goal that was consumed ("A", "B", "C", or "D")

        Returns:
            Goal tensor (one-hot encoded)
        """
        goal_tensor = np.zeros(4, dtype=np.float32)
        if consumed_goal in self.KEY_MAPPING:
            goal_tensor[self.KEY_MAPPING[consumed_goal]] = 1.0
        return goal_tensor

    def ReadAllGames(self, data_dir):
        """
        Read all game files from directory

        Args:
            data_dir: Directory containing game files

        Returns:
            List of parsed game data
        """
        games = []

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        # Find all test files
        test_files = []
        for filename in os.listdir(data_dir):
            if filename.startswith("test") and filename.endswith(".txt"):
                test_files.append(os.path.join(data_dir, filename))

        test_files.sort()  # Sort for consistent ordering

        print(f"Found {len(test_files)} game files in {data_dir}")

        for filepath in test_files:
            try:
                game_data = self.ReadOneGame(filepath)
                games.append(game_data)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                continue

        print(f"Successfully loaded {len(games)} games")
        return games

    def save_processed_data(self, games, output_path):
        """
        Save processed game data to pickle file

        Args:
            games: List of processed game data
            output_path: Path to save pickle file
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "wb") as f:
            pickle.dump(games, f)

        print(f"Saved processed data to {output_path}")

    def load_processed_data(self, input_path):
        """
        Load processed game data from pickle file

        Args:
            input_path: Path to pickle file

        Returns:
            List of processed game data
        """
        with open(input_path, "rb") as f:
            games = pickle.load(f)

        print(f"Loaded {len(games)} games from {input_path}")
        return games

    def get_data_statistics(self, games):
        """
        Get statistics about the loaded data

        Args:
            games: List of game data

        Returns:
            Dictionary containing statistics
        """
        if not games:
            return {}

        trajectory_lengths = [game["trajectory_length"] for game in games]
        goal_rewards_sums = [
            game["goal_rewards_sum"] for game in games if game["goal_rewards_sum"]
        ]

        # Count goal distribution
        goal_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
        for game in games:
            goal = game["consumed_goal"]
            if goal in goal_counts:
                goal_counts[goal] += 1

        # Action distribution
        all_actions = []
        for game in games:
            all_actions.extend(game["actions"])

        action_counts = {i: all_actions.count(i) for i in range(7)}

        stats = {
            "num_games": len(games),
            "trajectory_lengths": {
                "min": min(trajectory_lengths),
                "max": max(trajectory_lengths),
                "mean": np.mean(trajectory_lengths),
                "std": np.std(trajectory_lengths),
            },
            "goal_rewards_sums": {
                "min": min(goal_rewards_sums) if goal_rewards_sums else 0,
                "max": max(goal_rewards_sums) if goal_rewards_sums else 0,
                "mean": np.mean(goal_rewards_sums) if goal_rewards_sums else 0,
                "std": np.std(goal_rewards_sums) if goal_rewards_sums else 0,
            },
            "goal_distribution": goal_counts,
            "action_distribution": action_counts,
            "maze_size": (self.MAZE_WIDTH, self.MAZE_HEIGHT),
            "max_trajectory_size": self.MAX_TRAJECTORY_SIZE,
        }

        return stats

    def visualize_statistics(self, stats, save_path=None):
        """
        Visualize data statistics

        Args:
            stats: Statistics dictionary
            save_path: Optional path to save plots
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Trajectory length distribution
        axes[0, 0].hist([stats["trajectory_lengths"]["mean"]], bins=20, alpha=0.7)
        axes[0, 0].set_title("Trajectory Length Distribution")
        axes[0, 0].set_xlabel("Length")
        axes[0, 0].set_ylabel("Frequency")

        # Goal distribution
        goals = list(stats["goal_distribution"].keys())
        counts = list(stats["goal_distribution"].values())
        axes[0, 1].bar(goals, counts, alpha=0.7)
        axes[0, 1].set_title("Goal Distribution")
        axes[0, 1].set_xlabel("Goal")
        axes[0, 1].set_ylabel("Count")

        # Action distribution
        actions = list(stats["action_distribution"].keys())
        action_counts = list(stats["action_distribution"].values())
        action_names = ["left", "right", "forward", "pickup", "drop", "toggle", "done"]
        axes[1, 0].bar([action_names[i] for i in actions], action_counts, alpha=0.7)
        axes[1, 0].set_title("Action Distribution")
        axes[1, 0].set_xlabel("Action")
        axes[1, 0].set_ylabel("Count")
        axes[1, 0].tick_params(axis="x", rotation=45)

        # Reward distribution
        axes[1, 1].hist([stats["goal_rewards_sums"]["mean"]], bins=20, alpha=0.7)
        axes[1, 1].set_title("Goal Rewards Sum Distribution")
        axes[1, 1].set_xlabel("Sum")
        axes[1, 1].set_ylabel("Frequency")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved statistics plot to {save_path}")

        plt.show()


if __name__ == "__main__":
    import argparse
    from config import Config

    parser = argparse.ArgumentParser(description="Process KeyDoor experimental data")

    # Basic parameters
    parser.add_argument(
        "--config_override",
        action="store_true",
        help="Enable command line parameter overrides",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/exp3",
        help="Directory containing game data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/exp3",
        help="Directory for output files",
    )

    # Data reader parameters
    parser.add_argument("--time_step", type=int, help="Maximum time steps")
    parser.add_argument("--maze_width", type=int, help="Maze width")
    parser.add_argument("--maze_height", type=int, help="Maze height")
    parser.add_argument("--maze_depth", type=int, help="Maze depth (channels)")
    parser.add_argument(
        "--experiment_no", type=int, default=3, help="Experiment number"
    )

    # Processing options
    parser.add_argument(
        "--visualize", action="store_true", help="Generate visualization plots"
    )
    parser.add_argument(
        "--save_processed", action="store_true", help="Save processed data to pickle"
    )
    parser.add_argument(
        "--stats_only", action="store_true", help="Only compute and display statistics"
    )

    args = parser.parse_args()

    # Create config and update from args if override is enabled
    config = Config()
    if args.config_override:
        config.update_from_args(args)

    # Get data configuration
    data_config = config.get_data_config()

    # Initialize data reader with config parameters
    reader = DataReader(
        time_step=data_config["time_step"],
        w=data_config["maze_width"],
        h=data_config["maze_height"],
        d=data_config["maze_depth"],
        experiment_no=args.experiment_no,
    )

    # Read all games from data directory
    games = reader.ReadAllGames(args.data_dir)

    # Get statistics
    stats = reader.get_data_statistics(games)
    print("Data Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Visualize statistics if requested
    if args.visualize:
        save_path = os.path.join(args.output_dir, "exp3_data_statistics.png")
        reader.visualize_statistics(stats, save_path=save_path)

    # Save processed data if requested
    if args.save_processed:
        output_path = os.path.join(args.output_dir, "processed_games.pkl")
        reader.save_processed_data(games, output_path)
