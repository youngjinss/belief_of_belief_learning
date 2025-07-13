import os
import re
import pickle
import random
import numpy as np
from typing import Dict, List, Tuple, Any

"""
Data processing for exp4 (AchieverBlocker) ToMnet training
Creates both achiever and blocker samples from trajectory files
Each trajectory file generates 2 samples (1 achiever + 1 blocker)
"""


class DataGenerator:
    def __init__(self, time_step=500, w=9, h=9, d=9, config=None):
        """
        Initialize DataGenerator for AchieverBlocker environment

        Args:
            time_step: Maximum trajectory length
            w: Maze width (9x9 for AchieverBlocker)
            h: Maze height (9x9 for AchieverBlocker)
            d: Maze depth (9 layers: 8 original + 1 heading direction)
            config: Config object for getting action spaces
        """
        self.MAX_TRAJECTORY_SIZE = time_step
        self.MAZE_WIDTH = w
        self.MAZE_HEIGHT = h
        self.MAZE_DEPTH = d

        # Action spaces from config
        if config is not None:
            self.ACHIEVER_ACTION_SPACE = config.model_config["achiever_action_space"]
            self.BLOCKER_ACTION_SPACE = config.model_config["blocker_action_space"]
        else:
            # Fallback defaults if no config provided
            try:
                from config import Config

                config_obj = Config()
                self.ACHIEVER_ACTION_SPACE = config_obj.model_config[
                    "achiever_action_space"
                ]
                self.BLOCKER_ACTION_SPACE = config_obj.model_config[
                    "blocker_action_space"
                ]
            except:
                self.ACHIEVER_ACTION_SPACE = (
                    7  # up, right, down, left, stay, pickup, toggle
                )
                self.BLOCKER_ACTION_SPACE = 6  # up, right, down, left, stay, break

        # Goal mappings for blocker inference
        self.COLOR_TO_LETTER = {"red": "A", "green": "B", "blue": "C", "yellow": "D"}
        self.LETTER_TO_IDX = {"A": 0, "B": 1, "C": 2, "D": 3}

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
            "O": 10,  # Achiever
            "X": 11,  # Blocker
        }

    def parse_trajectory_file(self, filename: str) -> Dict[str, Any]:
        """Parse a single trajectory file and extract data for both agents"""
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File not found: {filename}")

        with open(filename, "r") as f:
            lines = f.readlines()

        # Parse maze
        maze = []
        parsing_maze = False
        trajectory_length = 0
        achiever_data = {}
        blocker_data = {}
        trajectory_steps = []
        current_section = None  # Track which section we're parsing

        for line in lines:
            line = line.strip()

            # Parse maze section
            if line == "MAZE:":
                parsing_maze = True
                continue
            elif line.startswith("Trajectory length:"):
                parsing_maze = False
                trajectory_length = int(line.split(":")[1].strip())
                continue
            elif parsing_maze and line:
                # Parse any non-empty line as a maze row
                maze_row = []
                for char in line:
                    if char in self.OBJECT_ENCODING:
                        maze_row.append(self.OBJECT_ENCODING[char])
                    else:
                        maze_row.append(1)  # Default to empty space
                maze.append(maze_row)
                continue

            # Parse trajectory steps
            if line.startswith("["):
                # Parse trajectory line: [achiever_x, achiever_y][blocker_x, blocker_y] : achiever_action,blocker_action : achiever_interaction,blocker_interaction
                pattern = r"\[(\d+),\s*(\d+)\]\[(\d+),\s*(\d+)\]\s*:\s*(\d+),(\d+)\s*:\s*(\w+),(\w+)"
                match = re.match(pattern, line)

                if match:
                    achiever_x = int(match.group(1))
                    achiever_y = int(match.group(2))
                    blocker_x = int(match.group(3))
                    blocker_y = int(match.group(4))
                    achiever_action = int(match.group(5))
                    blocker_action = int(match.group(6))
                    achiever_interaction = match.group(7)
                    blocker_interaction = match.group(8)

                    trajectory_steps.append(
                        {
                            "achiever_pos": (achiever_x, achiever_y),
                            "blocker_pos": (blocker_x, blocker_y),
                            "achiever_action": achiever_action,
                            "blocker_action": blocker_action,
                            "achiever_interaction": achiever_interaction,
                            "blocker_interaction": blocker_interaction,
                        }
                    )
                continue

            # Section markers
            if line == "Achiever:":
                current_section = "achiever"
                continue
            elif line == "Blocker:":
                current_section = "blocker"
                continue

            # Parse SR data
            if current_section and line == "SR_Data_Per_Timestep:":
                # Initialize SR data storage
                sr_data_key = f"{current_section}_sr_data"
                if current_section == "achiever":
                    achiever_data["sr_data_per_timestep"] = {}
                else:
                    blocker_data["sr_data_per_timestep"] = {}
                continue

            # Parse timestep SR data
            if current_section and line.startswith("Timestep_"):
                timestep = int(line.replace("Timestep_", "").replace(":", ""))
                if current_section == "achiever":
                    achiever_data["sr_data_per_timestep"][timestep] = {}
                else:
                    blocker_data["sr_data_per_timestep"][timestep] = {}
                continue

            # Parse SR gamma values
            if current_section and line.startswith("SR_gamma_"):
                parts = line.split(": ")
                gamma_str = parts[0].replace("SR_gamma_", "")
                gamma = float(gamma_str)

                # Parse sparse SR data
                sparse_data = []
                if len(parts) > 1 and parts[1].strip():
                    entries = parts[1].strip().split(";")
                    for entry in entries:
                        if entry:
                            pos_val = entry.split(":")
                            if len(pos_val) == 2:
                                pos = pos_val[0].split(",")
                                x, y = int(pos[0]), int(pos[1])
                                value = float(pos_val[1])
                                sparse_data.append(((x, y), value))

                # Store in appropriate section
                if (
                    current_section == "achiever"
                    and "sr_data_per_timestep" in achiever_data
                ):
                    # Find the current timestep
                    timesteps = list(achiever_data["sr_data_per_timestep"].keys())
                    if timesteps:
                        current_timestep = max(timesteps)
                        achiever_data["sr_data_per_timestep"][current_timestep][
                            str(gamma)
                        ] = sparse_data
                elif (
                    current_section == "blocker"
                    and "sr_data_per_timestep" in blocker_data
                ):
                    timesteps = list(blocker_data["sr_data_per_timestep"].keys())
                    if timesteps:
                        current_timestep = max(timesteps)
                        blocker_data["sr_data_per_timestep"][current_timestep][
                            str(gamma)
                        ] = sparse_data
                continue

            # Parse achiever data
            if line.startswith("Goal Consumed Rank"):
                rank_str = line.split(":")[1].strip().strip("[]")
                achiever_data["goal_rank"] = [
                    int(r.strip()) for r in rank_str.split(",")
                ]
                continue
            elif line.startswith("Goal Rewards:"):
                rewards_str = line.split(":")[1].strip()
                achiever_data["goal_rewards"] = [
                    float(r) for r in rewards_str.split(",")
                ]
                continue
            elif line.startswith("Goal Rewards Sum:"):
                achiever_data["goal_rewards_sum"] = float(line.split(":")[1].strip())
                continue
            elif line.startswith("Consumption Labels:"):
                consumption_str = line.split(":")[1].strip()
                achiever_data["consumption_labels"] = np.array(
                    [float(val) for val in consumption_str.split(",")], dtype=np.float32
                )
                continue

            # Parse blocker data
            if line.startswith("Infer Goal:"):
                blocker_data["inferred_goal"] = line.split(":")[1].strip()
                continue
            elif line.startswith("Interaction:"):
                blocker_data["interaction"] = line.split(":")[1].strip()
                continue

        maze = np.array(maze, dtype=np.int32)

        return {
            "maze": maze,
            "trajectory_length": trajectory_length,
            "trajectory_steps": trajectory_steps,
            "achiever_data": achiever_data,
            "blocker_data": blocker_data,
            "filename": filename,
        }

    def create_achiever_sample(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create achiever sample (similar to exp3 structure)"""

        # Determine intended goal from goal rank (highest rank = 1)
        goal_rank = parsed_data["achiever_data"]["goal_rank"]
        intended_goal = None
        if goal_rank:
            try:
                highest_rank_idx = goal_rank.index(1)
                goal_symbols = ["A", "B", "C", "D"]
                intended_goal = goal_symbols[highest_rank_idx]
            except ValueError:
                intended_goal = "A"  # Default fallback

        # Determine consumed goal from consumption labels
        consumption_labels = parsed_data["achiever_data"]["consumption_labels"]
        consumed_goal = None
        if consumption_labels is not None:
            # Check which doors were opened (indices 4-7)
            door_indices = [4, 5, 6, 7]
            goal_symbols = ["A", "B", "C", "D"]
            for i, door_idx in enumerate(door_indices):
                if consumption_labels[door_idx] > 0:
                    consumed_goal = goal_symbols[i]
                    break

        if consumed_goal is None:
            consumed_goal = intended_goal

        # Create trajectory tensor
        trajectory_tensor = self._create_trajectory_tensor(
            parsed_data["maze"],
            parsed_data["trajectory_steps"],
            "achiever",
            parsed_data["trajectory_length"],
        )

        # Create goal tensor (one-hot encoding of intended goal)
        goal_tensor = self._create_goal_tensor(intended_goal)

        # Extract actions for achiever
        actions = [step["achiever_action"] for step in parsed_data["trajectory_steps"]]

        # Get SR data if available
        sr_data_per_timestep = parsed_data["achiever_data"].get(
            "sr_data_per_timestep", {}
        )

        return {
            "trajectory": trajectory_tensor,
            "actions": actions,
            "goal": goal_tensor,
            "consumption_labels": consumption_labels,
            "intended_goal": intended_goal,
            "consumed_goal": consumed_goal,
            "agent": "achiever",
            "sr_data_per_timestep": sr_data_per_timestep,
            "filename": parsed_data["filename"],
        }

    def create_blocker_sample(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create blocker sample with adapted structure"""

        # Get blocker's inferred goal
        inferred_goal_color = parsed_data["blocker_data"]["inferred_goal"]
        inferred_goal_letter = self.COLOR_TO_LETTER.get(inferred_goal_color, "A")

        # Get interaction result
        interaction_result = parsed_data["blocker_data"]["interaction"]

        # Create consumption labels for blocker (only door dimensions matter)
        # Set 1.0 for the inferred goal door, 0.0 for others
        consumption_labels = np.zeros(
            8, dtype=np.float32
        )  # [key_A, key_B, key_C, key_D, door_A, door_B, door_C, door_D]

        if inferred_goal_letter in self.LETTER_TO_IDX:
            door_idx = (
                4 + self.LETTER_TO_IDX[inferred_goal_letter]
            )  # Door indices are 4-7
            consumption_labels[door_idx] = 1.0

        # Determine consumed goal based on interaction result
        consumed_goal = None
        if interaction_result == "1":  # Success
            consumed_goal = inferred_goal_letter
        elif interaction_result == "0":  # Failure
            consumed_goal = inferred_goal_letter  # Still attempted this goal
        else:  # "X" - no interaction
            consumed_goal = None

        # Create trajectory tensor
        trajectory_tensor = self._create_trajectory_tensor(
            parsed_data["maze"],
            parsed_data["trajectory_steps"],
            "blocker",
            parsed_data["trajectory_length"],
        )

        # Create goal tensor (one-hot encoding of inferred goal)
        goal_tensor = self._create_goal_tensor(inferred_goal_letter)

        # Extract actions for blocker
        actions = [step["blocker_action"] for step in parsed_data["trajectory_steps"]]

        # Get SR data if available
        sr_data_per_timestep = parsed_data["blocker_data"].get(
            "sr_data_per_timestep", {}
        )

        return {
            "trajectory": trajectory_tensor,
            "actions": actions,
            "goal": goal_tensor,
            "consumption_labels": consumption_labels,
            "intended_goal": inferred_goal_letter,
            "consumed_goal": consumed_goal,
            "agent": "blocker",
            "sr_data_per_timestep": sr_data_per_timestep,
            "filename": parsed_data["filename"],
        }

    def _create_trajectory_tensor(
        self,
        maze: np.ndarray,
        trajectory_steps: List[Dict],
        agent_type: str,
        trajectory_length: int,
    ) -> np.ndarray:
        """Create trajectory tensor for specified agent"""

        seq_len = min(trajectory_length, self.MAX_TRAJECTORY_SIZE)
        trajectory = np.zeros(
            (seq_len, self.MAZE_DEPTH, self.MAZE_HEIGHT, self.MAZE_WIDTH)
        )

        # Static layers (same for all timesteps)
        for t in range(seq_len):
            # Layer 0: Walls
            trajectory[t, 0] = (maze == 0).astype(np.float32)

            # Layer 1: Empty spaces
            trajectory[t, 1] = (maze == 1).astype(np.float32)

            # Layer 2: Keys
            key_mask = np.isin(maze, [2, 3, 4, 5])
            trajectory[t, 2] = key_mask.astype(np.float32)

            # Layer 3: Doors
            door_mask = np.isin(maze, [6, 7, 8, 9])
            trajectory[t, 3] = door_mask.astype(np.float32)

            # Layer 4: Red objects
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

            # Layer 8: Agent heading direction (placeholder - set to 0 for south)
            trajectory[t, 8] = np.zeros((self.MAZE_HEIGHT, self.MAZE_WIDTH))

        # Dynamic layers (agent position)
        for t in range(min(len(trajectory_steps), seq_len)):
            step = trajectory_steps[t]

            if agent_type == "achiever":
                pos_x, pos_y = step["achiever_pos"]
            else:  # blocker
                pos_x, pos_y = step["blocker_pos"]

            if 0 <= pos_x < self.MAZE_WIDTH and 0 <= pos_y < self.MAZE_HEIGHT:
                # Clear other objects at agent position
                trajectory[t, :, pos_y, pos_x] = 0
                # Set agent in empty space layer
                trajectory[t, 1, pos_y, pos_x] = 1
                # Set heading direction (default to south=2)
                trajectory[t, 8, pos_y, pos_x] = 2

        return trajectory.astype(np.float32)

    def _create_goal_tensor(self, goal_letter: str) -> np.ndarray:
        """Create goal tensor (one-hot encoded)"""
        goal_tensor = np.zeros(4, dtype=np.float32)
        if goal_letter in self.LETTER_TO_IDX:
            goal_tensor[self.LETTER_TO_IDX[goal_letter]] = 1.0
        return goal_tensor

    def process_directory(self, data_dir: str) -> List[Dict[str, Any]]:
        """Process all trajectory files in directory and create samples"""

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        # Find all test files
        test_files = []
        for filename in os.listdir(data_dir):
            if filename.startswith("test") and filename.endswith(".txt"):
                test_files.append(os.path.join(data_dir, filename))

        test_files.sort()
        print(f"Found {len(test_files)} trajectory files in {data_dir}")

        all_samples = []

        for filepath in test_files:
            try:
                # Parse trajectory file
                parsed_data = self.parse_trajectory_file(filepath)

                # Create achiever sample
                achiever_sample = self.create_achiever_sample(parsed_data)
                all_samples.append(achiever_sample)

                # Create blocker sample
                blocker_sample = self.create_blocker_sample(parsed_data)
                all_samples.append(blocker_sample)

            except Exception as e:
                print(f"Error processing {filepath}: {e}")
                continue

        # Shuffle samples
        random.shuffle(all_samples)

        print(
            f"Generated {len(all_samples)} samples ({len(all_samples)//2} achiever + {len(all_samples)//2} blocker)"
        )
        return all_samples

    def save_processed_data(self, samples: List[Dict[str, Any]], output_path: str):
        """Save processed samples to pickle file"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "wb") as f:
            pickle.dump(samples, f)

        print(f"Saved {len(samples)} samples to {output_path}")

    def load_processed_data(self, input_path: str) -> List[Dict[str, Any]]:
        """Load processed samples from pickle file"""
        with open(input_path, "rb") as f:
            samples = pickle.load(f)

        print(f"Loaded {len(samples)} samples from {input_path}")
        return samples

    def get_statistics(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about processed samples"""

        if not samples:
            return {}

        # Count by agent type
        achiever_count = sum(1 for s in samples if s["agent"] == "achiever")
        blocker_count = sum(1 for s in samples if s["agent"] == "blocker")

        # Goal distribution
        goal_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
        for sample in samples:
            goal = sample["intended_goal"]
            if goal in goal_counts:
                goal_counts[goal] += 1

        # Agent distribution
        agent_counts = {"achiever": achiever_count, "blocker": blocker_count}

        # Trajectory lengths
        trajectory_lengths = [sample["trajectory"].shape[0] for sample in samples]

        stats = {
            "total_samples": len(samples),
            "agent_distribution": agent_counts,
            "goal_distribution": goal_counts,
            "trajectory_lengths": {
                "min": min(trajectory_lengths),
                "max": max(trajectory_lengths),
                "mean": np.mean(trajectory_lengths),
                "std": np.std(trajectory_lengths),
            },
            "maze_size": (self.MAZE_WIDTH, self.MAZE_HEIGHT),
            "trajectory_depth": self.MAZE_DEPTH,
        }

        return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate processed data for exp4 ToMnet training"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing trajectory files",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory for output files"
    )
    parser.add_argument(
        "--time_step", type=int, default=500, help="Maximum trajectory length"
    )
    parser.add_argument("--maze_width", type=int, default=9, help="Maze width")
    parser.add_argument("--maze_height", type=int, default=9, help="Maze height")
    parser.add_argument(
        "--maze_depth", type=int, default=9, help="Maze depth (channels)"
    )

    args = parser.parse_args()

    # Initialize data generator
    generator = DataGenerator(
        time_step=args.time_step,
        w=args.maze_width,
        h=args.maze_height,
        d=args.maze_depth,
    )

    # Process directory
    samples = generator.process_directory(args.data_dir)

    # Get statistics
    stats = generator.get_statistics(samples)
    print("\nData Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Save processed data
    output_path = os.path.join(args.output_dir, "processed_samples.pkl")
    generator.save_processed_data(samples, output_path)

    print(f"\nProcessing complete. Generated {len(samples)} samples from both agents.")
