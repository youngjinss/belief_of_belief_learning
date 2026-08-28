import os
import re
import pickle
import random
import numpy as np
from typing import Dict, List, Any
import sys
import gc
import multiprocessing as mp
from tqdm import tqdm

# Add the lib directory to the path
lib_path = os.path.join(os.path.dirname(__file__), "..", "..", "lib")
sys.path.insert(0, lib_path)

# Add parent directory to path for imports
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from script.exp8.config import Config

config = Config()


def process_file_batch_worker(file_batch_and_config: tuple) -> List[Dict[str, Any]]:
    """Standalone worker function for multiprocessing"""
    file_batch, is_single_agent, config = file_batch_and_config

    # Create a DataGenerator instance for this worker
    generator = DataGenerator(config=config)
    batch_samples = []

    for filepath in file_batch:
        # Parse trajectory file
        parsed_data = generator.parse_trajectory_file(filepath)

        # Create achiever sample
        achiever_sample = generator.create_achiever_sample(parsed_data)
        batch_samples.append(achiever_sample)

        # Create blocker sample only in multi-agent mode
        if not is_single_agent and not parsed_data.get("is_single_agent", False):
            blocker_sample = generator.create_blocker_sample(parsed_data)
            batch_samples.append(blocker_sample)

        # Clean up after each file processing
        if not is_single_agent and not parsed_data.get("is_single_agent", False):
            del parsed_data, achiever_sample, blocker_sample
        else:
            del parsed_data, achiever_sample

    # Collect garbage once per batch instead of per file
    gc.collect()
    return batch_samples


# Add lib to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from utils import set_seed
from config import Config

"""
Data processing for exp8 (AchieverBlocker) ToMnet training
Creates both achiever and blocker samples from trajectory files
Each trajectory file generates 2 samples (1 achiever + 1 blocker)
"""


class DataGenerator:
    def __init__(self, time_step=10, w=9, h=9, d=10, config=None):
        """
        Initialize DataGenerator for AchieverBlocker environment

        Args:
            time_step: Maximum trajectory length
            w: Maze width (9x9 for AchieverBlocker)
            h: Maze height (9x9 for AchieverBlocker)
            d: Maze depth (10 layers: 8 original + 1 self position + 1 opponent position)
            config: Config object for getting action spaces
        """
        # Store config for access by multiprocessing workers
        self.config = config
        # Use actual environment max_steps from config
        self.MAX_TRAJECTORY_SIZE = config.max_steps
        self.MAZE_WIDTH = w
        self.MAZE_HEIGHT = h
        self.MAZE_DEPTH = d

        # Pre-compile regex patterns for better performance
        # Multi-agent pattern: [x1, y1][x2, y2] : action1,action2 : interaction1,interaction2
        self.trajectory_pattern = re.compile(
            r"\[(\d+),\s*(\d+)\]\[(\d+),\s*(\d+)\]\s*:\s*(\d+),(\d+)\s*:\s*(\w+),(\w+)"
        )
        # Single-agent pattern: [x, y] : action : interaction
        self.single_agent_pattern = re.compile(
            r"\[(\d+),\s*(\d+)\]\s*:\s*(\d+)\s*:\s*(\w+)"
        )
        self.timestep_pattern = re.compile(r"Timestep_(\d+):")
        self.sr_gamma_pattern = re.compile(r"SR_gamma_([0-9.]+):\s*(.*)")
        self.goal_rank_pattern = re.compile(r"Goal Consumed Rank\s*:\s*\[(.*)\]")
        self.goal_rewards_pattern = re.compile(r"Goal Rewards:\s*(.*)")
        self.consumption_pattern = re.compile(r"Consumption Labels:\s*(.*)")

        # Store config for single-agent mode detection
        self.config = config

        # Action spaces from config
        if config is not None:
            self.ACHIEVER_ACTION_SPACE = config.model_config["achiever_action_space"]
            self.is_single_agent_mode = config.is_single_agent_mode()
            self.BLOCKER_ACTION_SPACE = config.model_config["blocker_action_space"]
            set_seed(config.seed)  # Set seed when config is provided
        else:
            # Fallback defaults if no config provided
            config_obj = Config()
            self.config = config_obj
            set_seed(config_obj.seed)  # Set seed when Config is created
            self.ACHIEVER_ACTION_SPACE = config_obj.model_config[
                "achiever_action_space"
            ]
            self.BLOCKER_ACTION_SPACE = config_obj.model_config["blocker_action_space"]
            self.is_single_agent_mode = config_obj.is_single_agent_mode()

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
        is_single_agent = False  # Track if this is a single-agent trajectory

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

            # Parse trajectory steps using pre-compiled regex
            if line.startswith("["):
                # Try multi-agent pattern first
                match = self.trajectory_pattern.match(line)
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
                else:
                    # Try single-agent pattern
                    single_match = self.single_agent_pattern.match(line)
                    if single_match:
                        is_single_agent = True  # Mark as single-agent trajectory
                        achiever_x = int(single_match.group(1))
                        achiever_y = int(single_match.group(2))
                        achiever_action = int(single_match.group(3))
                        achiever_interaction = single_match.group(4)

                        trajectory_steps.append(
                            {
                                "achiever_pos": (achiever_x, achiever_y),
                                "blocker_pos": None,
                                "achiever_action": achiever_action,
                                "blocker_action": None,
                                "achiever_interaction": achiever_interaction,
                                "blocker_interaction": None,
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

            # Parse timestep SR data using pre-compiled regex
            if current_section:
                timestep_match = self.timestep_pattern.match(line)
                if timestep_match:
                    timestep = int(timestep_match.group(1))
                    if current_section == "achiever":
                        achiever_data["sr_data_per_timestep"][timestep] = {}
                    else:
                        blocker_data["sr_data_per_timestep"][timestep] = {}
                    continue

            # Parse SR gamma values using pre-compiled regex
            if current_section:
                sr_gamma_match = self.sr_gamma_pattern.match(line)
                if sr_gamma_match:
                    gamma = float(sr_gamma_match.group(1))
                    sparse_data_str = (
                        sr_gamma_match.group(2) if sr_gamma_match.group(2) else ""
                    )

                    # Parse sparse SR data
                    sparse_data = []
                    if sparse_data_str.strip():
                        entries = sparse_data_str.strip().split(";")
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

            # Parse achiever data using pre-compiled regex
            if line.startswith("Type:") and current_section == "achiever":
                achiever_data["type"] = int(line.split(":")[1].strip())
                continue

            goal_rank_match = self.goal_rank_pattern.match(line)
            if goal_rank_match:
                rank_str = goal_rank_match.group(1)
                achiever_data["goal_rank"] = [
                    int(r.strip()) for r in rank_str.split(",")
                ]
                continue

            goal_rewards_match = self.goal_rewards_pattern.match(line)
            if goal_rewards_match:
                rewards_str = goal_rewards_match.group(1)
                achiever_data["goal_rewards"] = [
                    float(r) for r in rewards_str.split(",")
                ]
                continue

            if line.startswith("Goal Rewards Sum:"):
                achiever_data["goal_rewards_sum"] = float(line.split(":")[1].strip())
                continue

            consumption_match = self.consumption_pattern.match(line)
            if consumption_match:
                consumption_str = consumption_match.group(1)
                achiever_data["consumption_labels"] = np.array(
                    [float(val) for val in consumption_str.split(",")], dtype=np.float32
                )
                continue

            # Parse blocker data
            if line.startswith("Type:"):
                blocker_data["type"] = int(line.split(":")[1].strip())
                continue
            elif line.startswith("Infer Goal:"):
                infer_goal_str = line.split(":")[1].strip()
                if "," in infer_goal_str:
                    # Multi-hot vector format: [key_red, key_green, key_blue, key_yellow, door_red, door_green, door_blue, door_yellow]
                    infer_goal_vector = [int(x) for x in infer_goal_str.split(",")]
                    blocker_data["inferred_goal_vector"] = infer_goal_vector

                    # For backward compatibility, also store the first door that was inferred
                    door_colors = ["red", "green", "blue", "yellow"]
                    door_indices = [4, 5, 6, 7]  # door positions in the vector
                    inferred_door = None
                    for i, door_idx in enumerate(door_indices):
                        if infer_goal_vector[door_idx] == 1:
                            inferred_door = door_colors[i]
                            break
                    blocker_data["inferred_goal"] = (
                        inferred_door if inferred_door else "X"
                    )
                else:
                    # Legacy format (single color or X)
                    blocker_data["inferred_goal"] = infer_goal_str
                    # Convert to multi-hot vector for consistency
                    infer_goal_vector = [0, 0, 0, 0, 0, 0, 0, 0]
                    if infer_goal_str in ["red", "green", "blue", "yellow"]:
                        color_to_door_idx = {
                            "red": 4,
                            "green": 5,
                            "blue": 6,
                            "yellow": 7,
                        }
                        infer_goal_vector[color_to_door_idx[infer_goal_str]] = 1
                    blocker_data["inferred_goal_vector"] = infer_goal_vector
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
            "is_single_agent": is_single_agent,
        }

    def create_achiever_sample(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create achiever sample (similar to exp3 structure)"""

        # Determine intended goal from goal rank (highest rank = 1)
        goal_rank = parsed_data["achiever_data"]["goal_rank"]
        intended_goal = None
        if goal_rank:
            highest_rank_idx = goal_rank.index(1)
            goal_symbols = ["A", "B", "C", "D"]
            intended_goal = goal_symbols[highest_rank_idx]

        # Create consumption labels using slicing approach - track all interactions during trajectory
        consumption_labels = np.zeros(
            8, dtype=np.float32
        )  # [key_A, key_B, key_C, key_D, door_A, door_B, door_C, door_D]

        # Extract all achiever interactions at once
        achiever_interactions = [
            step["achiever_interaction"] for step in parsed_data["trajectory_steps"]
        ]

        # Vectorized consumption tracking - accumulate all interactions throughout the trajectory
        door_color_to_idx = {"a": 4, "b": 5, "c": 6, "d": 7}  # door indices
        key_color_to_idx = {"A": 0, "B": 1, "C": 2, "D": 3}  # key indices

        # Create boolean masks for different interaction types
        interaction_array = np.array(achiever_interactions)

        # Track key pickups (vectorized) - any key pickup throughout trajectory
        for key_letter, key_idx in key_color_to_idx.items():
            key_mask = interaction_array == key_letter
            if np.any(key_mask):
                consumption_labels[key_idx] = 1.0

        # Track door interactions (vectorized) - any door interaction throughout trajectory
        consumed_goal = None
        door_to_goal_map = {"a": "A", "b": "B", "c": "C", "d": "D"}
        for door_letter, door_idx in door_color_to_idx.items():
            door_mask = interaction_array == door_letter
            if np.any(door_mask):
                consumption_labels[door_idx] = 1.0
                # Find the last door interaction for consumed goal
                consumed_goal = door_to_goal_map[door_letter]

        # If no door interaction occurred, use intended goal
        if consumed_goal is None:
            consumed_goal = intended_goal

        # Create self states tensor
        self_states_tensor = self._create_state_tensor(
            parsed_data["maze"],
            parsed_data["trajectory_steps"],
            "achiever",
            parsed_data["trajectory_length"],
            observability=self.config.observability,
            partial_view_size=self.config.get_partial_view_size(),
        )

        # Create opponent states tensor (perspective from opponent's view)
        oppo_states_tensor = None
        oppo_actions = None
        if not parsed_data.get("is_single_agent", False):
            oppo_states_tensor = self._create_state_tensor(
                parsed_data["maze"],
                parsed_data["trajectory_steps"],
                "blocker",  # Opponent is blocker
                parsed_data["trajectory_length"],
                observability=self.config.observability,
                partial_view_size=self.config.get_partial_view_size(),
            )
            # Extract opponent actions
            oppo_actions = [
                step["blocker_action"]
                for step in parsed_data["trajectory_steps"]
                if step["blocker_action"] is not None
            ]

        # Create goal tensor (one-hot encoding of intended goal)
        goal_tensor = self._create_goal_tensor(intended_goal)

        # Extract self actions for achiever
        self_actions = [
            step["achiever_action"] for step in parsed_data["trajectory_steps"]
        ]

        # Get SR data if available
        sr_data_per_timestep = parsed_data["achiever_data"].get(
            "sr_data_per_timestep", {}
        )

        # Get achiever type from parsed data
        achiever_type = parsed_data["achiever_data"].get("type", -1)

        return {
            "self_states": self_states_tensor,
            "self_actions": self_actions,
            "goal": goal_tensor,
            "consumption_labels": consumption_labels,
            "intended_goal": intended_goal,
            "consumed_goal": consumed_goal,
            "agent": "achiever",
            "type": achiever_type,
            "sr_data_per_timestep": sr_data_per_timestep,
            "filename": parsed_data["filename"],
            "oppo_states": oppo_states_tensor,
            "oppo_actions": oppo_actions,
        }

    def create_blocker_sample(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create blocker sample with adapted structure"""

        # Get blocker's inferred goal (use multi-hot vector if available)
        inferred_goal_vector = parsed_data["blocker_data"].get(
            "inferred_goal_vector", None
        )
        inferred_goal_color = parsed_data["blocker_data"]["inferred_goal"]
        inferred_goal_letter = self.COLOR_TO_LETTER.get(inferred_goal_color, "A")

        # Get blocker type (0 - randomly select, 1 - rule-based)
        blocker_type = parsed_data["blocker_data"].get("type", 0)

        # Get interaction result
        interaction_result = parsed_data["blocker_data"]["interaction"]

        # Create consumption labels for blocker using slicing approach - track all interactions during trajectory

        consumption_labels = np.zeros(
            8, dtype=np.float32
        )  # [key_A, key_B, key_C, key_D, door_A, door_B, door_C, door_D]

        # Extract all blocker interactions at once
        blocker_interactions = [
            step["blocker_interaction"] for step in parsed_data["trajectory_steps"]
        ]

        # Vectorized consumption tracking - accumulate all interactions throughout the trajectory
        door_color_to_idx = {"a": 4, "b": 5, "c": 6, "d": 7}  # door indices

        # Create boolean masks for door interactions
        interaction_array = np.array(blocker_interactions)

        # Track door interactions (vectorized) - any door interaction throughout trajectory
        consumed_goal = None
        door_to_goal_map = {"a": "A", "b": "B", "c": "C", "d": "D"}
        for door_letter, door_idx in door_color_to_idx.items():
            door_mask = interaction_array == door_letter
            if np.any(door_mask):
                consumption_labels[door_idx] = 1.0
                # Find the last door interaction for consumed goal
                consumed_goal = door_to_goal_map[door_letter]

        # If no door interaction occurred, consumed_goal remains None
        # Note: Blocker may not have a consumed goal if no door interactions occurred

        # Create self states tensor
        self_states_tensor = self._create_state_tensor(
            parsed_data["maze"],
            parsed_data["trajectory_steps"],
            "blocker",
            parsed_data["trajectory_length"],
            observability=self.config.observability,
            partial_view_size=self.config.get_partial_view_size(),
        )

        # Create opponent states tensor (perspective from opponent's view)
        oppo_states_tensor = None
        oppo_actions = None
        if not parsed_data.get("is_single_agent", False):
            oppo_states_tensor = self._create_state_tensor(
                parsed_data["maze"],
                parsed_data["trajectory_steps"],
                "achiever",  # Opponent is achiever
                parsed_data["trajectory_length"],
                observability=self.config.observability,
                partial_view_size=self.config.get_partial_view_size(),
            )
            # Extract opponent actions
            oppo_actions = [
                step["achiever_action"]
                for step in parsed_data["trajectory_steps"]
                if step["achiever_action"] is not None
            ]

        # Create goal tensor (one-hot encoding of inferred goal)
        goal_tensor = self._create_goal_tensor(inferred_goal_letter)

        # Extract self actions for blocker
        self_actions = [
            step["blocker_action"] for step in parsed_data["trajectory_steps"]
        ]

        # Get SR data if available
        sr_data_per_timestep = parsed_data["blocker_data"].get(
            "sr_data_per_timestep", {}
        )

        return {
            "self_states": self_states_tensor,
            "self_actions": self_actions,
            "goal": goal_tensor,
            "consumption_labels": consumption_labels,
            "intended_goal": inferred_goal_letter,
            "consumed_goal": consumed_goal,
            "agent": "blocker",
            "type": blocker_type,
            "sr_data_per_timestep": sr_data_per_timestep,
            "inferred_goal_vector": inferred_goal_vector,
            "filename": parsed_data["filename"],
            "oppo_states": oppo_states_tensor,
            "oppo_actions": oppo_actions,
        }

    def _create_state_tensor(
        self,
        maze: np.ndarray,
        trajectory_steps: List[Dict],
        agent_type: str,
        trajectory_length: int,
        observability: str = "full",
        partial_view_size: int = 7,
    ) -> np.ndarray:
        """Create self states tensor for specified agent using vectorized operations

        Channel structure (10 channels total):
        - Channel 0: Wall positions
        - Channel 1: Empty space
        - Channel 2: Any key present
        - Channel 3: Any door present
        - Channel 4: Red objects (keys/doors)
        - Channel 5: Green objects
        - Channel 6: Blue objects
        - Channel 7: Yellow objects
        - Channel 8: Self position (agent whose action is being predicted)
        - Channel 9: Opponent position (0 if single-agent mode)
        """

        seq_len = min(trajectory_length, self.MAX_TRAJECTORY_SIZE)
        self_states = np.zeros(
            (seq_len, self.MAZE_DEPTH, self.MAZE_HEIGHT, self.MAZE_WIDTH),
            dtype=np.float32,
        )

        # Vectorized static layer creation (same for all timesteps)
        # Pre-compute all static masks once
        wall_mask = (maze == 0).astype(np.float32)
        empty_mask = (maze == 1).astype(np.float32)
        key_mask = np.isin(maze, [2, 3, 4, 5]).astype(np.float32)
        door_mask = np.isin(maze, [6, 7, 8, 9]).astype(np.float32)
        red_mask = np.isin(maze, [2, 6]).astype(np.float32)
        green_mask = np.isin(maze, [3, 7]).astype(np.float32)
        blue_mask = np.isin(maze, [4, 8]).astype(np.float32)
        yellow_mask = np.isin(maze, [5, 9]).astype(np.float32)

        # Initialize position masks for self and opponent
        self_mask = np.zeros((self.MAZE_HEIGHT, self.MAZE_WIDTH), dtype=np.float32)
        opponent_mask = np.zeros((self.MAZE_HEIGHT, self.MAZE_WIDTH), dtype=np.float32)

        # Broadcast static layers to all timesteps at once
        self_states[:, 0] = wall_mask[np.newaxis, :, :]
        self_states[:, 1] = empty_mask[np.newaxis, :, :]
        self_states[:, 2] = key_mask[np.newaxis, :, :]
        self_states[:, 3] = door_mask[np.newaxis, :, :]
        self_states[:, 4] = red_mask[np.newaxis, :, :]
        self_states[:, 5] = green_mask[np.newaxis, :, :]
        self_states[:, 6] = blue_mask[np.newaxis, :, :]
        self_states[:, 7] = yellow_mask[np.newaxis, :, :]
        self_states[:, 8] = self_mask[np.newaxis, :, :]
        self_states[:, 9] = opponent_mask[np.newaxis, :, :]

        # Vectorized dynamic layer processing (agent positions)
        steps_to_process = min(len(trajectory_steps), seq_len)
        if steps_to_process > 0:
            # Extract positions for both agents
            achiever_positions = []
            blocker_positions = []

            for step in trajectory_steps[:steps_to_process]:
                achiever_pos = step["achiever_pos"]
                blocker_pos = step["blocker_pos"]
                achiever_positions.append(
                    achiever_pos if achiever_pos is not None else [-1, -1]
                )
                blocker_positions.append(
                    blocker_pos if blocker_pos is not None else [-1, -1]
                )

            achiever_positions = np.array(achiever_positions)
            blocker_positions = np.array(blocker_positions)

            # Determine self and opponent positions based on agent type
            if agent_type == "achiever":
                self_positions = achiever_positions
                opponent_positions = blocker_positions
            else:  # blocker
                self_positions = blocker_positions
                opponent_positions = achiever_positions

            # Process self positions
            valid_self = (
                (self_positions[:, 0] >= 0)
                & (self_positions[:, 0] < self.MAZE_WIDTH)
                & (self_positions[:, 1] >= 0)
                & (self_positions[:, 1] < self.MAZE_HEIGHT)
            )

            if np.any(valid_self):
                valid_times = np.arange(steps_to_process)[valid_self]
                valid_pos = self_positions[valid_self]

                # Set self position in channel 8
                self_states[valid_times, 8, valid_pos[:, 1], valid_pos[:, 0]] = 1

                # Also maintain the agent in empty space layer for compatibility
                self_states[valid_times, 1, valid_pos[:, 1], valid_pos[:, 0]] = 1

            # Process opponent positions (only in multi-agent mode)
            if not self.is_single_agent_mode:
                valid_opponent = (
                    (opponent_positions[:, 0] >= 0)
                    & (opponent_positions[:, 0] < self.MAZE_WIDTH)
                    & (opponent_positions[:, 1] >= 0)
                    & (opponent_positions[:, 1] < self.MAZE_HEIGHT)
                )

                if np.any(valid_opponent):
                    valid_times = np.arange(steps_to_process)[valid_opponent]
                    valid_pos = opponent_positions[valid_opponent]

                    # Set opponent position in channel 9
                    self_states[valid_times, 9, valid_pos[:, 1], valid_pos[:, 0]] = 1

        # Apply partial observation masking if needed
        if observability == "partial":
            self_states = self._apply_partial_observation_masking(
                self_states, self_positions, agent_type, partial_view_size
            )

        return self_states

    def _apply_partial_observation_masking(
        self,
        self_states: np.ndarray,
        self_positions: np.ndarray,
        agent_type: str,
        partial_view_size: int,
    ) -> np.ndarray:
        """
        Apply partial observation masking to the state tensor.

        Args:
            self_states: Full state tensor (seq_len, channels, height, width)
            self_positions: Agent positions for each timestep (seq_len, 2)
            agent_type: Type of agent ("achiever" or "blocker")
            partial_view_size: Size of partial observation window

        Returns:
            Masked state tensor with only visible areas preserved
        """
        seq_len, channels, height, width = self_states.shape
        masked_states = np.zeros_like(self_states)

        # Calculate view radius (how far agent can see in each direction)
        view_radius = partial_view_size // 2

        for t in range(seq_len):
            if t < len(self_positions):
                agent_pos = self_positions[t]

                # Skip invalid positions
                if agent_pos[0] < 0 or agent_pos[1] < 0:
                    continue

                agent_x, agent_y = int(agent_pos[0]), int(agent_pos[1])

                # Calculate visible area bounds
                x_min = max(0, agent_x - view_radius)
                x_max = min(width, agent_x + view_radius + 1)
                y_min = max(0, agent_y - view_radius)
                y_max = min(height, agent_y + view_radius + 1)

                # Copy visible area from original state
                masked_states[t, :, y_min:y_max, x_min:x_max] = self_states[
                    t, :, y_min:y_max, x_min:x_max
                ]

                # Always keep agent's own position visible (channel 8)
                if 0 <= agent_y < height and 0 <= agent_x < width:
                    masked_states[t, 8, agent_y, agent_x] = 1.0

        return masked_states

    def _create_goal_tensor(self, goal_letter: str) -> np.ndarray:
        """Create goal tensor (one-hot encoded)"""
        goal_tensor = np.zeros(4, dtype=np.float32)
        if goal_letter in self.LETTER_TO_IDX:
            goal_tensor[self.LETTER_TO_IDX[goal_letter]] = 1.0
        return goal_tensor

    def process_directory(
        self, data_dir: str, n_processes: int = None, use_multiprocessing: bool = True
    ) -> List[Dict[str, Any]]:
        """Process all trajectory files in directory with optional multiprocessing"""

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        # Find all test files
        test_files = []
        for filename in os.listdir(data_dir):
            if filename.startswith("test") and filename.endswith(".txt"):
                test_files.append(os.path.join(data_dir, filename))

        test_files.sort()
        print(f"Found {len(test_files)} trajectory files in {data_dir}")

        if not use_multiprocessing or len(test_files) < 100:
            # Use sequential processing for small datasets
            print("Using sequential processing...")
            all_samples = []
            for filepath in test_files:
                # Parse trajectory file
                parsed_data = self.parse_trajectory_file(filepath)

                # Create achiever sample
                achiever_sample = self.create_achiever_sample(parsed_data)
                all_samples.append(achiever_sample)

                # Create blocker sample only in multi-agent mode
                if not (
                    hasattr(self, "is_single_agent_mode") and self.is_single_agent_mode
                ) and not parsed_data.get("is_single_agent", False):
                    blocker_sample = self.create_blocker_sample(parsed_data)
                    all_samples.append(blocker_sample)

                # Clean up after each file processing
                if not (
                    hasattr(self, "is_single_agent_mode") and self.is_single_agent_mode
                ) and not parsed_data.get("is_single_agent", False):
                    del parsed_data, achiever_sample, blocker_sample
                else:
                    del parsed_data, achiever_sample
                if len(all_samples) % 200 == 0:  # Less frequent GC
                    gc.collect()

        else:
            # Use multiprocessing for large datasets
            if n_processes is None:
                n_processes = mp.cpu_count()

            print(f"Using multiprocessing with {n_processes} processes...")

            # Create batches of files for better load balancing
            batch_size = max(1, len(test_files) // (n_processes * 5))
            file_batches = [
                test_files[i : i + batch_size]
                for i in range(0, len(test_files), batch_size)
            ]

            # Add single-agent flag and config to each batch
            file_batches_with_config = [
                (
                    batch,
                    (
                        self.is_single_agent_mode
                        if hasattr(self, "is_single_agent_mode")
                        else False
                    ),
                    self.config,
                )
                for batch in file_batches
            ]

            # Process batches in parallel
            with mp.Pool(processes=n_processes, maxtasksperchild=100) as pool:
                batch_results = list(
                    tqdm(
                        pool.imap(process_file_batch_worker, file_batches_with_config),
                        total=len(file_batches_with_config),
                        desc="Processing trajectory files",
                    )
                )
                pool.close()
                pool.join()

            # Combine results from all batches
            all_samples = []
            for batch_result in batch_results:
                all_samples.extend(batch_result)

        # Shuffle samples
        random.shuffle(all_samples)

        # Count achiever and blocker samples
        achiever_count = sum(1 for s in all_samples if s.get("agent") == "achiever")
        blocker_count = sum(1 for s in all_samples if s.get("agent") == "blocker")

        if hasattr(self, "is_single_agent_mode") and self.is_single_agent_mode:
            print(
                f"Generated {len(all_samples)} samples ({achiever_count} achiever samples in single-agent mode)"
            )
        else:
            print(
                f"Generated {len(all_samples)} samples ({achiever_count} achiever + {blocker_count} blocker)"
            )
        return all_samples

    def save_processed_data(self, samples: List[Dict[str, Any]], output_path: str):
        """Save processed samples to pickle file"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "wb") as f:
            pickle.dump(samples, f)
        print(f"Saved {len(samples)} samples to {output_path}.")

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

        # Self states lengths
        trajectory_lengths = [sample["self_states"].shape[0] for sample in samples]

        stats = {
            "total_samples": len(samples),
            "agent_distribution": agent_counts,
            "goal_distribution": goal_counts,
            "self_states_lengths": {
                "min": min(trajectory_lengths),
                "max": max(trajectory_lengths),
                "mean": np.mean(trajectory_lengths),
                "std": np.std(trajectory_lengths),
            },
            "maze_size": (self.MAZE_WIDTH, self.MAZE_HEIGHT),
            "self_states_depth": self.MAZE_DEPTH,
        }

        return stats

    def ReadAllGames(self, data_dir: str) -> List[Dict[str, Any]]:
        """Compatibility method for evaluate.py - wraps process_directory"""
        return self.process_directory(data_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate processed data for exp8 ToMnet training"
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
        "--maze_depth", type=int, default=10, help="Maze depth (channels)"
    )

    args = parser.parse_args()

    # Override config with command line arguments if specified
    if args.config_override:
        config.update_from_args(args)

    # Initialize data generator
    generator = DataGenerator(
        time_step=args.time_step,
        w=args.maze_width,
        h=args.maze_height,
        d=args.maze_depth,
        config=config,
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
