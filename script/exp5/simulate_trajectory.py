import numpy as np
import sys
import os
import argparse
import time
from PIL import Image
import warnings

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


def env_to_maze_format(env, achiever_pos, blocker_pos):
    """
    Convert AchieverBlocker environment to maze format without outer walls
    (Matches the format used in generate.py)
    """
    maze_lines = []
    width, height = env.width, env.height

    # Process each row without adding outer walls
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


class GameSimulation:
    """Simulate game based on data file"""

    def __init__(self, data_file):
        self.data_file = data_file
        self.maze = []
        self.achiever_positions = []
        self.blocker_positions = []
        self.achiever_sr_data = {}
        self.blocker_sr_data = {}
        self.goal_rank = []
        self.trajectory_length = 0
        self.consumption_labels = []
        self.blocker_infer_goal = None

        # Parse data
        self.parse_data()

        # Find initial positions
        self.find_initial_positions()

        # Action names for display
        self.achiever_action_names = [
            "up",
            "right",
            "down",
            "left",
            "stay",
            "pickup",
            "toggle",
        ]

        self.blocker_action_names = [
            "up",
            "right",
            "down",
            "left",
            "stay",
            "broken",
        ]

    def parse_data(self):
        """Parse the data file"""
        with open(self.data_file, "r") as f:
            lines = f.readlines()

        # Parse maze
        maze_section = False
        trajectory_section = False
        achiever_section = False
        blocker_section = False
        sr_section = False

        # Additional trajectory data
        self.achiever_actions = []
        self.blocker_actions = []
        self.achiever_interactions = []
        self.blocker_interactions = []
        # No heading tracking needed with direct movement

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Parse MAZE section
            if line == "MAZE:":
                maze_section = True
                i += 1
                continue

            if maze_section and line.startswith("Trajectory length:"):
                maze_section = False
                self.trajectory_length = int(line.split(":")[1].strip())
                trajectory_section = True
                i += 1
                continue

            if maze_section and line.strip():
                self.maze.append(list(line.strip()))
                i += 1
                continue

            # Parse trajectory section (two-agent format)
            if trajectory_section and line.startswith("[") and "][" in line:
                # Format: [achiever_pos][blocker_pos] : achiever_action,blocker_action : achiever_interaction,blocker_interaction
                parts = line.split(" : ")
                if len(parts) >= 3:
                    # Parse positions
                    pos_part = parts[0]
                    achiever_pos_str = pos_part.split("][")[0].strip("[")
                    blocker_pos_str = pos_part.split("][")[1].strip("]")
                    achiever_pos = tuple(map(int, achiever_pos_str.split(", ")))
                    blocker_pos = tuple(map(int, blocker_pos_str.split(", ")))

                    # Parse actions
                    action_part = parts[1]
                    actions = action_part.split(",")
                    achiever_action = int(actions[0])
                    blocker_action = int(actions[1])

                    # Parse interactions
                    interaction_part = parts[2]
                    interactions = interaction_part.split(",")
                    achiever_interaction = interactions[0]
                    blocker_interaction = (
                        interactions[1] if len(interactions) > 1 else "X"
                    )

                    self.achiever_positions.append(achiever_pos)
                    self.blocker_positions.append(blocker_pos)
                    self.achiever_actions.append(achiever_action)
                    self.blocker_actions.append(blocker_action)
                    self.achiever_interactions.append(achiever_interaction)
                    self.blocker_interactions.append(blocker_interaction)
                i += 1
                continue

            # Parse Achiever section
            if line == "Achiever:":
                trajectory_section = False
                achiever_section = True
                i += 1
                continue

            if achiever_section:
                if line.startswith("Goal Consumed Rank:"):
                    rank_str = line.split(":")[1].strip()
                    self.goal_rank = eval(rank_str)
                elif line.startswith("Consumption Labels:"):
                    labels = line.split(":")[1].strip().split(",")
                    self.consumption_labels = [float(l) for l in labels]
                elif line == "SR_Data_Per_Timestep:":
                    sr_section = True
                    current_agent = "achiever"
                elif line == "Blocker:":
                    achiever_section = False
                    blocker_section = True
                    sr_section = False
                i += 1
                continue

            # Parse Blocker section
            if blocker_section:
                if line.startswith("Infer Goal:"):
                    self.blocker_infer_goal = line.split(":")[1].strip()
                elif line == "SR_Data_Per_Timestep:":
                    sr_section = True
                    current_agent = "blocker"
                i += 1
                continue

            # Parse SR data
            if sr_section and line.startswith("Timestep_"):
                timestep = int(line.split("_")[1].replace(":", ""))

                if current_agent == "achiever":
                    self.achiever_sr_data[timestep] = {}
                    sr_dict = self.achiever_sr_data[timestep]
                else:
                    self.blocker_sr_data[timestep] = {}
                    sr_dict = self.blocker_sr_data[timestep]

                # Read next 3 lines for gamma values
                for j in range(3):
                    if i + j + 1 < len(lines):
                        gamma_line = lines[i + j + 1].strip()
                        if gamma_line.startswith("SR_gamma_"):
                            parts = gamma_line.split(": ")
                            gamma_val = float(parts[0].split("_")[2])

                            # Handle sparse format
                            if len(parts) > 1:
                                sparse_data = parts[1]
                                entries = sparse_data.split(";")
                                sr_entries = []
                                for entry in entries:
                                    if ":" in entry:
                                        pos_val = entry.split(":")
                                        pos = tuple(map(int, pos_val[0].split(",")))
                                        value = float(pos_val[1])
                                        sr_entries.append((pos, value))
                                sr_dict[gamma_val] = sr_entries
                            else:
                                sr_dict[gamma_val] = []

                i += 4  # Skip to next timestep
                continue

            i += 1

    def find_initial_positions(self):
        """Find initial positions of agents and objects"""
        self.initial_positions = {}

        for i, row in enumerate(self.maze):
            for j, cell in enumerate(row):
                if cell == "O":
                    self.initial_positions["achiever"] = (
                        j,
                        i,
                    )  # (x, y) = (column, row)
                elif cell == "X":
                    self.initial_positions["blocker"] = (j, i)  # (x, y) = (column, row)
                elif cell in "ABCDabcd":
                    self.initial_positions[cell] = (j, i)  # (x, y) = (column, row)

    def create_minigrid_env(self):
        """Create MiniGrid environment from maze data"""
        # Determine environment size based on maze
        height = len(self.maze)
        width = len(self.maze[0]) if height > 0 else 0

        # Create environment based on size (but we'll completely overwrite the grid)
        if height == 5 and width == 5:
            env_name = "MiniGrid-AchieverBlocker-5x5-v1"
        elif height == 9 and width == 9:
            env_name = "MiniGrid-AchieverBlocker-9x9-v1"
        elif height == 11 and width == 11:
            env_name = "MiniGrid-AchieverBlocker-11x11-v1"
        else:
            env_name = "MiniGrid-AchieverBlocker-9x9-v1"  # Default

        # Create environment
        env = gym.make(env_name, max_steps=500)
        env = env.unwrapped if hasattr(env, "unwrapped") else env

        # Set render mode for image capture
        env.render_mode = "rgb_array"

        return env

    def _reconstruct_exact_environment(self, env):
        """Reconstruct the exact environment from maze data for two agents"""
        from gym_minigrid.minigrid import Grid, Wall, Key, Door

        # Get maze dimensions
        maze_height = len(self.maze)
        maze_width = len(self.maze[0]) if maze_height > 0 else 0

        # Create new grid with exact maze size
        env.grid = Grid(maze_width, maze_height)
        env.width = maze_width
        env.height = maze_height

        # Initialize agent inventories
        env.achiever_keys = []
        env.blocker_keys = []

        # Initialize agent directions (facing right by default)
        env.achiever_dir = 0  # 0=east, 1=south, 2=west, 3=north
        env.blocker_dir = 0

        # Reconstruct from maze data directly
        achiever_pos = None
        blocker_pos = None

        for y in range(maze_height):
            for x in range(maze_width):
                cell = self.maze[y][x]

                if cell == "#":
                    env.grid.set(x, y, Wall())
                elif cell in "ABCD":
                    color_map = {"A": "red", "B": "green", "C": "blue", "D": "yellow"}
                    color = color_map.get(cell, "red")
                    key = Key(color)
                    key.can_overlap = lambda: True
                    env.grid.set(x, y, key)
                elif cell in "abcd":
                    color_map = {"a": "red", "b": "green", "c": "blue", "d": "yellow"}
                    color = color_map.get(cell, "red")
                    door = Door(color, is_locked=True)
                    env.grid.set(x, y, door)
                elif cell == "O":
                    achiever_pos = (x, y)
                elif cell == "X":
                    blocker_pos = (x, y)

        # Set agent positions
        if len(self.achiever_positions) > 0:
            env.achiever_pos = np.array(self.achiever_positions[0])
        elif achiever_pos:
            env.achiever_pos = np.array(achiever_pos)
        else:
            env.achiever_pos = np.array([0, 0])  # Fallback

        if len(self.blocker_positions) > 0:
            env.blocker_pos = np.array(self.blocker_positions[0])
        elif blocker_pos:
            env.blocker_pos = np.array(blocker_pos)
        else:
            env.blocker_pos = np.array([1, 1])  # Fallback

        # Set target door color for achiever
        if self.goal_rank:
            colors = ["red", "green", "blue", "yellow"]
            target_idx = self.goal_rank.index(1) if 1 in self.goal_rank else 0
            env.target_door_color = colors[target_idx]
            env.mission = f"Achiever: collect {env.target_door_color} key and open {env.target_door_color} door. Blocker: prevent achiever."
        else:
            env.target_door_color = "red"
            env.mission = "Achiever: collect red key and open red door. Blocker: prevent achiever."

    def _reset_environment_state_from_trajectory(self, env, current_step):
        """Reset environment state to exactly match trajectory data up to current step"""
        from gym_minigrid.minigrid import Grid, Wall, Key, Door

        # Completely reconstruct grid from original maze data
        maze_height = len(self.maze)
        maze_width = len(self.maze[0]) if maze_height > 0 else 0
        env.grid = Grid(maze_width, maze_height)

        # Initialize fresh state
        env.achiever_keys = []
        collected_keys = set()
        opened_doors = set()

        # Replay all interactions from start to current step to determine what should be collected/opened
        for step_idx in range(current_step):
            if step_idx < len(self.achiever_interactions):
                interaction = self.achiever_interactions[step_idx]

                if interaction in "ABCD":
                    # Key pickup
                    color_map = {"A": "red", "B": "green", "C": "blue", "D": "yellow"}
                    key_color = color_map[interaction]
                    if key_color not in collected_keys:
                        collected_keys.add(key_color)
                        env.achiever_keys.append(key_color)

                elif interaction in "abcd":
                    # Door opening
                    color_map = {"a": "red", "b": "green", "c": "blue", "d": "yellow"}
                    door_color = color_map[interaction]
                    opened_doors.add(door_color)

        # Reconstruct grid from original maze data, excluding collected keys and opening doors
        for y in range(maze_height):
            for x in range(maze_width):
                cell = self.maze[y][x]

                if cell == "#":
                    env.grid.set(x, y, Wall())
                elif cell in "ABCD":
                    color_map = {"A": "red", "B": "green", "C": "blue", "D": "yellow"}
                    color = color_map.get(cell, "red")
                    # Only place key if it hasn't been collected according to trajectory
                    if color not in collected_keys:
                        key = Key(color)
                        key.can_overlap = lambda: True
                        env.grid.set(x, y, key)
                elif cell in "abcd":
                    color_map = {"a": "red", "b": "green", "c": "blue", "d": "yellow"}
                    color = color_map.get(cell, "red")
                    door = Door(color, is_locked=True)
                    # Open door if it should be opened according to trajectory
                    if color in opened_doors:
                        door.is_open = True
                        door.is_locked = False
                    env.grid.set(x, y, door)

    def render_to_image(self, env):
        """Render environment to PIL Image using native MiniGrid rendering (same as render_kd.py)"""
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

    def visualize_trajectory(self, save_gif=False, pause_time=0.5):
        """Visualize the two-agent trajectory using native MiniGrid rendering"""
        print("=== AchieverBlocker Game Simulation ===")
        print(f"Mission: Replay two-agent trajectory from data file")
        print(f"Trajectory length: {self.trajectory_length}")
        print(f"Goal rank: {self.goal_rank}")
        if self.blocker_infer_goal:
            print(f"Blocker inferred goal: {self.blocker_infer_goal}")

        # Create environment
        env = self.create_minigrid_env()

        # Reset environment
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs, _ = reset_result
        else:
            obs = reset_result

        # Reconstruct the exact environment from maze data
        self._reconstruct_exact_environment(env)

        frames = []

        # Capture initial frame
        if save_gif:
            initial_frame = self.render_to_image(env)
            if initial_frame:
                frames.append(initial_frame)

        # Display initial state
        try:
            env.render()
        except Exception as e:
            print(f"Warning: Could not render: {e}")

        print(f"Initial achiever position: {env.achiever_pos}")
        print(f"Initial blocker position: {env.blocker_pos}")
        print(f"Initial achiever keys: {env.achiever_keys}")

        print("\nReplaying trajectory...")

        # Replay each step
        for step in range(min(len(self.achiever_actions), self.trajectory_length)):
            achiever_action = self.achiever_actions[step]
            blocker_action = self.blocker_actions[step]
            achiever_position = self.achiever_positions[step]
            blocker_position = self.blocker_positions[step]
            achiever_interaction = (
                self.achiever_interactions[step]
                if step < len(self.achiever_interactions)
                else "X"
            )
            blocker_interaction = (
                self.blocker_interactions[step]
                if step < len(self.blocker_interactions)
                else "X"
            )

            # Get action names
            achiever_action_name = (
                self.achiever_action_names[achiever_action]
                if achiever_action < len(self.achiever_action_names)
                else f"action_{achiever_action}"
            )
            blocker_action_name = (
                self.blocker_action_names[blocker_action]
                if blocker_action < len(self.blocker_action_names)
                else f"action_{blocker_action}"
            )

            print(f"Step {step + 1}:")
            print(
                f"  Achiever: {achiever_action_name} at {achiever_position} -> {achiever_interaction}"
            )
            print(
                f"  Blocker:  {blocker_action_name} at {blocker_position} -> {blocker_interaction}"
            )

            if hasattr(env, "achiever_keys") and env.achiever_keys:
                print(f"  Achiever inventory: {env.achiever_keys}")
            if hasattr(env, "blocker_keys") and env.blocker_keys:
                print(f"  Blocker inventory: {env.blocker_keys}")

            # Update agent positions and directions for visualization
            try:
                # Update agent directions based on actions (for visualization)
                action_to_direction = {
                    0: 3,  # up -> north
                    1: 0,  # right -> east
                    2: 1,  # down -> south
                    3: 2,  # left -> west
                    # For stay, pickup, toggle actions, keep current direction
                }

                if achiever_action in action_to_direction:
                    env.achiever_dir = action_to_direction[achiever_action]

                if blocker_action in action_to_direction:
                    env.blocker_dir = action_to_direction[blocker_action]

                # Execute actions in environment and let it handle state naturally
                action_pair = (achiever_action, blocker_action)
                obs, rewards, terminated, truncated, info = env.step(action_pair)
                done = terminated or truncated

                # Update agent positions for rendering consistency (after env.step)
                env.achiever_pos = np.array(achiever_position)
                env.blocker_pos = np.array(blocker_position)

                # Reset environment state to match trajectory exactly (override automatic behavior)
                self._reset_environment_state_from_trajectory(env, step)

                # Update the main agent_pos for rendering (use achiever as primary)
                env.agent_pos = env.achiever_pos
                env.agent_dir = env.achiever_dir

                # Update step counter to match trajectory
                env.step_count = step + 1

                # Render environment (currently shows only achiever due to MiniGrid limitation)
                try:
                    env.render()
                except Exception as e:
                    print(f"Warning: Could not render: {e}")

                # Capture frame for GIF
                if save_gif:
                    frame = self.render_to_image(env)
                    if frame:
                        frames.append(frame)

                # Check if episode is done
                if done:
                    print(f"Episode ended at step {step + 1} with rewards: {rewards}")
                    print(
                        f"Environment says episode is done, but trajectory continues..."
                    )
                    print(f"Terminated: {terminated}, Truncated: {truncated}")
                    # Don't break here - continue with the trajectory regardless of environment state

                # Pause between steps
                if pause_time > 0:
                    time.sleep(pause_time)

            except Exception as e:
                print(f"Error during step {step + 1}: {e}")
                break

        # Save GIF if requested
        if save_gif and frames:
            gif_path = "achiever_blocker_trajectory.gif"
            print(f"Saving {len(frames)} frames to {gif_path}")
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=500,  # 500ms per frame
                loop=0,
            )
            print(f"GIF saved to {gif_path}")

        # Close environment
        env.close()

        print("\nSimulation completed!")

    def print_summary(self):
        """Print summary of the simulation data"""
        print("=== AchieverBlocker Game Simulation Summary ===")
        print(f"Maze size: {len(self.maze)}x{len(self.maze[0]) if self.maze else 0}")
        print(f"Goal Consumed Rank: {self.goal_rank}")
        print(f"Trajectory Length: {self.trajectory_length}")
        print(f"Consumption Labels: {self.consumption_labels}")
        if self.blocker_infer_goal:
            print(f"Blocker Inferred Goal: {self.blocker_infer_goal}")
        print(
            f"Number of timesteps with Achiever SR data: {len(self.achiever_sr_data)}"
        )
        print(f"Number of timesteps with Blocker SR data: {len(self.blocker_sr_data)}")
        print(f"Number of achiever position records: {len(self.achiever_positions)}")
        print(f"Number of blocker position records: {len(self.blocker_positions)}")
        print(f"\nInitial positions:")
        for key, pos in self.initial_positions.items():
            print(f"  {key}: {pos}")

        # Show first few SR entries for both agents
        print(f"\nFirst 3 Achiever SR data entries:")
        for i in range(min(3, len(self.achiever_sr_data))):
            print(f"  Timestep {i}: {self.achiever_sr_data.get(i, {})}")

        print(f"\nFirst 3 Blocker SR data entries:")
        for i in range(min(3, len(self.blocker_sr_data))):
            print(f"  Timestep {i}: {self.blocker_sr_data.get(i, {})}")


def main():
    parser = argparse.ArgumentParser(description="Simulate game from data file")
    parser.add_argument(
        "--data_file",
        type=str,
        default="../../data/exp5/test0.txt",
        help="Path to data file",
    )
    parser.add_argument("--save_gif", action="store_true", help="Save animation as GIF")
    parser.add_argument(
        "--pause",
        type=float,
        default=0.1,
        help="Pause duration between actions in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--summary", action="store_true", help="Print summary without visualization"
    )

    args = parser.parse_args()

    # Create simulation
    sim = GameSimulation(args.data_file)

    if args.summary:
        sim.print_summary()
    else:
        sim.visualize_trajectory(save_gif=args.save_gif, pause_time=args.pause)


if __name__ == "__main__":
    main()
