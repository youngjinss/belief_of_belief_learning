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


def env_to_maze_format(env, agent_pos):
    """
    Convert KeyDoor environment to maze format without outer walls
    (Matches the format used in generate.py)
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


class GameSimulation:
    """Simulate game based on data file"""

    def __init__(self, data_file):
        self.data_file = data_file
        self.maze = []
        self.agent_positions = []
        self.sr_data = {}
        self.goal_rank = []
        self.trajectory_length = 0
        self.consumption_labels = []

        # Parse data
        self.parse_data()

        # Find initial positions
        self.find_initial_positions()

        # Action names for display
        self.action_names = [
            "up",
            "right",
            "down",
            "left",
            "stay",
            "pickup",
            "toggle",
        ]

    def parse_data(self):
        """Parse the data file"""
        with open(self.data_file, "r") as f:
            lines = f.readlines()

        # Parse maze
        maze_section = False
        sr_section = False
        position_section = False

        # Additional trajectory data
        self.actions = []
        self.interactions = []
        # No heading tracking needed with direct movement

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            if line == "Maze:":
                maze_section = True
                i += 1
                continue

            if maze_section and line.startswith("Goal Consumed Rank"):
                maze_section = False
                # Parse goal rank
                rank_str = line.split(":")[1].strip()
                self.goal_rank = eval(rank_str)
                i += 1
                continue

            if (
                maze_section
                and line.strip()
                and not line.startswith("Goal Consumed Rank")
            ):
                self.maze.append(list(line.strip()))
                i += 1
                continue

            if line.startswith("Trajectory length:"):
                self.trajectory_length = int(line.split(":")[1].strip())
                i += 1
                continue

            if line.startswith("Consumption Labels:"):
                labels = line.split(":")[1].strip().split(",")
                self.consumption_labels = [float(l) for l in labels]
                i += 1
                continue

            if line == "SR_Data_Per_Timestep:":
                sr_section = True
                i += 1
                continue

            if sr_section and line.startswith("Timestep_"):
                timestep = int(line.split("_")[1].replace(":", ""))
                self.sr_data[timestep] = {}

                # Read next 3 lines for gamma values
                for j in range(3):
                    if i + j + 1 < len(lines):
                        gamma_line = lines[i + j + 1].strip()
                        if gamma_line.startswith("SR_gamma_"):
                            parts = gamma_line.split(": ")
                            gamma_val = float(parts[0].split("_")[2])

                            # Handle sparse format: "1,5:0.046875;1,6:0.0625;..."
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
                                self.sr_data[timestep][gamma_val] = sr_entries
                            else:
                                self.sr_data[timestep][gamma_val] = []

                i += 4  # Skip to next timestep
                continue

            if sr_section and line.startswith("[") and ":" in line:
                # Position log section: [x, y] : action : interaction
                parts = line.split(" : ")
                if len(parts) >= 3:
                    pos_str = parts[0].strip("[]")
                    pos = tuple(map(int, pos_str.split(", ")))
                    action = int(parts[1])
                    interaction = parts[2]

                    self.agent_positions.append(pos)
                    self.actions.append(action)
                    self.interactions.append(interaction)
                i += 1
                continue

            i += 1

    def find_initial_positions(self):
        """Find initial positions of agent and objects"""
        self.initial_positions = {}

        for i, row in enumerate(self.maze):
            for j, cell in enumerate(row):
                if cell == "O":
                    self.initial_positions["agent"] = (i, j)
                elif cell in "ABCDabcd":
                    self.initial_positions[cell] = (i, j)

    def create_minigrid_env(self):
        """Create MiniGrid environment from maze data"""
        # Determine environment size based on maze
        height = len(self.maze)
        width = len(self.maze[0]) if height > 0 else 0

        # Create environment based on size (but we'll completely overwrite the grid)
        if height == 5 and width == 5:
            env_name = "MiniGrid-KeyDoor-5x5-v0"
        elif height == 9 and width == 9:
            env_name = "MiniGrid-KeyDoor-9x9-v0"
        elif height == 11 and width == 11:
            env_name = "MiniGrid-KeyDoor-9x9-v0"  # 11x11 maze data uses 9x9 environment
        else:
            env_name = "MiniGrid-KeyDoor-9x9-v0"  # Default

        # Create environment
        env = gym.make(env_name, max_steps=500)
        env = env.unwrapped if hasattr(env, "unwrapped") else env

        # Set render mode for image capture
        env.render_mode = "rgb_array"

        return env

    def _reconstruct_exact_environment(self, env):
        """Reconstruct the exact environment from maze data"""
        from gym_minigrid.minigrid import Grid, Wall, Key, Door

        # Get maze dimensions (now 9x9 without outer walls)
        maze_height = len(self.maze)
        maze_width = len(self.maze[0]) if maze_height > 0 else 0

        # Create new grid with exact maze size
        env.grid = Grid(maze_width, maze_height)
        env.width = maze_width
        env.height = maze_height

        # Clear agent keys
        env.agent_keys = []

        # Reconstruct from maze data directly
        agent_pos = None

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
                    door.can_overlap = (
                        lambda color=color: not door.is_locked
                        or color in env.agent_keys
                    )
                    env.grid.set(x, y, door)
                elif cell == "O":
                    agent_pos = (x, y)

        # Set agent position
        if len(self.agent_positions) > 0:
            # Use trajectory starting position
            env.agent_pos = np.array(self.agent_positions[0])
        elif agent_pos:
            # Use maze position
            env.agent_pos = np.array(agent_pos)
        else:
            # Fallback: find agent position from maze if not found in trajectory
            for y in range(maze_height):
                for x in range(maze_width):
                    if self.maze[y][x] == "O":
                        env.agent_pos = np.array([x, y])
                        break

        # Set target door color
        if self.goal_rank:
            colors = ["red", "green", "blue", "yellow"]
            target_idx = self.goal_rank.index(1) if 1 in self.goal_rank else 0
            env.target_door_color = colors[target_idx]
            env.mission = f"collect {env.target_door_color} key and open {env.target_door_color} door"
        else:
            env.target_door_color = "red"
            env.mission = "collect red key and open red door"

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
        """Visualize the agent's trajectory using native MiniGrid rendering"""
        print("=== Game Simulation ===")
        print(f"Mission: Replay trajectory from data file")
        print(f"Trajectory length: {self.trajectory_length}")
        print(f"Goal rank: {self.goal_rank}")

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

        print(f"Initial agent position: {env.agent_pos}")
        print(f"Initial agent keys: {env.agent_keys}")
        print(f"Initial positions from maze: {self.initial_positions}")

        print("\nReplaying trajectory...")

        # Replay each step
        for step in range(min(len(self.actions), self.trajectory_length)):
            action = self.actions[step]
            position = self.agent_positions[step]
            interaction = (
                self.interactions[step] if step < len(self.interactions) else "X"
            )
            # Get action name
            action_name = (
                self.action_names[action]
                if action < len(self.action_names)
                else f"action_{action}"
            )

            # Update agent direction based on action for better visualization
            action_to_direction = {
                0: 3,  # up -> north
                1: 0,  # right -> east
                2: 1,  # down -> south
                3: 2,  # left -> west
                # For stay, pickup, toggle actions, keep current direction
            }

            if action in action_to_direction:
                env.agent_dir = action_to_direction[action]
                direction_names = ["east", "south", "west", "north"]
                direction_name = direction_names[env.agent_dir]
                print(
                    f"Step {step + 1}: {action_name} at {position} facing {direction_name} -> {interaction}"
                )
            else:
                print(f"Step {step + 1}: {action_name} at {position} -> {interaction}")

            if hasattr(env, "agent_keys") and env.agent_keys:
                print(f"  Agent inventory: {env.agent_keys}")

            # Take step in environment
            try:
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Let MiniGrid handle key pickup naturally - ignore interaction data for keys
                # The interaction data is just for logging, not for controlling the environment

                # Handle observation if it's a dict
                if isinstance(obs, dict):
                    obs = obs.get("image", obs)

                # Render environment
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
                    print(f"Episode ended at step {step + 1} with reward: {reward:.2f}")
                    print(
                        f"Environment says episode is done, but trajectory continues..."
                    )
                    print(f"Terminated: {terminated}, Truncated: {truncated}")
                    # Don't break here - continue with the trajectory regardless of environment state
                    done = False

                # Pause between steps
                if pause_time > 0:
                    time.sleep(pause_time)

            except Exception as e:
                print(f"Error during step {step + 1}: {e}")
                break

        # Save GIF if requested
        if save_gif and frames:
            gif_path = "trajectory_simulation.gif"
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
        print("=== Game Simulation Summary ===")
        print(f"Maze size: {len(self.maze)}x{len(self.maze[0]) if self.maze else 0}")
        print(f"Goal Consumed Rank: {self.goal_rank}")
        print(f"Trajectory Length: {self.trajectory_length}")
        print(f"Consumption Labels: {self.consumption_labels}")
        print(f"Number of timesteps with SR data: {len(self.sr_data)}")
        print(f"Number of position records: {len(self.agent_positions)}")
        print(f"\nInitial positions:")
        for key, pos in self.initial_positions.items():
            print(f"  {key}: {pos}")

        # Show first few SR entries
        print(f"\nFirst 5 SR data entries:")
        for i in range(min(5, len(self.sr_data))):
            print(f"  Timestep {i}: {self.sr_data.get(i, {})}")


def main():
    parser = argparse.ArgumentParser(description="Simulate game from data file")
    parser.add_argument(
        "--data_file",
        type=str,
        default="../../data/exp3/test0.txt",
        help="Path to data file",
    )
    parser.add_argument("--save_gif", action="store_true", help="Save animation as GIF")
    parser.add_argument(
        "--pause",
        type=float,
        default=0.5,
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
