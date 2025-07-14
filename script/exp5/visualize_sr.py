#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap


def parse_maze(maze_lines):
    """Parse maze from text lines"""
    maze = []
    for line in maze_lines:
        maze.append(list(line))
    return np.array(maze)


def parse_sr_data(sr_line):
    """Parse SR data from a line like 'SR_gamma_0.5: 1,7:0.5;2,7:0.25;...'"""
    if ":" not in sr_line:
        return {}

    # Split on the colon to get the data part
    parts = sr_line.split(":", 1)
    if len(parts) < 2:
        return {}

    data_part = parts[1].strip()
    sr_dict = {}

    # Parse position:value pairs separated by semicolons
    if data_part:
        pairs = data_part.split(";")
        for pair in pairs:
            if ":" in pair:
                try:
                    pos_str, value_str = pair.split(":")
                    x, y = map(int, pos_str.split(","))
                    value = float(value_str)
                    sr_dict[(x, y)] = value
                except ValueError:
                    continue

    return sr_dict


def visualize_sr_on_maze(maze, sr_data_dict, timestep=0, title_prefix=""):
    """Visualize SR values overlaid on the maze"""
    height, width = maze.shape

    # Create figure with subplots for each gamma
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"{title_prefix}SR Values at Timestep {timestep}",
        fontsize=16,
        fontweight="bold",
    )

    gammas = ["0.5", "0.9", "0.99"]
    gamma_keys = ["SR_gamma_0.5", "SR_gamma_0.9", "SR_gamma_0.99"]

    for idx, (gamma, gamma_key) in enumerate(zip(gammas, gamma_keys)):
        ax = axes[idx]

        # Create base maze visualization
        maze_visual = np.zeros((height, width))
        sr_values = np.zeros((height, width))

        # Get SR data for this gamma
        sr_data = sr_data_dict.get(gamma_key, {})

        # Fill in SR values
        for (x, y), value in sr_data.items():
            if 0 <= y < height and 0 <= x < width:
                sr_values[y, x] = value

        # Create color map for SR values
        sr_cmap = LinearSegmentedColormap.from_list("sr", ["white", "red", "darkred"])

        # Plot SR heatmap
        sr_max = max(sr_data.values()) if sr_data else 1.0
        im = ax.imshow(sr_values, cmap=sr_cmap, alpha=0.7, vmin=0, vmax=sr_max)

        # Overlay maze elements
        for i in range(height):
            for j in range(width):
                cell = maze[i, j]

                # Draw maze elements
                if cell == "#":  # Wall
                    rect = patches.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        linewidth=2,
                        edgecolor="black",
                        facecolor="gray",
                        alpha=0.8,
                    )
                    ax.add_patch(rect)
                elif cell == "O":  # Agent
                    circle = patches.Circle((j, i), 0.3, color="blue", alpha=0.9)
                    ax.add_patch(circle)
                    ax.text(
                        j,
                        i,
                        "A",
                        ha="center",
                        va="center",
                        fontweight="bold",
                        color="white",
                    )
                elif cell in "ABCD":  # Goals
                    circle = patches.Circle((j, i), 0.25, color="green", alpha=0.8)
                    ax.add_patch(circle)
                    ax.text(
                        j,
                        i,
                        cell,
                        ha="center",
                        va="center",
                        fontweight="bold",
                        color="white",
                    )
                elif cell in "abcd":  # Keys
                    diamond = patches.RegularPolygon(
                        (j, i),
                        4,
                        radius=0.2,
                        orientation=np.pi / 4,
                        color="gold",
                        alpha=0.8,
                    )
                    ax.add_patch(diamond)
                    ax.text(
                        j,
                        i,
                        cell.upper(),
                        ha="center",
                        va="center",
                        fontweight="bold",
                        color="black",
                        fontsize=8,
                    )
                elif cell in "ABCD" and cell.isupper():  # Doors (uppercase)
                    rect = patches.Rectangle(
                        (j - 0.3, i - 0.3), 0.6, 0.6, color="brown", alpha=0.8
                    )
                    ax.add_patch(rect)
                    ax.text(
                        j,
                        i,
                        cell,
                        ha="center",
                        va="center",
                        fontweight="bold",
                        color="white",
                        fontsize=8,
                    )

                # Add SR value text if significant
                if sr_values[i, j] > 0.01:
                    ax.text(
                        j,
                        i - 0.4,
                        f"{sr_values[i, j]:.3f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        bbox=dict(
                            boxstyle="round,pad=0.2", facecolor="white", alpha=0.7
                        ),
                    )

        # Formatting
        ax.set_xlim(-0.5, width - 0.5)
        ax.set_ylim(-0.5, height - 0.5)
        ax.set_aspect("equal")
        ax.set_title(f"γ = {gamma}", fontsize=14, fontweight="bold")
        ax.invert_yaxis()  # To match the maze orientation

        # Remove ticks
        ax.set_xticks(range(width))
        ax.set_yticks(range(height))
        ax.grid(True, alpha=0.3)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("SR Value", rotation=270, labelpad=15)

    plt.tight_layout()
    return fig


def analyze_test_file(filename):
    """Analyze and visualize SR data from a test file"""
    print(f"Analyzing {filename}...")

    with open(filename, "r") as f:
        lines = f.readlines()

    # Parse maze (lines 1-9, removing "Maze:" header)
    maze_lines = []
    maze_start = 1  # Skip "Maze:" line
    for i in range(maze_start, maze_start + 9):
        if i < len(lines):
            maze_lines.append(lines[i].strip())

    maze = parse_maze(maze_lines)
    print(f"Maze shape: {maze.shape}")

    # Find goal info
    goal_line = None
    consumption_line = None
    for line in lines:
        if line.startswith("Goal Consumed Rank"):
            goal_line = line.strip()
        elif line.startswith("Consumption Labels"):
            consumption_line = line.strip()

    print(f"Goal info: {goal_line}")
    print(f"Consumption: {consumption_line}")

    # Parse SR data for multiple timesteps
    sr_timesteps = {}
    current_timestep = None

    for line in lines:
        line = line.strip()
        if line.startswith("Timestep_"):
            current_timestep = int(line.split("_")[1].rstrip(":"))
            sr_timesteps[current_timestep] = {}
        elif line.startswith("SR_gamma_") and current_timestep is not None:
            gamma_key = line.split(":")[0]
            sr_data = parse_sr_data(line)
            sr_timesteps[current_timestep][gamma_key] = sr_data

    print(f"Found SR data for timesteps: {list(sr_timesteps.keys())}")

    # Visualize first few timesteps
    timesteps_to_show = (
        [0, 5, 10, 15, 20] if len(sr_timesteps) > 7 else list(sr_timesteps.keys())[:4]
    )

    for t in timesteps_to_show:
        if t in sr_timesteps:
            fig = visualize_sr_on_maze(
                maze,
                sr_timesteps[t],
                timestep=t,
                title_prefix=f"{filename.split('/')[-1]} - ",
            )
            plt.savefig(
                f"sr_visualization_timestep_{t}.png", dpi=150, bbox_inches="tight"
            )
            print(f"Saved visualization for timestep {t}")

    plt.show()

    # Print some statistics
    print("\n=== SR Statistics ===")
    for t in timesteps_to_show[:2]:  # Just first 2 timesteps
        if t in sr_timesteps:
            print(f"\nTimestep {t}:")
            for gamma_key in ["SR_gamma_0.5", "SR_gamma_0.9", "SR_gamma_0.99"]:
                if gamma_key in sr_timesteps[t]:
                    sr_data = sr_timesteps[t][gamma_key]
                    values = list(sr_data.values())
                    if values:
                        print(
                            f"  {gamma_key}: {len(values)} positions, "
                            f"sum={sum(values):.3f}, max={max(values):.3f}, "
                            f"min={min(values):.3f}"
                        )


if __name__ == "__main__":
    # Analyze the test file we looked at
    test_file = (
        "/Users/youngjins/Desktop/codes/25_belief/belief_trading/data/exp5/test98.txt"
    )
    analyze_test_file(test_file)
