"""Two-player trajectory replay: grid, both agents, and their SR beliefs.

Renders a saved AchieverBlocker trajectory as an animated GIF showing the maze,
the keys and doors, *both* agents moving, and each agent's successor
representation as a heatmap alongside.

This exists because `script/exp*/simulate_trajectory.py` cannot produce such a
video. It replays through MiniGrid's own renderer, which draws only the achiever
(see its own comment, "currently shows only achiever due to MiniGrid
limitation"), and its frames barely change, so PIL's GIF writer collapses 101
captured frames down to 9. It also builds a `-v1` environment for `-v2` data,
and its SR parser returns nothing. Rendering straight from the trajectory file
sidesteps all four problems: the file already holds everything needed.

    python -m beliefrl.viz.replay data/.../test0.txt -o replay.gif
    python -m beliefrl.viz.replay data/.../test0.txt --gamma 0.9 --fps 5
"""

import argparse
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter

# Door letters map to colours in the order the experiments use throughout:
# config's door_colors / goal_names are red, green, blue, yellow.
DOOR_COLORS = {"A": "red", "B": "green", "C": "blue", "D": "gold"}
KEY_COLORS = {"a": "red", "b": "green", "c": "blue", "d": "gold"}

WALL = "#"

ACHIEVER_ACTIONS = ["up", "right", "down", "left", "stay", "pickup", "toggle"]
BLOCKER_ACTIONS = ["up", "right", "down", "left", "stay", "break"]


@dataclass
class Trajectory:
    """One replayed game, parsed straight from the trajectory text file."""

    maze: List[str]
    width: int
    height: int
    length: int
    achiever_pos: List[Tuple[int, int]] = field(default_factory=list)
    blocker_pos: List[Tuple[int, int]] = field(default_factory=list)
    achiever_actions: List[int] = field(default_factory=list)
    blocker_actions: List[int] = field(default_factory=list)
    achiever_interactions: List[str] = field(default_factory=list)
    blocker_interactions: List[str] = field(default_factory=list)
    # {timestep: {gamma: {(x, y): value}}}
    achiever_sr: Dict[int, Dict[str, Dict[Tuple[int, int], float]]] = field(
        default_factory=dict
    )
    blocker_sr: Dict[int, Dict[str, Dict[Tuple[int, int], float]]] = field(
        default_factory=dict
    )
    keys: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    doors: Dict[str, Tuple[int, int]] = field(default_factory=dict)

    def gammas(self) -> List[str]:
        for table in (self.achiever_sr, self.blocker_sr):
            for per_gamma in table.values():
                if per_gamma:
                    return sorted(per_gamma)
        return []


_STEP_RE = re.compile(
    r"^\[(\d+),\s*(\d+)\]\[(\d+),\s*(\d+)\]\s*:\s*(\d+),(\d+)\s*:\s*(\S+),(\S+)\s*$"
)


def _parse_sr_line(line: str) -> Dict[Tuple[int, int], float]:
    """`SR_gamma_0.9: 1,1:0.53;1,2:0.06` -> {(1, 1): 0.53, (1, 2): 0.06}."""
    _, _, payload = line.partition(":")
    cells: Dict[Tuple[int, int], float] = {}
    for item in payload.strip().split(";"):
        if not item:
            continue
        coord, _, value = item.rpartition(":")
        x, _, y = coord.partition(",")
        try:
            cells[(int(x), int(y))] = float(value)
        except ValueError:
            continue
    return cells


def parse_trajectory(path: str) -> Trajectory:
    """Parse a trajectory file into maze, per-step positions, and SR tables."""
    with open(path) as handle:
        lines = [line.rstrip("\n") for line in handle]

    if not lines or lines[0].strip() != "MAZE:":
        raise ValueError(f"{path}: expected a 'MAZE:' header on the first line")

    maze: List[str] = []
    index = 1
    while index < len(lines) and not lines[index].startswith("Trajectory length"):
        maze.append(lines[index])
        index += 1

    height = len(maze)
    width = max((len(row) for row in maze), default=0)
    length = 0
    if index < len(lines):
        length = int(lines[index].split(":")[1].strip())
        index += 1

    traj = Trajectory(maze=maze, width=width, height=height, length=length)

    # Keys and doors are marked in the maze itself.
    for y, row in enumerate(maze):
        for x, ch in enumerate(row):
            if ch in KEY_COLORS:
                traj.keys[ch] = (x, y)
            elif ch in DOOR_COLORS:
                traj.doors[ch] = (x, y)

    # Per-step lines, then the two agent sections with their SR tables.
    section: Optional[str] = None
    timestep: Optional[int] = None

    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if not line:
            continue

        match = _STEP_RE.match(line)
        if match:
            ax, ay, bx, by, aa, ba, ai, bi = match.groups()
            traj.achiever_pos.append((int(ax), int(ay)))
            traj.blocker_pos.append((int(bx), int(by)))
            traj.achiever_actions.append(int(aa))
            traj.blocker_actions.append(int(ba))
            traj.achiever_interactions.append(ai)
            traj.blocker_interactions.append(bi)
            continue

        if line == "Achiever:":
            section, timestep = "achiever", None
        elif line == "Blocker:":
            section, timestep = "blocker", None
        elif line.startswith("Timestep_"):
            timestep = int(line[len("Timestep_"):].rstrip(":"))
        elif line.startswith("SR_gamma_") and section and timestep is not None:
            gamma = line.split(":")[0][len("SR_gamma_"):]
            table = traj.achiever_sr if section == "achiever" else traj.blocker_sr
            table.setdefault(timestep, {})[gamma] = _parse_sr_line(line)

    return traj


def _draw_grid(ax, traj: Trajectory, step: int) -> None:
    """Maze, keys, doors, and both agents at one step."""
    ax.clear()
    ax.set_xlim(-0.5, traj.width - 0.5)
    ax.set_ylim(traj.height - 0.5, -0.5)  # row 0 at the top
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    for y, row in enumerate(traj.maze):
        for x, ch in enumerate(row):
            if ch == WALL:
                ax.add_patch(
                    patches.Rectangle((x - 0.5, y - 0.5), 1, 1, color="#3a3a3a")
                )

    for ch, (x, y) in traj.doors.items():
        ax.add_patch(
            patches.Rectangle(
                (x - 0.5, y - 0.5), 1, 1,
                facecolor=DOOR_COLORS[ch], edgecolor="black",
                linewidth=1.5, alpha=0.85,
            )
        )
        ax.text(x, y, ch, ha="center", va="center", fontsize=9, fontweight="bold")

    for ch, (x, y) in traj.keys.items():
        ax.add_patch(
            patches.Circle((x, y), 0.22, facecolor=KEY_COLORS[ch], edgecolor="black")
        )

    ax.add_patch(
        patches.Circle(
            traj.achiever_pos[step], 0.36,
            facecolor="#1f77b4", edgecolor="white", linewidth=2, zorder=5,
        )
    )
    ax.add_patch(
        patches.RegularPolygon(
            traj.blocker_pos[step], numVertices=4, radius=0.40, orientation=0.785,
            facecolor="#ff7f0e", edgecolor="white", linewidth=2, zorder=5,
        )
    )

    a_idx = traj.achiever_actions[step]
    b_idx = traj.blocker_actions[step]
    a_act = ACHIEVER_ACTIONS[a_idx] if a_idx < len(ACHIEVER_ACTIONS) else str(a_idx)
    b_act = BLOCKER_ACTIONS[b_idx] if b_idx < len(BLOCKER_ACTIONS) else str(b_idx)
    ax.set_title(
        f"step {step + 1}/{len(traj.achiever_pos)}\n"
        f"achiever: {a_act}    blocker: {b_act}",
        fontsize=10,
    )


def _draw_sr(ax, traj, table, step, gamma, label, cmap) -> None:
    """One agent's successor representation as a heatmap over the grid."""
    ax.clear()
    grid = [[0.0] * traj.width for _ in range(traj.height)]
    cells = table.get(step, {}).get(gamma, {})
    for (x, y), value in cells.items():
        if 0 <= y < traj.height and 0 <= x < traj.width:
            grid[y][x] = value

    ax.imshow(grid, cmap=cmap, vmin=0.0, vmax=1.0, interpolation="nearest")
    for y, row in enumerate(traj.maze):
        for x, ch in enumerate(row):
            if ch == WALL:
                ax.add_patch(
                    patches.Rectangle((x - 0.5, y - 0.5), 1, 1, color="#3a3a3a")
                )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{label} SR (gamma={gamma})\n{len(cells)} cells", fontsize=10)


def render_gif(
    traj: Trajectory,
    out_path: str,
    gamma: Optional[str] = None,
    fps: int = 4,
    max_steps: Optional[int] = None,
) -> str:
    """Write the replay as an animated GIF. Returns the path written."""
    available = traj.gammas()
    if gamma is None:
        gamma = available[-1] if available else None
    if gamma is not None and available and gamma not in available:
        raise ValueError(f"gamma {gamma!r} not in file; available: {available}")

    n_steps = len(traj.achiever_pos)
    if max_steps is not None:
        n_steps = min(n_steps, max_steps)
    if n_steps == 0:
        raise ValueError("trajectory contains no steps to render")

    has_sr = bool(available)
    n_panels = 3 if has_sr else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(4.2 * n_panels, 4.6))
    if n_panels == 1:
        axes = [axes]

    writer = PillowWriter(fps=fps)
    with writer.saving(fig, out_path, dpi=110):
        for step in range(n_steps):
            _draw_grid(axes[0], traj, step)
            if has_sr:
                _draw_sr(axes[1], traj, traj.achiever_sr, step, gamma, "achiever", "Blues")
                _draw_sr(axes[2], traj, traj.blocker_sr, step, gamma, "blocker", "Oranges")
            fig.tight_layout()
            writer.grab_frame()
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a two-player AchieverBlocker trajectory as a GIF."
    )
    parser.add_argument("data_file", help="Path to a trajectory .txt file")
    parser.add_argument("-o", "--output", default=None, help="Output .gif path")
    parser.add_argument(
        "--gamma", default=None,
        help="SR discount to display (default: the largest in the file)",
    )
    parser.add_argument("--fps", type=int, default=4, help="Frames per second")
    parser.add_argument(
        "--max-steps", type=int, default=None, help="Render only the first N steps"
    )
    args = parser.parse_args()

    traj = parse_trajectory(args.data_file)
    out = args.output or os.path.splitext(os.path.basename(args.data_file))[0] + ".gif"

    print(f"maze        : {traj.width}x{traj.height}")
    print(f"steps       : {len(traj.achiever_pos)}")
    print(f"keys/doors  : {sorted(traj.keys)} / {sorted(traj.doors)}")
    print(f"SR gammas   : {traj.gammas() or 'none in file'}")
    path = render_gif(
        traj, out, gamma=args.gamma, fps=args.fps, max_steps=args.max_steps
    )
    print(f"wrote       : {path}")


if __name__ == "__main__":
    main()
