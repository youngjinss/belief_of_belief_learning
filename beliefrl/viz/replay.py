"""Two-player trajectory replay: grid, both agents, and their SR beliefs.

Renders a saved AchieverBlocker trajectory as an animated GIF.

By default it drives the environment's own MiniGrid renderer, so the output is
the real game image: doors set into the walls, key icons in the room, and both
agents as triangles. `--schematic` switches to a matplotlib diagram instead,
which is what `--sr` needs to show each agent's successor representation.

This exists because `script/exp*/simulate_trajectory.py` cannot produce such a
video. It replays through MiniGrid's own renderer, which draws only the achiever
(see its own comment, "currently shows only achiever due to MiniGrid
limitation"), and its frames barely change, so PIL's GIF writer collapses 101
captured frames down to 9. It also builds a `-v1` environment for `-v2` data,
and its SR parser returns nothing. Rendering straight from the trajectory file
sidesteps all four problems: the file already holds everything needed.

    python -m beliefrl.viz.replay data/.../test0.txt -o replay.gif
    python -m beliefrl.viz.replay data/.../test0.txt --sr --gamma 0.9 --fps 5
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

# Letter case follows env_to_maze_format in beliefrl/data/generation.py:
# keys are written UPPERCASE and sit inside the room, doors are lowercase and
# sit in the surrounding wall. Colours are red, green, blue, yellow in order.
KEY_COLORS = {"A": "red", "B": "green", "C": "blue", "D": "gold"}
DOOR_COLORS = {"a": "red", "b": "green", "c": "blue", "d": "gold"}

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
                facecolor=DOOR_COLORS[ch], edgecolor="black", linewidth=1.5,
            )
        )
        # Show the exact character the trajectory file uses for this door.
        ax.text(
            x, y, ch, ha="center", va="center",
            fontsize=10, fontweight="bold", color="white",
        )

    for ch, (x, y) in traj.keys.items():
        ax.add_patch(
            patches.Circle(
                (x, y), 0.26, facecolor=KEY_COLORS[ch],
                edgecolor="black", linewidth=1.2,
            )
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
    dpi: int = 110,
    show_sr: bool = False,
) -> str:
    """Write the replay as an animated GIF. Returns the path written.

    By default this renders the game alone. Pass show_sr=True to add each
    agent's successor-representation heatmap beside the grid.
    """
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

    has_sr = show_sr and bool(available)
    n_panels = 3 if has_sr else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(5.2 if n_panels == 1 else 4.2 * n_panels, 5.2 if n_panels == 1 else 4.6))
    if n_panels == 1:
        axes = [axes]

    writer = PillowWriter(fps=fps)
    with writer.saving(fig, out_path, dpi=dpi):
        for step in range(n_steps):
            _draw_grid(axes[0], traj, step)
            if has_sr:
                _draw_sr(axes[1], traj, traj.achiever_sr, step, gamma, "achiever", "Blues")
                _draw_sr(axes[2], traj, traj.blocker_sr, step, gamma, "blocker", "Oranges")
            fig.tight_layout()
            writer.grab_frame()
    plt.close(fig)
    return out_path


# --- native MiniGrid rendering -------------------------------------------

# The v2 environment's own render() draws both agents (achiever as a red
# triangle, blocker as a blue one) through gym_minigrid.rendering.Renderer.
# Driving it directly gives the real game image rather than a schematic.
# The "shows only achiever" comment in script/exp*/simulate_trajectory.py
# refers to the v1 environment that script builds, not to v2.

_KEY_LETTER_COLORS = {"A": "red", "B": "green", "C": "blue", "D": "yellow"}
_DOOR_LETTER_COLORS = {"a": "red", "b": "green", "c": "blue", "d": "yellow"}


def _build_env(traj: "Trajectory"):
    """Recreate the environment for this maze, with its grid rebuilt exactly."""
    import numpy as np

    from beliefrl.env import _LIB_ENV  # noqa: F401  (puts lib/env on sys.path)
    from gym_minigrid.envs.achiever_blocker_v2 import (
        AchieverBlocker5x5EnvV2,
        AchieverBlocker9x9EnvV2,
        AchieverBlocker11x11EnvV2,
    )
    from gym_minigrid.minigrid import Door, Grid, Key, Wall

    sizes = {5: AchieverBlocker5x5EnvV2, 9: AchieverBlocker9x9EnvV2,
             11: AchieverBlocker11x11EnvV2}
    if traj.width not in sizes:
        raise ValueError(f"no v2 environment for a {traj.width}x{traj.height} maze")

    env = sizes[traj.width](
        preference={"red": 1.0, "green": 0.5, "blue": 0.25, "yellow": 0.125},
        cost={"red": 0.25, "green": 0.25, "blue": 0.25, "yellow": 0.25},
        max_steps=max(traj.length, 1) + 1,
        observability="full",
        partial_view_size=3,
    )
    env.seed(0)
    env.reset()

    env.grid = Grid(traj.width, traj.height)
    for y, row in enumerate(traj.maze):
        for x, ch in enumerate(row):
            if ch == WALL:
                env.grid.set(x, y, Wall())
            elif ch in _KEY_LETTER_COLORS:
                env.grid.set(x, y, Key(_KEY_LETTER_COLORS[ch]))
            elif ch in _DOOR_LETTER_COLORS:
                env.grid.set(x, y, Door(_DOOR_LETTER_COLORS[ch], is_locked=True))
            else:
                env.grid.set(x, y, None)
    return env, np


def render_gif_native(
    traj: "Trajectory",
    out_path: str,
    fps: int = 4,
    max_steps: Optional[int] = None,
    scale: int = 3,
) -> str:
    """Render the replay through the environment's own MiniGrid renderer."""
    from PIL import Image

    env, np = _build_env(traj)

    n_steps = len(traj.achiever_pos)
    if max_steps is not None:
        n_steps = min(n_steps, max_steps)
    if n_steps == 0:
        raise ValueError("trajectory contains no steps to render")

    frames = []
    for step in range(n_steps):
        env.achiever_pos = np.array(traj.achiever_pos[step])
        env.blocker_pos = np.array(traj.blocker_pos[step])
        array = env.render(mode="rgb_array")
        image = Image.fromarray(array)
        if scale != 1:
            image = image.resize(
                (image.width * scale, image.height * scale), Image.NEAREST
            )
        frames.append(image)

    frames[0].save(
        out_path, save_all=True, append_images=frames[1:],
        duration=int(1000 / max(fps, 1)), loop=0,
    )
    return out_path


def render_gif_live(
    out_path: str,
    size: str = "9x9",
    seed: int = 42,
    fps: int = 4,
    max_steps: int = 40,
    scale: int = 3,
    observability: str = "full",
) -> str:
    """Play a fresh game with the value agents and render it.

    The trajectory files under data/ were generated before the agents' grid
    frame was fixed, so replaying them shows agents that stall. This runs the
    current agents live instead.
    """
    import numpy as np
    from PIL import Image

    from beliefrl.agents.achiever.level0value import Level0ValueAchiever
    from beliefrl.agents.blocker.level0value import Level0ValueBlocker
    from beliefrl.env import _LIB_ENV  # noqa: F401
    from gym_minigrid.envs.achiever_blocker_v2 import (
        AchieverBlocker5x5EnvV2,
        AchieverBlocker9x9EnvV2,
        AchieverBlocker11x11EnvV2,
    )

    envs = {"5x5": AchieverBlocker5x5EnvV2, "9x9": AchieverBlocker9x9EnvV2,
            "11x11": AchieverBlocker11x11EnvV2}
    preference = {"red": 0.32, "green": 0.14, "blue": 1.0, "yellow": 0.16}
    cost = {"red": 0.5, "green": 0.1, "blue": 0.16, "yellow": 0.22}

    # The value agents sample actions from np.random, so the RNGs must be
    # seeded too -- seeding only the environment leaves the rollout
    # non-reproducible. Launch with PYTHONHASHSEED=0 as well: several agents
    # branch on set iteration order, which no in-process seeding can pin.
    import random as _random

    _random.seed(seed)
    np.random.seed(seed)

    env = envs[size](
        preference=preference, cost=cost, max_steps=max_steps + 1,
        observability=observability, partial_view_size=6,
    )
    env.seed(seed)
    reset = env.reset()
    obs = reset[0] if isinstance(reset, tuple) else reset

    achiever = Level0ValueAchiever(
        observability=observability, goal_rewards=preference
    )
    blocker = Level0ValueBlocker(observability=observability)
    achiever.reset()
    blocker.reset()

    frames = []
    for _ in range(max_steps):
        array = env.render(mode="rgb_array")
        image = Image.fromarray(array)
        if scale != 1:
            image = image.resize(
                (image.width * scale, image.height * scale), Image.NEAREST
            )
        frames.append(image)

        achiever.update_observation(obs)
        blocker.update_observation(obs)
        # env.step unpacks `achiever_action, blocker_action = actions`.
        obs, _, terminated, truncated, _ = env.step(
            (achiever.get_action(obs), blocker.get_action(obs))
        )
        if terminated or truncated:
            array = env.render(mode="rgb_array")
            image = Image.fromarray(array)
            if scale != 1:
                image = image.resize(
                    (image.width * scale, image.height * scale), Image.NEAREST
                )
            frames.append(image)
            break

    frames[0].save(
        out_path, save_all=True, append_images=frames[1:],
        duration=int(1000 / max(fps, 1)), loop=0,
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a two-player AchieverBlocker trajectory as a GIF."
    )
    parser.add_argument(
        "data_file", nargs="?", default=None,
        help="Path to a trajectory .txt file (omit when using --live)",
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Play a fresh game with the current agents instead of replaying a file",
    )
    parser.add_argument("--size", default="9x9", choices=["5x5", "9x9", "11x11"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-o", "--output", default=None, help="Output .gif path")
    parser.add_argument(
        "--gamma", default=None,
        help="SR discount to display (default: the largest in the file)",
    )
    parser.add_argument("--fps", type=int, default=4, help="Frames per second")
    parser.add_argument(
        "--max-steps", type=int, default=None, help="Render only the first N steps"
    )
    parser.add_argument("--dpi", type=int, default=110, help="Output resolution")
    parser.add_argument(
        "--sr", action="store_true",
        help="Schematic mode only: also show each agent's SR heatmap",
    )
    parser.add_argument(
        "--schematic", action="store_true",
        help="Use the matplotlib schematic instead of the game's own renderer",
    )
    parser.add_argument(
        "--scale", type=int, default=3,
        help="Native mode only: integer upscale of the game image",
    )
    args = parser.parse_args()

    if args.live:
        out = args.output or f"live_{args.size}.gif"
        path = render_gif_live(
            out, size=args.size, seed=args.seed, fps=args.fps,
            max_steps=args.max_steps or 40, scale=args.scale,
        )
        print(f"wrote       : {path}")
        return

    if not args.data_file:
        parser.error("a trajectory file is required unless --live is given")

    traj = parse_trajectory(args.data_file)
    out = args.output or os.path.splitext(os.path.basename(args.data_file))[0] + ".gif"

    print(f"maze        : {traj.width}x{traj.height}")
    print(f"steps       : {len(traj.achiever_pos)}")
    print(f"keys/doors  : {sorted(traj.keys)} / {sorted(traj.doors)}")
    print(f"SR gammas   : {traj.gammas() or 'none in file'}")
    if args.schematic or args.sr:
        path = render_gif(
            traj, out, gamma=args.gamma, fps=args.fps, max_steps=args.max_steps,
            dpi=args.dpi, show_sr=args.sr,
        )
    else:
        path = render_gif_native(
            traj, out, fps=args.fps, max_steps=args.max_steps, scale=args.scale,
        )
    print(f"wrote       : {path}")


if __name__ == "__main__":
    main()
