"""Re-exports of the gym_minigrid primitives the agents use.

Importing this module guarantees `lib/env` is on `sys.path` first, so agent
modules can simply do:

    from beliefrl.env.minigrid import Key, Door, Wall, Grid

instead of each rebuilding the path by hand.
"""

from . import _LIB_ENV  # noqa: F401  (import for the sys.path side effect)

from gym_minigrid.minigrid import Door, Grid, Key, Wall

__all__ = ["Door", "Grid", "Key", "Wall"]
