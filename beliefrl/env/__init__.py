"""Environment access for the core.

`lib/env/gym_minigrid` is a plain directory rather than an installed package, so
it has to be placed on `sys.path` before `import gym_minigrid` resolves. Doing
that here, once, replaces the `sys.path.insert` preamble that was repeated in
every agent module.
"""

import os
import sys

_LIB_ENV = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "lib", "env")
)

if _LIB_ENV not in sys.path:
    sys.path.insert(0, _LIB_ENV)
