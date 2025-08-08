import numpy as np
import sys
import os

# Add the lib directory to the path
lib_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "lib")
sys.path.insert(0, lib_path)

# Add the env directory to the path
env_path = os.path.join(lib_path, "env")
sys.path.insert(0, env_path)

from utils import set_seed

# Add current directory for config import
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from config import Config

# Set seed using Config default value
config = Config()
set_seed(config.seed)


class RandomAgent:
    """
    Random Blocker Agent for AchieverBlocker environment.

    This agent uses movement actions (0-4: up, right, down, left, stay)
    and the "broken" action (5) to end the game when positioned at a door.
    """

    def __init__(self):
        """
        Initialize random blocker agent.
        """
        self.env = None

        # All possible actions: up(0), right(1), down(2), left(3), stay(4), break(5)
        self.all_actions = [0, 1, 2, 3, 4, 5]

    def get_action(self, obs):
        """
        Get action for blocker agent.
        Returns random action from the full action space.

        Args:
            obs: Environment observation

        Returns:
            action: Random action from available action space
        """
        # Randomly choose from all available actions
        return np.random.choice(self.all_actions)

    def set_env(self, env):
        """Set environment reference for action decisions"""
        self.env = env

    def reset(self):
        """Reset agent state for new episode"""
        pass