import numpy as np
import sys
import os

# Add the lib directory to the path
lib_path = os.path.join(os.path.dirname(__file__), "..", "..", "lib")
sys.path.insert(0, lib_path)

# Add the env directory to the path
env_path = os.path.join(lib_path, "env")
sys.path.insert(0, env_path)

from gym_minigrid.minigrid import Key, Door, Wall


class RandomAgent:
    """
    Random Blocker Agent for AchieverBlocker environment.

    This agent uses movement actions (0-4: up, right, down, left, stay)
    and the "broken" action (5) to end the game when positioned at a door.
    """

    def __init__(self, movement_prob=0.9, broken_prob=0.1):
        """
        Initialize random blocker agent.

        Args:
            movement_prob: Probability of choosing movement action vs staying
            broken_prob: Probability of choosing "broken" action when at a door
        """
        self.env = None
        self.movement_prob = movement_prob
        self.broken_prob = broken_prob

        # Movement actions: up(0), right(1), down(2), left(3), stay(4)
        self.movement_actions = [0, 1, 2, 3, 4]

    def get_action(self, obs):
        """
        Get action for blocker agent.
        Returns movement actions (0-4): up, right, down, left, stay
        or "broken" action (5) when positioned at a door.

        Args:
            obs: Environment observation

        Returns:
            action: Movement action (0-4) or broken action (5)
        """
        # Check if blocker is at a door position and should consider "broken" action
        if self.env is not None:
            blocker_pos = getattr(self.env, "blocker_pos", None)
            if blocker_pos is not None:
                cell = self.env.grid.get(*blocker_pos)
                if isinstance(cell, Door):
                    # At a door position - consider "broken" action
                    if np.random.random() < self.broken_prob:
                        return 5  # "broken" action

        # Randomly choose between movement and staying
        rand = np.random.random()

        if rand < self.movement_prob:
            # Choose random movement action (up, right, down, left)
            return np.random.choice([0, 1, 2, 3])
        else:
            # Stay in place
            return 4

    def set_env(self, env):
        """Set environment reference for action decisions"""
        self.env = env

    def reset(self):
        """Reset agent state for new episode"""
        pass
