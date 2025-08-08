import numpy as np
import sys
import os

# Add current directory for config import
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from config import Config


class RandomAgent:
    """A random agent that explores the AchieverBlocker environment.

    Achiever uses 7-action space: up, right, down, left, stay, pickup, toggle
    (no "done" action - that's only for blockers)
    """

    def __init__(self, action_space=None, movement_prob=0.9):
        # Get action space from config if not provided
        if action_space is None:
            try:
                from config import Config

                config = Config()
                action_space = config.model_config["achiever_action_space"]
            except:
                action_space = 7  # Fallback default
        self.action_space = action_space
        self.movement_prob = movement_prob

    def get_action(self, obs):
        """Return a random action with bias towards movement (automatic pickup/door opening)."""
        rand = np.random.random()

        if rand < self.movement_prob:
            # Focus on movement actions: up, right, down, left, stay
            return np.random.choice(
                [0, 1, 2, 3, 4]
            )  # 0=up, 1=right, 2=down, 3=left, 4=stay
        else:
            # Rarely use pickup or toggle actions (mostly unnecessary due to automatic behavior)
            # But keep them for compatibility
            return np.random.choice([5, 6])  # 5=pickup, 6=toggle

    def analyze_feedback(self, reward, done):
        """Random agent doesn't learn from feedback."""
        pass

    def reset(self):
        """Reset agent state for new episode"""
        pass