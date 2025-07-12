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
    
    This agent only uses movement actions (0-4: up, right, down, left, stay)
    and doesn't use pickup or toggle actions, as the blocker's role is to
    position itself strategically to block the achiever's access to doors.
    """
    
    def __init__(self, movement_prob=0.9):
        """
        Initialize random blocker agent.
        
        Args:
            movement_prob: Probability of choosing movement action vs staying
        """
        self.env = None
        self.movement_prob = movement_prob
        
        # Only use movement actions: up(0), right(1), down(2), left(3), stay(4)
        self.movement_actions = [0, 1, 2, 3, 4]
        
    def get_action(self, obs):
        """
        Get random action for blocker agent.
        Only returns movement actions (0-4): up, right, down, left, stay
        
        Args:
            obs: Environment observation
            
        Returns:
            action: Random movement action (0-4)
        """
        # Randomly choose between movement and staying
        rand = np.random.random()
        
        if rand < self.movement_prob:
            # Choose random movement action (up, right, down, left)
            return np.random.choice([0, 1, 2, 3])
        else:
            # Stay in place
            return 4
    
    def reset(self):
        """Reset agent state for new episode"""
        pass