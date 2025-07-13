import numpy as np
import sys
import os
from collections import deque

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


class GoalDirectAgent:
    """
    Goal-directed Blocker Agent for AchieverBlocker environment.
    
    Strategy:
    1. Stay (action 4) until achiever picks up first key
    2. Infer target door color from achiever's first key
    3. Navigate to target door
    4. Use break action (5) to end game
    """

    def __init__(self):
        """Initialize goal-directed blocker agent."""
        self.target_door_color = None
        self.target_door_pos = None
        self.achiever_has_key = False
        self.path_to_door = []
        self.current_path_index = 0
        self.grid = None
        self.blocker_pos = None
        self.achiever_pos = None

    def get_action(self, obs):
        """
        Get action for goal-directed blocker agent.

        Args:
            obs: Environment observation dict with keys:
                - 'blocker': blocker's visual observation
                - 'achiever_keys': array of keys achiever has collected
                - 'achiever_pos': achiever's position
                - 'blocker_pos': blocker's position

        Returns:
            action: Movement action (0-4) or break action (5)
        """
        if obs is None:
            return 4  # Stay if no observation

        # Update internal state from observations
        self._update_from_obs(obs)

        # Check if achiever has picked up a key for the first time
        if not self.achiever_has_key:
            self._check_achiever_key_pickup(obs)

        # If achiever hasn't picked up a key yet, stay in place
        if not self.achiever_has_key:
            return 4  # Stay

        # If we don't have a target yet, infer it from achiever's key
        if self.target_door_color is None:
            self._infer_target_door(obs)

        # If we're at the target door, break it
        if self._at_target_door():
            return 5  # Break action

        # Navigate to target door
        return self._navigate_to_door(obs)

    def _update_from_obs(self, obs):
        """Update internal state from observations."""
        self.blocker_pos = tuple(obs['blocker_pos'])
        self.achiever_pos = tuple(obs['achiever_pos'])
        # Extract grid from blocker's visual observation
        self.grid = obs['blocker']

    def _check_achiever_key_pickup(self, obs):
        """Check if achiever has picked up any key."""
        achiever_keys = obs['achiever_keys']
        if len(achiever_keys) > 0 and achiever_keys.sum() > 0:
            self.achiever_has_key = True

    def _infer_target_door(self, obs):
        """Infer target door color from achiever's first key."""
        achiever_keys = obs['achiever_keys']
        if len(achiever_keys) > 0 and achiever_keys.sum() > 0:
            # Find first key collected (first non-zero index)
            # Color mapping: 0=red, 1=green, 2=blue, 3=yellow
            color_map = ["red", "green", "blue", "yellow"]
            first_key_idx = None
            for i, has_key in enumerate(achiever_keys):
                if has_key > 0:
                    first_key_idx = i
                    break
            
            if first_key_idx is not None and first_key_idx < len(color_map):
                first_key_color = color_map[first_key_idx]
                self.target_door_color = first_key_color
                self.target_door_pos = self._find_door_position_from_obs(first_key_color, obs)
                
                # Plan path to target door
                if self.target_door_pos:
                    self.path_to_door = self._find_path_to_door(obs)
                    self.current_path_index = 0

    def _find_door_position_from_obs(self, color, obs):
        """Find position of door with given color from observations."""
        # Use door positions from observations if available
        if obs and 'door_positions' in obs:
            return obs['door_positions'].get(color, None)
        
        # Fallback to hardcoded positions if observations don't have door info
        # Common door positions in 9x9 grid (based on environment structure)
        door_positions = {
            "red": (4, 8),    # bottom center
            "green": (4, 0),  # top center  
            "blue": (8, 4),   # right center
            "yellow": (0, 4)  # left center
        }
        return door_positions.get(color, None)

    def _at_target_door(self):
        """Check if blocker is at the target door position."""
        if self.target_door_pos is None:
            return False
        return self.blocker_pos == self.target_door_pos

    def _find_path_to_door(self, obs=None):
        """Find path from current position to target door using BFS."""
        if self.target_door_pos is None:
            return []

        start = self.blocker_pos
        goal = self.target_door_pos
        
        if start == goal:
            return []

        # Get grid size from observations or use default
        grid_size = 9  # default
        if obs and 'grid_info' in obs:
            grid_size = obs['grid_info']['width']

        # BFS to find path
        queue = deque([(start, [])])
        visited = {start}
        
        # Movement directions: up, right, down, left
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        
        # For now, use a simplified pathfinding that assumes open space
        # This will need to be improved when implementing full partial observability
        # to parse the visual grid observation properly
        
        while queue:
            (x, y), path = queue.popleft()
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                
                # Check bounds using grid size from observations
                if 0 <= nx < grid_size and 0 <= ny < grid_size:
                    if (nx, ny) not in visited:
                        # Simplified walkability check - avoid achiever position
                        if (nx, ny) != self.achiever_pos:
                            visited.add((nx, ny))
                            new_path = path + [(nx, ny)]
                            
                            if (nx, ny) == goal:
                                return new_path
                            
                            queue.append(((nx, ny), new_path))
        
        return []  # No path found

    def _navigate_to_door(self, obs=None):
        """Navigate to target door following planned path."""
        if not self.path_to_door or self.current_path_index >= len(self.path_to_door):
            # Recalculate path if needed
            self.path_to_door = self._find_path_to_door(obs)
            self.current_path_index = 0
            
            if not self.path_to_door:
                return 4  # Stay if no path

        # Get next position in path
        target_pos = self.path_to_door[self.current_path_index]
        current_pos = self.blocker_pos
        
        # Calculate direction to move
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        
        # Move towards target
        if dy < 0:  # Move up
            action = 0
        elif dx > 0:  # Move right
            action = 1
        elif dy > 0:  # Move down
            action = 2
        elif dx < 0:  # Move left
            action = 3
        else:
            # Already at target position, advance to next
            self.current_path_index += 1
            return self._navigate_to_door(obs)
        
        # Advance path index for next step
        self.current_path_index += 1
        
        return action

    def set_env(self, env):
        """Set environment reference for action decisions (legacy method)."""
        # This method is now deprecated since we use observations
        pass

    def reset(self):
        """Reset agent state for new episode."""
        self.target_door_color = None
        self.target_door_pos = None
        self.achiever_has_key = False
        self.path_to_door = []
        self.current_path_index = 0
        self.grid = None
        self.blocker_pos = None
        self.achiever_pos = None
