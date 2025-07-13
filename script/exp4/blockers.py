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
        self.env = None
        self.target_door_color = None
        self.target_door_pos = None
        self.achiever_has_key = False
        self.path_to_door = []
        self.current_path_index = 0

    def get_action(self, obs):
        """
        Get action for goal-directed blocker agent.

        Args:
            obs: Environment observation

        Returns:
            action: Movement action (0-4) or break action (5)
        """
        if self.env is None:
            return 4  # Stay if no environment reference

        # Check if achiever has picked up a key for the first time
        if not self.achiever_has_key:
            self._check_achiever_key_pickup()

        # If achiever hasn't picked up a key yet, stay in place
        if not self.achiever_has_key:
            return 4  # Stay

        # If we don't have a target yet, infer it from achiever's key
        if self.target_door_color is None:
            self._infer_target_door()

        # If we're at the target door, break it
        if self._at_target_door():
            return 5  # Break action

        # Navigate to target door
        return self._navigate_to_door()

    def _check_achiever_key_pickup(self):
        """Check if achiever has picked up any key."""
        if hasattr(self.env, 'achiever_keys') and len(self.env.achiever_keys) > 0:
            self.achiever_has_key = True

    def _infer_target_door(self):
        """Infer target door color from achiever's first key."""
        if hasattr(self.env, 'achiever_keys') and len(self.env.achiever_keys) > 0:
            # Use the first key color as the inferred target
            first_key_color = self.env.achiever_keys[0]
            self.target_door_color = first_key_color
            self.target_door_pos = self._find_door_position(first_key_color)
            
            # Plan path to target door
            if self.target_door_pos:
                self.path_to_door = self._find_path_to_door()
                self.current_path_index = 0

    def _find_door_position(self, color):
        """Find position of door with given color."""
        for x in range(self.env.grid.width):
            for y in range(self.env.grid.height):
                obj = self.env.grid.get(x, y)
                if isinstance(obj, Door) and obj.color == color:
                    return (x, y)
        return None

    def _at_target_door(self):
        """Check if blocker is at the target door position."""
        if self.target_door_pos is None:
            return False
        blocker_pos = tuple(self.env.blocker_pos)
        return blocker_pos == self.target_door_pos

    def _find_path_to_door(self):
        """Find path from current position to target door using BFS."""
        if self.target_door_pos is None:
            return []

        start = tuple(self.env.blocker_pos)
        goal = self.target_door_pos
        
        if start == goal:
            return []

        # BFS to find path
        queue = deque([(start, [])])
        visited = {start}
        
        # Movement directions: up, right, down, left
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        
        while queue:
            (x, y), path = queue.popleft()
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                
                # Check bounds
                if 0 <= nx < self.env.grid.width and 0 <= ny < self.env.grid.height:
                    if (nx, ny) not in visited:
                        # Check if position is walkable (not a wall)
                        obj = self.env.grid.get(nx, ny)
                        if obj is None or isinstance(obj, (Key, Door)):
                            # Don't block achiever's position
                            achiever_pos = tuple(self.env.achiever_pos)
                            if (nx, ny) != achiever_pos:
                                visited.add((nx, ny))
                                new_path = path + [(nx, ny)]
                                
                                if (nx, ny) == goal:
                                    return new_path
                                
                                queue.append(((nx, ny), new_path))
        
        return []  # No path found

    def _navigate_to_door(self):
        """Navigate to target door following planned path."""
        if not self.path_to_door or self.current_path_index >= len(self.path_to_door):
            # Recalculate path if needed
            self.path_to_door = self._find_path_to_door()
            self.current_path_index = 0
            
            if not self.path_to_door:
                return 4  # Stay if no path

        # Get next position in path
        target_pos = self.path_to_door[self.current_path_index]
        current_pos = tuple(self.env.blocker_pos)
        
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
            return self._navigate_to_door()
        
        # Advance path index for next step
        self.current_path_index += 1
        
        return action

    def set_env(self, env):
        """Set environment reference for action decisions."""
        self.env = env

    def reset(self):
        """Reset agent state for new episode."""
        self.target_door_color = None
        self.target_door_pos = None
        self.achiever_has_key = False
        self.path_to_door = []
        self.current_path_index = 0
