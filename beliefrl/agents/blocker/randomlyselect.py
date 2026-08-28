from collections import deque

import numpy as np
class RandomlySelectedAgent:
    """
    Randomly selected Blocker Agent for AchieverBlocker environment.
    Level-0 Reasoning Algorithm

    Strategy (Multi-attempt):
    1. Select the target color randomly from remaining doors
    2. Navigate to target door
    3. Use break action (5) to attempt breaking
    4. If game continues (wrong door), select from remaining doors and repeat
    """

    def __init__(self, stay_probability=0.7):
        """
        Initialize randomly selected blocker agent.
        """
        self.target_door_color = None
        self.target_door_pos = None
        self.path_to_door = []
        self.current_path_index = 0
        self.grid = None
        self.blocker_pos = None
        self.achiever_pos = None
        self.target_selected = False

        # Grid dimensions - will be set from observations
        self.width = None
        self.height = None

        # Stay probability during navigation
        self.stay_probability = stay_probability

        # Multi-attempt tracking
        self.tried_doors = set()  # Track which doors have been attempted
        self.available_doors = {"red", "green", "blue", "yellow"}
        self.just_attempted_break = False
        self.last_action = None

    def get_action(self, obs):
        """
        Get action for randomly selected blocker agent.

        Args:
            obs: Environment observation dict with keys:
                - 'blocker': blocker's visual observation
                - 'achiever_keys': array of keys achiever has collected
                - 'achiever_pos': achiever's position
                - 'blocker_pos': blocker's position
                - 'door_positions': dict of door positions by color

        Returns:
            action: Movement action (0-4) or break action (5)
        """
        if obs is None:
            return 4  # Stay if no observation

        # Update internal state from observations
        self._update_from_obs(obs)

        # Check if we just attempted to break and game is still continuing
        if self.just_attempted_break:
            # Game didn't end, so we broke the wrong door
            # Mark current target as tried and select new target
            if self.target_door_color:
                self.tried_doors.add(self.target_door_color)
            self._reset_for_new_attempt()
            self.just_attempted_break = False

        # Select target door randomly if not already selected
        if not self.target_selected:
            self._select_random_target_door(obs)

        # If we're at the target door, break it
        if self._at_target_door():
            self.just_attempted_break = True
            self.last_action = 5
            return 5  # Break action

        # Navigate to target door
        action = self._navigate_to_door(obs)
        self.last_action = action
        return action

    def _update_from_obs(self, obs):
        """Update internal state from observations."""
        # Update grid dimensions from observations
        if "grid_info" in obs and self.width is None:
            self.width = obs["grid_info"]["width"]
            self.height = obs["grid_info"]["height"]

        self.blocker_pos = tuple(obs["blocker_pos"])
        self.achiever_pos = tuple(obs["achiever_pos"])
        # Extract grid from blocker's visual observation
        self.grid = obs["blocker"]

    def _select_random_target_door(self, obs):
        """Select target door color randomly from remaining untried doors."""
        # Get remaining doors that haven't been tried
        remaining_doors = list(self.available_doors - self.tried_doors)

        if not remaining_doors:
            # All doors have been tried, reset and start over
            self.tried_doors.clear()
            remaining_doors = list(self.available_doors)

        # Randomly select a door color from remaining doors
        self.target_door_color = np.random.choice(remaining_doors)

        # Find the position of the selected door
        self.target_door_pos = self._find_door_position_from_obs(
            self.target_door_color, obs
        )

        # Plan path to target door if position is found
        if self.target_door_pos:
            self.path_to_door = self._find_path_to_door(obs)
            self.current_path_index = 0
            self.target_selected = True

    def _reset_for_new_attempt(self):
        """Reset targeting state for a new attempt."""
        self.target_selected = False
        self.target_door_color = None
        self.target_door_pos = None
        self.path_to_door = []
        self.current_path_index = 0

    def _find_door_position_from_obs(self, color, obs):
        """Find position of door with given color from observations."""
        # Use door positions from observations
        if obs and "door_positions" in obs:
            return obs["door_positions"].get(color, None)
        return None

    def _at_target_door(self):
        """Check if blocker is at the target door position."""
        if self.target_door_pos is None:
            return False
        return self.blocker_pos == self.target_door_pos

    def _find_path_to_door(self, obs=None):
        """Find path from current position to target door using BFS with wall avoidance."""
        if self.target_door_pos is None:
            return []

        start = self.blocker_pos
        goal = self.target_door_pos

        if start == goal:
            return []

        # Get grid size from instance dimensions
        grid_size = self.width if self.width is not None else 9  # fallback

        # Get wall positions from observations
        wall_positions = set()
        if obs and "wall_positions" in obs:
            wall_positions = set(tuple(pos) for pos in obs["wall_positions"])

        # BFS to find path
        queue = deque([(start, [])])
        visited = {start}

        # Movement directions: up, right, down, left
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]

        while queue:
            (x, y), path = queue.popleft()

            for dx, dy in directions:
                nx, ny = x + dx, y + dy

                # Check bounds using grid size from observations
                if 0 <= nx < grid_size and 0 <= ny < grid_size:
                    if (nx, ny) not in visited:
                        # Check if position is walkable
                        if self._is_walkable_position(nx, ny, wall_positions, obs):
                            visited.add((nx, ny))
                            new_path = path + [(nx, ny)]

                            if (nx, ny) == goal:
                                return new_path

                            queue.append(((nx, ny), new_path))

        return []  # No path found

    def _is_walkable_position(self, x, y, wall_positions, obs):
        """Check if a position is walkable (not a wall, not occupied by achiever)."""
        pos = (x, y)

        # Check if position is a wall
        if pos in wall_positions:
            return False

        # Avoid achiever position to prevent collision
        if pos == self.achiever_pos:
            return False

        # Goal door position is always walkable (we want to reach it)
        if pos == self.target_door_pos:
            return True

        # All other positions are walkable (keys, empty spaces, other doors)
        return True

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
        self.path_to_door = []
        self.current_path_index = 0
        self.grid = None
        self.blocker_pos = None
        self.achiever_pos = None
        self.target_selected = False
        # Reset multi-attempt state
        self.tried_doors.clear()
        self.just_attempted_break = False
        self.last_action = None
        # Keep width/height since they don't change between episodes
