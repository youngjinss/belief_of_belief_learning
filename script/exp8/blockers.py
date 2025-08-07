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
from utils import set_seed

# Add current directory for config import
sys.path.append(os.path.dirname(__file__))
from config import Config
from value_agent import BaseValueAgent

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

        # Grid dimensions - will be set from observations
        self.width = None
        self.height = None

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
        # Update grid dimensions from observations
        if "grid_info" in obs and self.width is None:
            self.width = obs["grid_info"]["width"]
            self.height = obs["grid_info"]["height"]

        self.blocker_pos = tuple(obs["blocker_pos"])
        self.achiever_pos = tuple(obs["achiever_pos"])
        # Extract grid from blocker's visual observation
        self.grid = obs["blocker"]

    def _check_achiever_key_pickup(self, obs):
        """Check if achiever has picked up any key."""
        achiever_keys = obs["achiever_keys"]
        if len(achiever_keys) > 0 and achiever_keys.sum() > 0:
            self.achiever_has_key = True

    def _infer_target_door(self, obs):
        """Infer target door color from achiever's first key."""
        achiever_keys = obs["achiever_keys"]
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
                self.target_door_pos = self._find_door_position_from_obs(
                    first_key_color, obs
                )

                # Plan path to target door
                if self.target_door_pos:
                    self.path_to_door = self._find_path_to_door(obs)
                    self.current_path_index = 0

    def _find_door_position_from_obs(self, color, obs):
        """Find position of door with given color from observations."""
        # Use door positions from observations
        if obs and "door_positions" in obs:
            return obs["door_positions"].get(color, None)

        # No fallback - if door positions aren't in observations, return None
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
        self.achiever_has_key = False
        self.path_to_door = []
        self.current_path_index = 0
        self.grid = None
        self.blocker_pos = None
        self.achiever_pos = None
        # Keep width/height since they don't change between episodes
        # self.width = None
        # self.height = None


class Level0ValueBlocker(BaseValueAgent):
    """
    Level-0 Value-based Blocker Agent for AchieverBlocker environment.

    Strategy (Multi-attempt):
    1. Select the target color randomly from remaining doors
    2. Navigate to target door using value iteration planning
    3. Use break action (5) to attempt breaking
    4. If game continues (wrong door), select from remaining doors and repeat

    Uses BaseValueAgent planning approach but with RandomlySelectedAgent strategy.
    """

    def __init__(
        self,
        movement_cost=0.01,
        wall_penalty=2.0,
        conflict_penalty=2.0,
        gamma=0.99,
        temperature=0.1,
        q_value_clip=100,
    ):
        """
        Initialize Level-0 value-based blocker agent.
        """
        # Initialize base class
        super().__init__(
            observability="full",
            movement_cost=movement_cost,
            wall_penalty=wall_penalty,
            conflict_penalty=conflict_penalty,
            gamma=gamma,
            temperature=temperature,
            q_value_clip=q_value_clip,
        )

        # Blocker-specific attributes
        self.target_inferred_color = None
        self.target_door_pos = None
        self.path_to_door = []
        self.current_path_index = 0
        self.blocker_pos = None
        self.achiever_pos = None
        self.target_selected = False

        # Remove stay probability - use direct value iteration decisions

        # Multi-attempt tracking
        self.tried_doors = set()  # Track which doors have been attempted
        self.available_doors = {"red", "green", "blue", "yellow"}
        self.just_attempted_break = False
        self.last_action = None

    def _update_agent_position(self, obs):
        """Update blocker position from observations"""
        new_pos = tuple(obs["blocker_pos"])
        if new_pos != self.agent_pos:
            self.agent_pos = new_pos

    def _update_grid_reference(self, obs):
        """Update grid reference from observations"""
        self.grid = obs["blocker"]

    def _get_opponent_position(self, obs):
        """Get achiever position for conflict penalty"""
        if obs and "achiever_pos" in obs:
            return tuple(obs["achiever_pos"])
        return None

    def update_observation(self, obs):
        """Update agent's understanding of the environment"""
        if obs is None:
            return

        # Call base class update
        super().update_observation(obs)

        # Update blocker-specific state
        self.blocker_pos = self.agent_pos
        self.achiever_pos = self._get_opponent_position(obs)

    def get_action(self, obs):
        """
        Get action for Level-0 value-based blocker agent.

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
        self.update_observation(obs)

        # Check if we just attempted to break and game is still continuing
        if self.just_attempted_break:
            # Game didn't end, so we broke the wrong door
            # Mark current target as tried and select new target
            if self.target_inferred_color:
                self.tried_doors.add(self.target_inferred_color)
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

        # Navigate to target door using value iteration
        action = self._navigate_with_value_iteration(self.target_door_pos, obs)
        self.last_action = action
        return action

    def _select_random_target_door(self, obs):
        """Select target door color randomly from remaining untried doors."""
        # Get remaining doors that haven't been tried
        remaining_doors = list(self.available_doors - self.tried_doors)

        if not remaining_doors:
            # All doors have been tried, reset and start over
            self.tried_doors.clear()
            remaining_doors = list(self.available_doors)

        # Randomly select a door color from remaining doors
        self.target_inferred_color = np.random.choice(remaining_doors)

        # Find the position of the selected door
        self.target_door_pos = self._find_door_position_from_obs(
            self.target_inferred_color, obs
        )

        # Mark target as selected if position is found
        if self.target_door_pos:
            self.target_selected = True

    def _reset_for_new_attempt(self):
        """Reset targeting state for a new attempt."""
        self.target_selected = False
        self.target_inferred_color = None
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

    @property
    def target_door_color(self):
        """
        Get the current target door color for interaction checking.
        Uses target_inferred_color for Level0ValueBlocker.

        Returns:
            str: Current target door color
        """
        return self.target_inferred_color

    def set_env(self, env):
        """Set environment reference for action decisions (legacy method)."""
        # This method is now deprecated since we use observations
        pass

    def reset(self):
        """Reset agent state for new episode."""
        super().reset()
        self.target_inferred_color = None
        self.target_door_pos = None
        self.path_to_door = []
        self.current_path_index = 0
        self.blocker_pos = None
        self.achiever_pos = None
        self.target_selected = False
        # Reset multi-attempt state
        self.tried_doors.clear()
        self.just_attempted_break = False
        self.last_action = None
        # Keep width/height since they don't change between episodes


class Level1ValueBlocker(BaseValueAgent):
    """
    Level-1 Value-based Blocker Agent for AchieverBlocker environment.

    Simple "Stay, Watch, and Go" Strategy:

    Phase 1: Observation
    - Stay in place and wait until achiever picks up the first key
    - Store the observed key color

    Phase 2: Target inference and navigation
    - Infer that the first picked key is the target
    - Navigate to the corresponding door using value iteration

    Phase 3: Door breaking
    - Attempt to break the target door
    - If game continues (wrong door), wait for next key pickup
    - Use the second observed key as the new target and repeat

    The agent uses a simple reactive strategy without bluffing or prediction,
    relying solely on observed key pickups to infer the achiever's target.
    """

    def __init__(
        self,
        movement_cost=0.01,
        wall_penalty=2.0,
        conflict_penalty=2.0,
        gamma=0.99,
        temperature=0.1,
        q_value_clip=100,
    ):
        """Initialize Level-1 value-based blocker agent."""
        # Initialize base class
        super().__init__(
            observability="full",
            movement_cost=movement_cost,
            wall_penalty=wall_penalty,
            conflict_penalty=conflict_penalty,
            gamma=gamma,
            temperature=temperature,
            q_value_clip=q_value_clip,
        )

        # Inferred target
        self.target_color = None
        self.target_pos = None
        self.achiever_has_key = False

        # Navigation state
        self.path_to_door = []
        self.current_path_index = 0
        self.blocker_pos = None
        self.achiever_pos = None

        # Phase tracking
        self.phase = 1  # 1: wait and observe, 2: go to inferred door, 3: break door

        # Multi-attempt tracking
        self.observed_keys = []  # Track keys observed from achiever
        self.current_key_index = 0  # Index of current key being used for inference
        self.just_attempted_break = False
        self.last_action = None

    def _update_agent_position(self, obs):
        """Update blocker position from observations"""
        new_pos = tuple(obs["blocker_pos"])
        if new_pos != self.agent_pos:
            self.agent_pos = new_pos

    def _update_grid_reference(self, obs):
        """Update grid reference from observations"""
        self.grid = obs["blocker"]

    def _get_opponent_position(self, obs):
        """Get achiever position for conflict penalty"""
        if obs and "achiever_pos" in obs:
            return tuple(obs["achiever_pos"])
        return None

    def update_observation(self, obs):
        """Update agent's understanding of the environment"""
        if obs is None:
            return

        # Call base class update
        super().update_observation(obs)

        # Update blocker-specific state
        self.blocker_pos = self.agent_pos
        self.achiever_pos = self._get_opponent_position(obs)

    @property
    def target_door_color(self):
        """
        Get the current target door color for interaction checking.
        For Level1ValueBlocker, this returns the inferred target color.

        Returns:
            str: Current target door color
        """
        return self.target_color

    @property
    def target_inferred_color(self):
        """
        Get the current target inferred color for Level1ValueBlocker.
        This matches the interface expected by other parts of the system.

        Returns:
            str: Current target inferred color
        """
        return self.target_door_color

    def get_action(self, obs):
        """
        Get action for Level-1 value-based blocker agent.

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
        self.update_observation(obs)

        # Check if we just attempted to break and game is still continuing
        if self.just_attempted_break:
            # Game didn't end, so we broke the wrong door
            # Move to next observed key if available
            self.current_key_index += 1
            if self.current_key_index >= len(self.observed_keys):
                # No more observed keys, reset to Phase 1 to wait for more keys
                self.phase = 1
                self.target_color = None
                self.target_pos = None
            else:
                # Use next observed key, go back to Phase 2
                self.phase = 2
                self.target_color = None
                self.target_pos = None
            self.just_attempted_break = False

        # Check if achiever has picked up a key and store it
        self._check_and_store_achiever_keys(obs)

        # Phase 1: Wait and observe until achiever picks up a key
        if self.phase == 1:
            if self.achiever_has_key:
                self.phase = 2  # Move to navigation phase
                return self._handle_phase_2(obs)
            else:
                return 4  # Stay and wait

        # Phase 2: Navigate to inferred target door
        elif self.phase == 2:
            return self._handle_phase_2(obs)

        # Phase 3: Break the door
        elif self.phase == 3:
            self.just_attempted_break = True
            self.last_action = 5
            return 5  # Break action

        return 4  # Default: stay

    def _handle_phase_2(self, obs):
        """Handle phase 2: infer target and navigate to it using value iteration."""
        # If we don't have a target yet, infer it from current observed key
        if self.target_color is None:
            self._infer_target_from_observed_keys(obs)

        # Ensure we have a valid target before proceeding
        if self.target_pos is None:
            return 4  # Stay if we couldn't infer target

        # If we're at the target door, break it
        if self._at_target_door():
            self.phase = 3
            return 5  # Break action

        # Navigate to target door using value iteration
        return self._navigate_to_door_with_value_iteration(obs)

    def _at_target_door(self):
        """Check if blocker is at the target door position."""
        if self.target_pos is None:
            return False
        # Ensure both positions are tuples for comparison
        return tuple(self.blocker_pos) == tuple(self.target_pos)

    def _navigate_to_door_with_value_iteration(self, obs=None):
        """Navigate to target door using value iteration."""
        return self._navigate_with_value_iteration(self.target_pos, obs)

    def _find_door_position_from_obs(self, color, obs):
        """Find position of door with given color from observations."""
        # Use door positions from observations
        if obs and "door_positions" in obs:
            return obs["door_positions"].get(color, None)
        return None

    def set_env(self, env):
        """Set environment reference for action decisions (legacy method)."""
        # This method is now deprecated since we use observations
        pass

    def reset(self):
        """Reset agent state for new episode."""
        super().reset()
        # Inferred target
        self.target_color = None
        self.target_pos = None
        self.achiever_has_key = False

        # Navigation state
        self.path_to_door = []
        self.current_path_index = 0
        self.blocker_pos = None
        self.achiever_pos = None

        # Phase tracking
        self.phase = 1

        # Reset multi-attempt state
        self.observed_keys.clear()
        self.current_key_index = 0
        self.just_attempted_break = False
        self.last_action = None

        # Keep width/height since they don't change between episodes

    def _check_and_store_achiever_keys(self, obs):
        """Check for new achiever key pickups and store them in observed_keys."""
        if obs and "achiever_keys" in obs:
            achiever_keys = obs["achiever_keys"]
            color_map = ["red", "green", "blue", "yellow"]

            # Find any keys the achiever has and store color names
            for i, has_key in enumerate(achiever_keys):
                if has_key > 0 and i < len(color_map):
                    key_color = color_map[i]
                    if key_color not in self.observed_keys:
                        self.observed_keys.append(key_color)

            # Update achiever_has_key flag for backward compatibility
            if (
                len(achiever_keys) > 0
                and achiever_keys.sum() > 0
                and not self.achiever_has_key
            ):
                self.achiever_has_key = True

    def _infer_target_from_observed_keys(self, obs):
        """Infer target door color from current observed key."""
        if self.current_key_index < len(self.observed_keys):
            # Use the key at current_key_index for inference
            key_color = self.observed_keys[self.current_key_index]
            self.target_color = key_color

            # Find position of inferred door
            self.target_pos = self._find_door_position_from_obs(self.target_color, obs)


class RuleBasedAgent:
    """
    Rule-based Blocker Agent for AchieverBlocker environment.
    Level-1 Reasoning Algorithm

    Multi-attempt Strategy:
    1. Select random door color (bluffing) -> navigate to it
    2. Stay at door until achiever picks up first key
    3. Store observed key color in observed_keys list
    4. Infer target door from first observed key -> navigate and break
    5. If game continues (wrong door), remove first observed key
    6. Infer target from next observed key -> navigate and break
    7. Continue cycling through observed_keys until game ends
    """

    def __init__(self, stay_probability=0.7):
        """Initialize rule-based blocker agent."""
        # Phase 1: Initial random target
        self.initial_target_color = None
        self.initial_target_pos = None
        self.initial_target_selected = False

        # Phase 2: Final inferred target
        self.final_target_color = None
        self.final_target_pos = None
        self.achiever_has_key = False

        # Navigation state
        self.path_to_door = []
        self.current_path_index = 0
        self.grid = None
        self.blocker_pos = None
        self.achiever_pos = None

        # Phase tracking
        self.phase = (
            1  # 1: go to random door, 2: stay at door, 3: go to inferred door, 4: break
        )

        # Grid dimensions - will be set from observations
        self.width = None
        self.height = None

        # Stay probability during navigation
        self.stay_probability = stay_probability

        # Multi-attempt tracking
        self.observed_keys = []  # Track keys observed from achiever
        self.current_key_index = 0  # Index of current key being used for inference
        self.just_attempted_break = False
        self.last_action = None

    @property
    def target_door_color(self):
        """
        Get the current target door color for interaction checking.

        Constraint: Once final inference happens (after achiever picks up key),
        this should be fixed to the inferred color and not change anymore.

        Returns:
            str: Current target door color (initial random or final inferred)
        """
        # Once final target is inferred and fixed, always return that
        if self.final_target_color is not None:
            return self.final_target_color
        # Otherwise return initial random target
        return self.initial_target_color

    def get_action(self, obs):
        """
        Get action for rule-based blocker agent.

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
            # Move to next observed key if available
            self.current_key_index += 1
            if self.current_key_index >= len(self.observed_keys):
                # No more observed keys, reset to Phase 2 to wait for more keys
                self.phase = 2
                self.final_target_color = None
                self.final_target_pos = None
            else:
                # Use next observed key, go back to Phase 3
                self.phase = 3
                self.final_target_color = None
                self.final_target_pos = None
            self.just_attempted_break = False

        # Check if achiever has picked up a key and store it
        self._check_and_store_achiever_keys(obs)

        # Phase 1: Select initial random target and navigate to it
        if self.phase == 1:
            if not self.initial_target_selected:
                self._select_initial_random_target(obs)

            # Check if we've reached the initial target door
            if self._at_initial_target_door():
                self.phase = 2  # Move to waiting phase
                return 4  # Stay at the door
            else:
                return self._navigate_to_initial_door(obs)

        # Phase 2: Stay at initial door until achiever picks up key
        elif self.phase == 2:
            if self.achiever_has_key:
                self.phase = 3  # Move to inferring phase
                return self._handle_phase_3(obs)
            else:
                return 4  # Stay and wait

        # Phase 3: Infer final target from achiever's key and navigate to it
        elif self.phase == 3:
            return self._handle_phase_3(obs)

        # Phase 4: Break the final door
        elif self.phase == 4:
            self.just_attempted_break = True
            self.last_action = 5
            return 5  # Break action

        return 4  # Default: stay

    def _handle_phase_3(self, obs):
        """Handle phase 3: infer target and navigate to it."""
        # If we don't have a final target yet, infer it from current observed key
        if self.final_target_color is None:
            self._infer_final_target_from_observed_keys(obs)

        # Ensure we have a valid final target before proceeding
        if self.final_target_pos is None:
            return 4  # Stay if we couldn't infer target

        # If we're at the final target door, break it
        if self._at_final_target_door():
            self.phase = 4
            return 5  # Break action

        # Navigate to final target door
        return self._navigate_to_final_door(obs)

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

    def _select_initial_random_target(self, obs):
        """Select initial target door color randomly."""
        color_map = ["red", "green", "blue", "yellow"]

        # Randomly select a door color
        self.initial_target_color = np.random.choice(color_map)

        # Find the position of the selected door
        door_pos = self._find_door_position_from_obs(self.initial_target_color, obs)
        self.initial_target_pos = tuple(door_pos) if door_pos else None

        # Plan path to initial target door if position is found
        if self.initial_target_pos:
            self.path_to_door = self._find_path_to_door(self.initial_target_pos, obs)
            self.current_path_index = 0
            self.initial_target_selected = True

    def _at_initial_target_door(self):
        """Check if blocker is at the initial target door position."""
        if self.initial_target_pos is None:
            return False
        # Ensure both positions are tuples for comparison
        return tuple(self.blocker_pos) == tuple(self.initial_target_pos)

    def _at_final_target_door(self):
        """Check if blocker is at the final target door position."""
        if self.final_target_pos is None:
            return False
        # Ensure both positions are tuples for comparison
        return tuple(self.blocker_pos) == tuple(self.final_target_pos)

    def _navigate_to_initial_door(self, obs=None):
        """Navigate to initial target door following planned path."""
        return self._navigate_to_door_generic(
            self.initial_target_pos, obs, is_final=False
        )

    def _navigate_to_final_door(self, obs=None):
        """Navigate to final target door following planned path."""
        return self._navigate_to_door_generic(self.final_target_pos, obs, is_final=True)

    def _check_achiever_key_pickup(self, obs):
        """Check if achiever has picked up any key."""
        achiever_keys = obs["achiever_keys"]
        if len(achiever_keys) > 0 and achiever_keys.sum() > 0:
            self.achiever_has_key = True

    def _infer_final_target_door(self, obs):
        """Infer final target door color from achiever's first key."""
        achiever_keys = obs["achiever_keys"]
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
                self.final_target_color = first_key_color
                door_pos = self._find_door_position_from_obs(first_key_color, obs)
                self.final_target_pos = tuple(door_pos) if door_pos else None

                # Plan path to final target door
                if self.final_target_pos:
                    self.path_to_door = self._find_path_to_door(
                        self.final_target_pos, obs
                    )
                    self.current_path_index = 0

    def _find_door_position_from_obs(self, color, obs):
        """Find position of door with given color from observations."""
        # Use door positions from observations
        if obs and "door_positions" in obs:
            return obs["door_positions"].get(color, None)
        return None

    def _find_path_to_door(self, target_pos, obs=None):
        """Find path from current position to target door using BFS with wall avoidance."""
        if target_pos is None:
            return []

        start = self.blocker_pos
        goal = target_pos

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
                        if self._is_walkable_position(
                            nx, ny, wall_positions, target_pos, obs
                        ):
                            visited.add((nx, ny))
                            new_path = path + [(nx, ny)]

                            if (nx, ny) == goal:
                                return new_path

                            queue.append(((nx, ny), new_path))

        return []  # No path found

    def _is_walkable_position(self, x, y, wall_positions, target_pos, obs):
        """Check if a position is walkable (not a wall, not occupied by achiever)."""
        pos = (x, y)

        # Check if position is a wall
        if pos in wall_positions:
            return False

        # Avoid achiever position to prevent collision
        if pos == self.achiever_pos:
            return False

        # Goal door position is always walkable (we want to reach it)
        if pos == target_pos:
            return True

        # All other positions are walkable (keys, empty spaces, other doors)
        return True

    def _navigate_to_door_generic(self, target_pos, obs=None, is_final=False):
        """Navigate to target door following planned path."""
        if not self.path_to_door or self.current_path_index >= len(self.path_to_door):
            # Recalculate path if needed
            self.path_to_door = self._find_path_to_door(target_pos, obs)
            self.current_path_index = 0

            if not self.path_to_door:
                return 4  # Stay if no path

        # Use stay probability to decide whether to stay or move
        if np.random.random() < self.stay_probability and not is_final:
            return 4  # Stay

        # Get next position in path
        target_pos_in_path = self.path_to_door[self.current_path_index]
        current_pos = self.blocker_pos

        # Calculate direction to move
        dx = target_pos_in_path[0] - current_pos[0]
        dy = target_pos_in_path[1] - current_pos[1]

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
            return self._navigate_to_door_generic(target_pos, obs)

        # Advance path index for next step
        self.current_path_index += 1

        return action

    def set_env(self, env):
        """Set environment reference for action decisions (legacy method)."""
        # This method is now deprecated since we use observations
        pass

    def reset(self):
        """Reset agent state for new episode."""
        # Phase 1: Initial random target
        self.initial_target_color = None
        self.initial_target_pos = None
        self.initial_target_selected = False

        # Phase 2: Final inferred target
        self.final_target_color = None
        self.final_target_pos = None
        self.achiever_has_key = False

        # Navigation state
        self.path_to_door = []
        self.current_path_index = 0
        self.grid = None
        self.blocker_pos = None
        self.achiever_pos = None

        # Phase tracking
        self.phase = 1

        # Reset multi-attempt state
        self.observed_keys.clear()
        self.current_key_index = 0
        self.just_attempted_break = False
        self.last_action = None

        # Reset distance tracking
        self.distance_history.clear()
        self.reduction_rates.clear()
        self.consecutive_reductions.clear()
        self.timestep_counter = 0
        self.predicted_target_color = None

        # Keep width/height since they don't change between episodes

    def _check_and_store_achiever_keys(self, obs):
        """Check for new achiever key pickups and store them in observed_keys."""
        if obs and "achiever_keys" in obs:
            achiever_keys = obs["achiever_keys"]
            color_map = ["red", "green", "blue", "yellow"]

            # Find any keys the achiever has and store color names
            for i, has_key in enumerate(achiever_keys):
                if has_key > 0 and i < len(color_map):
                    key_color = color_map[i]
                    if key_color not in self.observed_keys:
                        self.observed_keys.append(key_color)

            # Update achiever_has_key flag for backward compatibility
            if (
                len(achiever_keys) > 0
                and achiever_keys.sum() > 0
                and not self.achiever_has_key
            ):
                self.achiever_has_key = True

    def _infer_final_target_from_observed_keys(self, obs):
        """Infer final target door color from current observed key."""
        if self.current_key_index < len(self.observed_keys):
            # Use the key at current_key_index for inference
            key_color = self.observed_keys[self.current_key_index]
            self.final_target_color = key_color

            # Find position of inferred door
            self.final_target_pos = self._find_door_position_from_obs(
                self.final_target_color, obs
            )
