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


class BaseValueAgent:
    """
    Base Value-based Agent class containing common value iteration logic

    This class provides the core value iteration functionality that is shared
    across all Level*Value* agents (achievers and blockers).

    Key features:
    - Vectorized value iteration for optimal path planning
    - Stochastic policy with temperature-based action selection
    - Configurable parameters for movement costs, penalties, and exploration
    - Grid navigation utilities
    - Automatic key pickup and door opening support
    """

    def __init__(
        self,
        observability="full",
        movement_cost=0.01,
        wall_penalty=2.0,
        conflict_penalty=2.0,
        consumption_penalty=1.0,
        gamma=0.99,
        temperature=0.1,
        q_value_clip=100,
        **kwargs  # Allow subclasses to pass additional parameters
    ):
        """
        Initialize base value agent

        Args:
            observability (str): Observation mode ("full" or "partial")
            movement_cost (float): Cost per movement action
            wall_penalty (float): Penalty for hitting walls or invalid moves
            conflict_penalty (float): Penalty for agent conflicts
            consumption_penalty (float): Penalty for key consumption actions (pickup/toggle)
            gamma (float): Discount factor for future rewards
            temperature (float): Temperature for softmax action selection (0 = deterministic)
            q_value_clip (float): Range for Q-value clipping [-q_value_clip, q_value_clip]
            **kwargs: Additional parameters for subclasses
        """
        self.observability = observability
        self.agent_pos = None
        self.grid = None

        # Grid dimensions - will be set from observations
        self.width = None
        self.height = None

        # Value iteration parameters
        self.movement_cost = movement_cost
        self.wall_penalty = wall_penalty
        self.conflict_penalty = conflict_penalty
        self.consumption_penalty = consumption_penalty
        self.gamma = gamma
        self.temperature = temperature
        self.q_value_clip = q_value_clip
        
        # Set role type for consumption penalty application
        self.role = kwargs.get('role', 'achiever')  # Default to achiever
        
        # Target door color for consumption penalty (preferred key)
        self.target_door_color = kwargs.get('target_door_color', None)

        # Value function and policy
        self.value_function = None
        self.policy = None
        self.converged = False

        # Grid coordinate system: (x, y) where x=column, y=row, positive y is down
        self.actions = [
            (0, -1),  # up
            (1, 0),  # right
            (0, 1),  # down
            (-1, 0),  # left
        ]  # dx, dy for up, right, down, left

    def update_observation(self, obs):
        """Update agent's understanding of the environment"""
        if obs is None:
            return

        # Update grid dimensions from observations
        if "grid_info" in obs and self.width is None:
            self.width = obs["grid_info"]["width"]
            self.height = obs["grid_info"]["height"]

        # Update agent position (subclasses should override this)
        self._update_agent_position(obs)

        # Update grid reference (subclasses should override this)
        self._update_grid_reference(obs)

    def _update_agent_position(self, obs):
        """Update agent position - to be overridden by subclasses"""
        raise NotImplementedError("Subclasses must implement _update_agent_position")

    def _update_grid_reference(self, obs):
        """Update grid reference - to be overridden by subclasses"""
        raise NotImplementedError("Subclasses must implement _update_grid_reference")

    def _plan_value_iteration(
        self, target_pos, obs=None, max_iterations=100, convergence_threshold=0.01
    ):
        """
        Highly optimized vectorized value iteration to compute optimal action

        Args:
            target_pos (tuple): Target position (x, y)
            obs (dict): Current observations
            max_iterations (int): Maximum iterations for convergence
            convergence_threshold (float): Threshold for convergence detection

        Returns:
            int: Optimal action index (0-3 for movement directions)
        """
        # Get grid size and constants
        width = self.width if self.width is not None else 9
        height = self.height if self.height is not None else 9
        n_actions = 4
        
        # Get opponent position once
        opponent_pos = self._get_opponent_position(obs)
        
        # Precompute all grid coordinates and action deltas
        x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height), indexing="ij")
        actions = np.array([(0, -1), (1, 0), (0, 1), (-1, 0)])  # up, right, down, left
        
        # Precompute walkability mask using actual grid data
        walkable_mask = self._compute_walkability_mask(width, height)
        
        # Initialize value function with target reward
        value_function = np.zeros((width, height), dtype=np.float32)
        value_function[target_pos[0], target_pos[1]] = 10.0
        
        # Precompute consumption penalty mask for achievers
        consumption_mask = self._compute_consumption_mask(width, height) if self.role == 'achiever' else None
        
        # Precompute all next positions for all actions at once
        next_positions = np.zeros((width, height, n_actions, 2), dtype=np.int32)
        valid_moves = np.zeros((width, height, n_actions), dtype=bool)
        
        for action_idx, (dx, dy) in enumerate(actions):
            next_x = x_coords + dx
            next_y = y_coords + dy
            next_positions[:, :, action_idx, 0] = next_x
            next_positions[:, :, action_idx, 1] = next_y
            valid_moves[:, :, action_idx] = (
                (next_x >= 0) & (next_x < width) & (next_y >= 0) & (next_y < height)
            )
        
        # Run optimized value iteration
        for iteration in range(max_iterations):
            old_values = value_function.copy()
            
            # Vectorized Q-value computation for all positions and actions simultaneously
            q_values_all = self._compute_q_values_vectorized(
                value_function, next_positions, valid_moves, walkable_mask, 
                consumption_mask, opponent_pos, target_pos, x_coords, y_coords
            )
            
            # Update value function
            new_values = np.max(q_values_all, axis=2)
            
            # Keep target value high and penalize unwalkable positions
            new_values[target_pos[0], target_pos[1]] = 10.0
            value_function = np.where(walkable_mask, new_values, -self.wall_penalty)
            
            # Check convergence with vectorized operations
            if np.max(np.abs(value_function - old_values)) < convergence_threshold:
                break
        
        # Get optimal action for current position using vectorized evaluation
        if not self._is_walkable(self.agent_pos):
            return None
            
        return self._select_action_vectorized(value_function, target_pos, opponent_pos)
    
    def _compute_walkability_mask(self, width, height):
        """Compute walkability mask using actual grid data"""
        if self.grid is None:
            return np.ones((width, height), dtype=bool)
            
        walkable_mask = np.ones((width, height), dtype=bool)
        for x in range(width):
            for y in range(height):
                if not self._is_walkable((x, y)):
                    walkable_mask[x, y] = False
        return walkable_mask
    
    def _compute_consumption_mask(self, width, height):
        """Compute consumption penalty mask for non-preferred keys"""
        if self.grid is None or self.target_door_color is None:
            return np.zeros((width, height), dtype=np.float32)
            
        consumption_mask = np.zeros((width, height), dtype=np.float32)
        for x in range(width):
            for y in range(height):
                if self._is_non_preferred_key_at_position((x, y)):
                    consumption_mask[x, y] = self.consumption_penalty
        return consumption_mask
    
    def _compute_q_values_vectorized(self, value_function, next_positions, valid_moves, 
                                   walkable_mask, consumption_mask, opponent_pos, target_pos, 
                                   x_coords, y_coords):
        """Vectorized Q-value computation for all positions and actions"""
        width, height, n_actions = next_positions.shape[:3]
        q_values = np.zeros((width, height, n_actions), dtype=np.float32)
        
        for action_idx in range(n_actions):
            # Get next positions for this action
            next_x = next_positions[:, :, action_idx, 0]
            next_y = next_positions[:, :, action_idx, 1]
            valid = valid_moves[:, :, action_idx]
            
            # Base movement cost
            rewards = np.full((width, height), -self.movement_cost, dtype=np.float32)
            
            # Handle valid moves
            valid_next_x = np.where(valid, next_x, x_coords)
            valid_next_y = np.where(valid, next_y, y_coords)
            
            # Get next values
            next_values = np.where(
                valid,
                self.gamma * value_function[valid_next_x, valid_next_y],
                self.gamma * value_function[x_coords, y_coords]
            )
            
            # Apply penalties vectorized
            # Invalid move penalty
            rewards = np.where(valid, rewards, rewards - self.wall_penalty)
            
            # Walkability penalty
            next_walkable = np.where(valid, walkable_mask[valid_next_x, valid_next_y], True)
            rewards = np.where(next_walkable, rewards, rewards - self.wall_penalty)
            next_values = np.where(
                next_walkable, next_values, self.gamma * value_function[x_coords, y_coords]
            )
            
            # Consumption penalty for non-preferred keys
            if consumption_mask is not None:
                consumption_penalty = np.where(
                    valid, consumption_mask[valid_next_x, valid_next_y], 0
                )
                rewards -= consumption_penalty
            
            # Opponent conflict penalty
            if opponent_pos is not None:
                opponent_conflict = (
                    valid & (valid_next_x == opponent_pos[0]) & (valid_next_y == opponent_pos[1])
                )
                rewards = np.where(opponent_conflict, rewards - self.conflict_penalty, rewards)
                next_values = np.where(
                    opponent_conflict, self.gamma * value_function[x_coords, y_coords], next_values
                )
            
            # Target bonus
            target_bonus = (
                valid & (valid_next_x == target_pos[0]) & (valid_next_y == target_pos[1])
            )
            rewards = np.where(target_bonus, rewards + 10.0, rewards)
            
            # Store Q-values
            q_values[:, :, action_idx] = rewards + next_values
        
        return q_values
    
    def _select_action_vectorized(self, value_function, target_pos, opponent_pos):
        """Vectorized action selection using precomputed Q-values"""
        current_pos = self.agent_pos
        x, y = current_pos
        
        # Compute Q-values for current position only
        q_values = np.zeros(4, dtype=np.float32)
        
        for action in range(4):
            dx, dy = self.actions[action]
            new_pos = (x + dx, y + dy)
            
            # Base reward
            reward = -self.movement_cost
            
            # Check bounds
            if new_pos[0] < 0 or new_pos[0] >= self.width or new_pos[1] < 0 or new_pos[1] >= self.height:
                reward -= self.wall_penalty
                next_value = self.gamma * value_function[x, y]
            else:
                # Check walkability
                if not self._is_walkable(new_pos):
                    reward -= self.wall_penalty
                    next_value = self.gamma * value_function[x, y]
                else:
                    # Consumption penalty
                    if self.role == 'achiever' and self._is_non_preferred_key_at_position(new_pos):
                        reward -= self.consumption_penalty
                    
                    # Opponent conflict
                    if opponent_pos is not None and new_pos == opponent_pos:
                        reward -= self.conflict_penalty
                        next_value = self.gamma * value_function[x, y]
                    else:
                        # Target bonus
                        if new_pos == target_pos:
                            reward += 10.0
                        next_value = self.gamma * value_function[new_pos[0], new_pos[1]]
            
            q_values[action] = reward + next_value
        
        # Action selection with optimized softmax
        if self.temperature > 0:
            q_values_clipped = np.clip(q_values, -self.q_value_clip, self.q_value_clip)
            scaled_q = q_values_clipped / self.temperature
            scaled_q_shifted = scaled_q - np.max(scaled_q)
            exp_q = np.exp(scaled_q_shifted)
            action_probs = exp_q / np.sum(exp_q)
            
            # Check for numerical issues
            if np.any(np.isnan(action_probs)) or np.sum(action_probs) == 0:
                return np.random.choice(4)
            return np.random.choice(4, p=action_probs)
        else:
            return np.argmax(q_values)

    def _evaluate_action(
        self, pos, action, value_function, target_pos, width, height, opponent_pos=None
    ):
        """Optimized action evaluation with early returns"""
        x, y = pos
        dx, dy = self.actions[action]
        new_pos = (x + dx, y + dy)

        # Base movement cost
        reward = -self.movement_cost
        
        # Check bounds first (most common case)
        if new_pos[0] < 0 or new_pos[0] >= width or new_pos[1] < 0 or new_pos[1] >= height:
            return reward - self.wall_penalty + self.gamma * value_function[x, y]
        
        # Check walkability
        if not self._is_walkable(new_pos):
            return reward - self.wall_penalty + self.gamma * value_function[x, y]
        
        # Check opponent conflict
        if opponent_pos is not None and new_pos == opponent_pos:
            return reward - self.conflict_penalty + self.gamma * value_function[x, y]
        
        # Apply consumption penalty for non-preferred keys (achiever only)
        if self.role == 'achiever' and self.target_door_color is not None:
            if self._is_non_preferred_key_at_position(new_pos):
                reward -= self.consumption_penalty
        
        # Target bonus
        if new_pos == target_pos:
            reward += 10.0
        
        return reward + self.gamma * value_function[new_pos[0], new_pos[1]]

    def _get_opponent_position(self, obs):
        """Get opponent position from observations - to be overridden by subclasses"""
        # Default implementation - no opponent position
        return None

    def _convert_to_minigrid_action(self, value_action):
        """Convert value iteration action to MiniGrid direct movement action"""
        if value_action is None:
            return 4  # Stay

        # Value action directly maps to MiniGrid action for direct movement
        # value_action: 0=up, 1=right, 2=down, 3=left
        # MiniGrid actions: 0=up, 1=right, 2=down, 3=left, 4=stay
        return value_action

    def _is_walkable(self, pos):
        """Check if position is walkable by parsing the visual grid"""
        # Check bounds using grid dimensions from observations
        width = self.width if self.width is not None else 9  # fallback
        height = self.height if self.height is not None else 9  # fallback
        if pos[0] < 0 or pos[0] >= width or pos[1] < 0 or pos[1] >= height:
            return False

        # Parse the visual grid to check if position is walkable
        if self.grid is not None:
            # Grid coordinates: pos[0] = x (column), pos[1] = y (row)
            cell = self.grid.get(pos[0], pos[1])

            # Walls are not walkable
            if cell is not None and isinstance(cell, Wall):
                return False

            # Empty space (None), keys, and doors are walkable
            # Keys will be picked up automatically when stepped on
            # Doors will be opened automatically when stepped on (if agent has key)
            return True

        # Fallback: assume walkable if we can't parse the grid
        return True

    def _navigate_with_value_iteration(self, target_pos, obs=None):
        """Navigate using value iteration and convert to MiniGrid actions"""
        # Run value iteration to get optimal action
        optimal_action = self._plan_value_iteration(target_pos, obs)

        if optimal_action is None:
            return 4  # Stay if no action found

        # Convert value iteration action to MiniGrid action
        return self._convert_to_minigrid_action(optimal_action)
    
    def set_target_door_color(self, color):
        """Set the target door color (preferred key)"""
        self.target_door_color = color
        
    def _is_non_preferred_key_at_position(self, pos):
        """Check if there's a non-preferred key at the given position"""
        if self.grid is None or self.target_door_color is None:
            return False
            
        # Check if position is within bounds
        width = self.width if self.width is not None else 9
        height = self.height if self.height is not None else 9
        if pos[0] < 0 or pos[0] >= width or pos[1] < 0 or pos[1] >= height:
            return False
            
        # Check if there's a key at this position
        from gym_minigrid.minigrid import Key
        cell = self.grid.get(pos[0], pos[1])
        if isinstance(cell, Key):
            # Return True if it's NOT the preferred key color
            return cell.color != self.target_door_color
            
        return False

    def reset(self):
        """Reset agent state for new episode"""
        self.agent_pos = None
        self.grid = None
        self.value_function = None
        self.policy = None
        self.converged = False
        # Note: Don't reset width/height as they don't change between episodes
