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
        self.gamma = gamma
        self.temperature = temperature
        self.q_value_clip = q_value_clip
        
        # Value function and policy
        self.value_function = None
        self.policy = None
        self.converged = False
        
        # Grid coordinate system: (x, y) where x=column, y=row, positive y is down
        self.actions = [
            (0, -1),  # up
            (1, 0),   # right  
            (0, 1),   # down
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
        Vectorized value iteration to compute optimal action for reaching target
        
        Args:
            target_pos (tuple): Target position (x, y)
            obs (dict): Current observations
            max_iterations (int): Maximum iterations for convergence
            convergence_threshold (float): Threshold for convergence detection
            
        Returns:
            int: Optimal action index (0-3 for movement directions)
        """
        # Get grid size from instance dimensions
        width = self.width if self.width is not None else 9  # fallback
        height = self.height if self.height is not None else 9  # fallback
        n_actions = 4
        
        # Get opponent position from observations (for conflict penalty)
        opponent_pos = self._get_opponent_position(obs)
        
        # Initialize value function
        value_function = np.zeros((width, height))
        
        # Set high reward for target position
        value_function[target_pos[0], target_pos[1]] = 10.0
        
        # Precompute action deltas and grid coordinates
        actions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # up, right, down, left
        x_coords, y_coords = np.meshgrid(
            np.arange(width), np.arange(height), indexing="ij"
        )
        coord_mask = np.ones((width, height), dtype=bool)
        
        # Mask for target position (keep it unchanged)
        coord_mask[target_pos[0], target_pos[1]] = False
        
        # Precompute walkability mask (vectorized)
        walkable_mask = np.ones((width, height), dtype=bool)
        # For the simplified walkability check, most positions are walkable
        # Walls are typically only at borders - this is a simplified approach
        walkable_mask[0, :] = True  # Top border (doors can be here)
        walkable_mask[-1, :] = True  # Bottom border
        walkable_mask[:, 0] = True  # Left border
        walkable_mask[:, -1] = True  # Right border
        # Interior positions are walkable by default (already True)
        
        # Run vectorized value iteration
        for iteration in range(max_iterations):
            old_values = value_function.copy()
            
            # Vectorized Q-value computation for all positions and actions
            q_values_all = np.zeros((width, height, n_actions))
            
            for action_idx, (dx, dy) in enumerate(actions):
                # Compute next positions for all grid cells
                next_x = x_coords + dx
                next_y = y_coords + dy
                
                # Bounds checking
                valid_moves = (
                    (next_x >= 0) & (next_x < width) & (next_y >= 0) & (next_y < height)
                )
                
                # Initialize rewards and next values
                rewards = np.full((width, height), -self.movement_cost)
                next_values = np.zeros((width, height))
                
                # Handle valid moves
                valid_next_x = np.where(valid_moves, next_x, x_coords)
                valid_next_y = np.where(valid_moves, next_y, y_coords)
                
                # Get next values (stay in place for invalid moves)
                next_values = np.where(
                    valid_moves,
                    self.gamma * old_values[valid_next_x, valid_next_y],
                    self.gamma * old_values[x_coords, y_coords],
                )
                
                # Apply penalties for invalid moves
                rewards = np.where(valid_moves, rewards, rewards - self.wall_penalty)
                
                # Apply walkability penalties
                next_walkable = np.where(
                    valid_moves,
                    walkable_mask[valid_next_x, valid_next_y],
                    True,  # Staying in place is always "walkable"
                )
                rewards = np.where(next_walkable, rewards, rewards - self.wall_penalty)
                next_values = np.where(
                    next_walkable,
                    next_values,
                    self.gamma * old_values[x_coords, y_coords],
                )
                
                # Apply opponent position penalty
                if opponent_pos is not None:
                    opponent_conflict = (
                        valid_moves
                        & (valid_next_x == opponent_pos[0])
                        & (valid_next_y == opponent_pos[1])
                    )
                    rewards = np.where(
                        opponent_conflict, rewards - self.conflict_penalty, rewards
                    )
                    next_values = np.where(
                        opponent_conflict,
                        self.gamma * old_values[x_coords, y_coords],
                        next_values,
                    )
                
                # Apply target bonus
                target_bonus = (
                    valid_moves
                    & (valid_next_x == target_pos[0])
                    & (valid_next_y == target_pos[1])
                )
                rewards = np.where(target_bonus, rewards + 10.0, rewards)
                
                # Store Q-values
                q_values_all[:, :, action_idx] = rewards + next_values
            
            # Update value function (max over actions)
            new_values = np.max(q_values_all, axis=2)
            
            # Apply masks: keep target value high, set unwalkable positions to penalty
            value_function = np.where(coord_mask, new_values, value_function)
            value_function = np.where(walkable_mask, value_function, -self.wall_penalty)
            
            # Check convergence
            if np.max(np.abs(value_function - old_values)) < convergence_threshold:
                break
        
        # Get optimal action for current position
        if not self._is_walkable(self.agent_pos):
            return None
            
        current_pos = self.agent_pos
        q_values = []
        for action in range(n_actions):
            q_val = self._evaluate_action(
                current_pos,
                action,
                value_function,
                target_pos,
                width,
                height,
                opponent_pos,
            )
            q_values.append(q_val)
        
        # Choose action with softmax policy
        if self.temperature > 0:
            q_values = np.array(q_values)
            q_values_clipped = np.clip(q_values, -self.q_value_clip, self.q_value_clip)
            
            # Numerically stable softmax
            # Step 1: Scale by temperature
            scaled_q = q_values_clipped / self.temperature
            # Step 2: Subtract max to prevent overflow (stable softmax)
            scaled_q_shifted = scaled_q - np.max(scaled_q)
            # Step 3: Compute exponentials (now safe from overflow)
            exp_q = np.exp(scaled_q_shifted)
            # Step 4: Normalize to get probabilities
            action_probs = exp_q / np.sum(exp_q)
            
            # Check for numerical issues
            if np.any(np.isnan(action_probs)) or np.any(np.isinf(action_probs)) or np.sum(action_probs) == 0:
                # Fallback to uniform random action
                action = np.random.choice(n_actions)
            else:
                # Sample action stochastically
                action = np.random.choice(n_actions, p=action_probs)
        else:
            # Deterministic policy
            action = np.argmax(q_values)
        
        return action
    
    def _evaluate_action(
        self, pos, action, value_function, target_pos, width, height, opponent_pos=None
    ):
        """Evaluate expected value of taking action from position"""
        x, y = pos  # Grid coordinates (x=column, y=row)
        dx, dy = self.actions[action]
        new_pos = (x + dx, y + dy)
        
        # Base movement cost
        reward = -self.movement_cost
        
        # Check bounds using the width/height from value iteration
        if (
            new_pos[0] < 0
            or new_pos[0] >= width
            or new_pos[1] < 0
            or new_pos[1] >= height
        ):
            reward -= self.wall_penalty
            next_value = self.gamma * value_function[x, y]  # Stay in current position
        else:
            # Check if position is walkable
            if not self._is_walkable(new_pos):
                reward -= self.wall_penalty
                next_value = (
                    self.gamma * value_function[x, y]
                )  # Stay in current position
            else:
                # Check if new position conflicts with opponent position
                if opponent_pos is not None and new_pos == opponent_pos:
                    reward -= (
                        self.conflict_penalty
                    )  # Heavy penalty for trying to move to opponent's position
                    next_value = (
                        self.gamma * value_function[x, y]
                    )  # Stay in current position
                else:
                    # Bonus for reaching target
                    if new_pos == target_pos:
                        reward += 10.0
                    
                    next_value = self.gamma * value_function[new_pos[0], new_pos[1]]
        
        return reward + next_value
    
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
    
    def reset(self):
        """Reset agent state for new episode"""
        self.agent_pos = None
        self.grid = None
        self.value_function = None
        self.policy = None
        self.converged = False
        # Note: Don't reset width/height as they don't change between episodes