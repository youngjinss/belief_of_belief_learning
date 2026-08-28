import os

import numpy as np

from beliefrl.env.minigrid import Door, Key, Wall
class BaseValueAgent:
    """
    Base Value-based Agent class for partial observation environments
    
    Enhanced for exp8 with memory management and clockwise exploration strategies.
    Provides shared functionality for Level*Value* agents in partial observation mode.
    
    Key enhancements for partial observation:
    - Memory system to persist discovered key/door positions
    - Clockwise wall-following exploration when targets not found
    - Enhanced fallback behaviors for exploration mode
    - Robust error handling and input validation
    """

    def __init__(
        self,
        observability="partial",
        movement_cost=0.01,
        wall_penalty=2.0,
        conflict_penalty=2.0,
        consumption_penalty=1.0,
        gamma=0.99,
        temperature=0.1,
        q_value_clip=100,
        role="achiever",
        grid_width=None,
        grid_height=None,
    ):
        """
        Initialize base value agent for partial observation
        
        Args:
            observability (str): Observation mode ("full" or "partial")
            movement_cost (float): Cost per movement action
            wall_penalty (float): Penalty for hitting walls or invalid moves
            conflict_penalty (float): Penalty for agent conflicts  
            consumption_penalty (float): Penalty for key consumption actions
            gamma (float): Discount factor for future rewards
            temperature (float): Temperature for softmax action selection
            q_value_clip (float): Range for Q-value clipping
            role (str): Agent role ("achiever" or "blocker")
        """
        self.observability = observability
        self.role = role
        self.agent_pos = None
        self.grid = None

        # Grid dimensions - set from parameters or will be determined from observations
        self.width = grid_width
        self.height = grid_height

        # Value iteration parameters
        self.movement_cost = movement_cost
        self.wall_penalty = wall_penalty
        self.conflict_penalty = conflict_penalty
        self.consumption_penalty = consumption_penalty
        self.gamma = gamma
        self.temperature = temperature
        self.q_value_clip = q_value_clip

        # Target door color for consumption penalty (preferred key)
        self._preferred_door_color = None

        # Memory system for partial observation
        self.memory = {}  # Stores discovered key/door positions
        self.discovered_positions = set()  # Track all discovered positions
        self.wall_positions = set()  # Track discovered wall positions
        
        # Add debug flag to track memory issues
        self._debug_memory = os.getenv("DEBUG_MODE")
        
        # Exploration state for clockwise wall-following
        self.last_direction = None  # Last attempted direction
        self.exploration_mode = True
        
        # Direction mapping for clockwise rotation: up�right�down�left�up
        self.directions = [0, 1, 2, 3]  # up, right, down, left
        self.direction_deltas = [(0, -1), (1, 0), (0, 1), (-1, 0)]

        # Grid coordinate system: (x, y) where x=column, y=row, positive y is down
        self.actions = [
            (0, -1),  # up
            (1, 0),   # right
            (0, 1),   # down
            (-1, 0),  # left
        ]

    def act(self, obs):
        """
        Main action method that handles memory updates and calls get_action
        
        This method ensures proper memory management for partial observation
        and should be used by all Level*Value* agents.
        """
        self.update_observation(obs)
        self._update_memory(obs, self.agent_pos)
        return self.get_action(obs)

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

    def _update_memory(self, obs, agent_pos):
        """
        Update memory with discovered objects in partial observation
        
        Args:
            obs (dict): Current observations
            agent_pos (tuple): Current agent position
        """
        if obs is None or agent_pos is None:
            return
        
        # Debug memory persistence 
        if self._debug_memory and hasattr(self, '_last_memory_size'):
            if len(self.memory) < self._last_memory_size:
                print(f"DEBUG: WARNING - Memory shrank from {self._last_memory_size} to {len(self.memory)}!")
                print(f"DEBUG: Current memory: {list(self.memory.keys())}")
        
        # In partial observation, use both grid scanning AND observation data
        if self.observability == "partial":
            # Method 1: Use observation data for visible objects (more reliable)
            # Handle achiever observations
            if "achiever_visible_keys" in obs:
                for color, pos in obs["achiever_visible_keys"].items():
                    if pos is not None:
                        self.memory[f"key_{color}"] = tuple(pos)
                        
            if "achiever_visible_doors" in obs:
                for color, pos in obs["achiever_visible_doors"].items():
                    if pos is not None:
                        self.memory[f"door_{color}"] = tuple(pos)
                        if os.getenv("DEBUG_MODE"):
                            print(f"DEBUG: Saved door_{color} at {tuple(pos)} to memory")
                            print(f"DEBUG: Memory now has: {list(self.memory.keys())}")
            
            # Handle blocker observations
            if "blocker_visible_keys" in obs:
                for color, pos in obs["blocker_visible_keys"].items():
                    if pos is not None:
                        self.memory[f"key_{color}"] = tuple(pos)
                        
            if "blocker_visible_doors" in obs:
                for color, pos in obs["blocker_visible_doors"].items():
                    if pos is not None:
                        self.memory[f"door_{color}"] = tuple(pos)
                        if os.getenv("DEBUG_MODE"):
                            print(f"DEBUG: Saved door_{color} at {tuple(pos)} to memory")
                            print(f"DEBUG: Memory now has: {list(self.memory.keys())}")
            
            # Method 2: Grid scanning as backup (in case obs data is not available)
            partial_view_size = 5  # Standard 5x5 view
            half_size = partial_view_size // 2
            
            agent_x, agent_y = agent_pos
            
            # Scan the partial view area
            for dx in range(-half_size, half_size + 1):
                for dy in range(-half_size, half_size + 1):
                    scan_x = agent_x + dx
                    scan_y = agent_y + dy
                    
                    # Check bounds
                    if (scan_x >= 0 and scan_x < (self.width or 9) and 
                        scan_y >= 0 and scan_y < (self.height or 9)):
                        
                        pos = (scan_x, scan_y)
                        self.discovered_positions.add(pos)
                        
                        # Check what's at this position
                        if self.grid is not None:
                            cell = self.grid.get(scan_x, scan_y)
                            
                            # Store walls in memory
                            if isinstance(cell, Wall):
                                self.wall_positions.add(pos)
                            
                            # Store keys in memory
                            elif isinstance(cell, Key):
                                key_color = cell.color
                                self.memory[f"key_{key_color}"] = pos
                                
                            # Store doors in memory  
                            elif isinstance(cell, Door):
                                door_color = cell.color
                                self.memory[f"door_{door_color}"] = pos
                                if os.getenv("DEBUG_MODE"):
                                    print(f"DEBUG: Grid scan saved door_{door_color} at {pos} to memory")
        
        # Track memory size for debugging
        if self._debug_memory:
            self._last_memory_size = len(self.memory)
            if self.memory:
                print(f"DEBUG: Memory now contains {len(self.memory)} items: {list(self.memory.keys())}")
        else:
            # In full observation, use observation data directly
            if "key_positions" in obs:
                for color, pos in obs["key_positions"].items():
                    if pos is not None:
                        self.memory[f"key_{color}"] = tuple(pos)
                        
            if "door_positions" in obs:
                for color, pos in obs["door_positions"].items():
                    if pos is not None:
                        self.memory[f"door_{color}"] = tuple(pos)

    def get_action(self, obs):
        """Get action - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement get_action")

    def _get_clockwise_direction(self, current_direction):
        """
        Get next direction based on wall configuration.
        
        Strategy:
        - Check all 4 directions for walkability
        - Prioritize directions opposite to walls
        - If all directions blocked -> raise error (environment bug)
        """
        if self.agent_pos is None:
            return 0  # Default to up if no position
            
        # Check walkability for all directions
        walkable = [self._is_walkable((self.agent_pos[0] + dx, self.agent_pos[1] + dy)) 
                    for dx, dy in self.direction_deltas]
        
        # Count walkable directions
        walkable_count = sum(walkable)
        
        # If no walkable directions, this is an environment bug
        if walkable_count == 0:
            raise RuntimeError(
                f"ERROR: Agent at position {self.agent_pos} has no walkable directions! "
                f"This is a bug in the environment configuration."
            )
        
        # If only one direction is walkable, choose it
        if walkable_count == 1:
            return walkable.index(True)
        
        # Calculate preference scores for each direction
        # Higher score = more preferred
        # Strategy: prefer directions away from walls
        scores = [0] * 4
        
        for i in range(4):
            if walkable[i]:
                # Base score for walkable direction
                scores[i] = 1
                
                # Add bonus for being opposite to blocked directions
                opposite_dir = (i + 2) % 4
                if not walkable[opposite_dir]:
                    scores[i] += 2
                
                # Add small bonus for perpendicular blocked directions
                perpendicular_dirs = [(i + 1) % 4, (i - 1) % 4]
                for perp_dir in perpendicular_dirs:
                    if not walkable[perp_dir]:
                        scores[i] += 1
                
                # Prefer to continue in current direction if not stuck
                if current_direction is not None and i == current_direction:
                    scores[i] += 0.5
        
        # Choose direction with highest score
        # In case of tie, this naturally follows array order (up, right, down, left)
        best_score = max(scores)
        best_directions = [i for i, score in enumerate(scores) if score == best_score]
        
        # If current direction is among best, keep it
        if current_direction in best_directions:
            return current_direction
            
        # Otherwise return first best direction
        return best_directions[0]

    def _should_change_direction(self):
        """
        Enhanced direction change detection with improved obstacle detection
        
        Returns True if agent should change direction due to obstacles or being stuck.
        """
        if self.agent_pos is None or self.last_direction is None:
            if os.getenv("DEBUG_MODE"):
                print(f"DEBUG: _should_change_direction = True (pos={self.agent_pos}, dir={self.last_direction})")
            return True
            
        # Check if current direction is blocked
        dx, dy = self.direction_deltas[self.last_direction]
        next_pos = (self.agent_pos[0] + dx, self.agent_pos[1] + dy)
        
        if os.getenv("DEBUG_MODE"):
            print(f"DEBUG: Checking direction {self.last_direction} from {self.agent_pos} to {next_pos}")
            print(f"DEBUG: Grid bounds: width={self.width}, height={self.height}")
        
        # Check boundaries first
        if next_pos[0] < 0 or next_pos[0] >= self.width or next_pos[1] < 0 or next_pos[1] >= self.height:
            if os.getenv("DEBUG_MODE"):
                print(f"DEBUG: _should_change_direction = True (boundary hit)")
            return True
            
        # Then check walkability
        if os.getenv("DEBUG_MODE") and next_pos == (1, 0):
            print(f"DEBUG: About to call _is_walkable({next_pos})")
        walkable = self._is_walkable(next_pos)
        if os.getenv("DEBUG_MODE") and next_pos == (1, 0):
            print(f"DEBUG: _is_walkable({next_pos}) returned {walkable}")
        if not walkable:
            if os.getenv("DEBUG_MODE"):
                print(f"DEBUG: _should_change_direction = True (not walkable)")
            return True
            
        if os.getenv("DEBUG_MODE"):
            print(f"DEBUG: _should_change_direction = False (path clear)")
        return False

    def _explore_with_clockwise_pattern(self):
        """
        Enhanced exploration with intelligent direction selection
        
        Uses wall configuration to choose optimal exploration direction
        """
        if os.getenv("DEBUG_MODE"):
            print(f"DEBUG: _explore_with_clockwise_pattern at pos {self.agent_pos}")
            print(f"DEBUG: last_direction={self.last_direction}")
        
        should_change = self._should_change_direction()
        if os.getenv("DEBUG_MODE"):
            print(f"DEBUG: _should_change_direction() = {should_change}")
        
        if should_change:
            try:
                old_direction = self.last_direction
                self.last_direction = self._get_clockwise_direction(self.last_direction)
                if os.getenv("DEBUG_MODE"):
                    print(f"DEBUG: Changed direction from {old_direction} to {self.last_direction}")
            except RuntimeError as e:
                # This should never happen in a properly configured environment
                print(f"CRITICAL ERROR: {e}")
                # Try to return a random valid action as last resort
                import random
                for _ in range(10):  # Try up to 10 random actions
                    action = random.randint(0, 3)
                    dx, dy = self.direction_deltas[action]
                    new_pos = (self.agent_pos[0] + dx, self.agent_pos[1] + dy)
                    if self._is_walkable(new_pos):
                        return action
                # If still no valid action, return stay
                return 4
        
        final_action = self.last_direction if self.last_direction is not None else 0
        
        if os.getenv("DEBUG_MODE"):
            print(f"DEBUG: Returning action {final_action}")
        return final_action

    def _plan_value_iteration(
        self, target_pos, obs=None, max_iterations=100, convergence_threshold=0.01
    ):
        """
        Optimized vectorized value iteration for optimal path planning
        """
        # Get grid size and constants
        width = self.width if self.width is not None else 9
        height = self.height if self.height is not None else 9
        n_actions = 4

        # Get opponent position
        opponent_pos = self._get_opponent_position(obs)

        # Precompute grid coordinates
        x_coords, y_coords = np.meshgrid(
            np.arange(width), np.arange(height), indexing="ij"
        )
        actions = np.array([(0, -1), (1, 0), (0, 1), (-1, 0)])

        # Precompute walkability mask
        walkable_mask = self._compute_walkability_mask(width, height)

        # Initialize value function
        value_function = np.zeros((width, height), dtype=np.float32)
        value_function[target_pos[0], target_pos[1]] = 10.0

        # Precompute consumption penalty mask
        consumption_mask = (
            self._compute_consumption_mask(width, height)
            if self.role == "achiever"
            else None
        )

        # Precompute next positions
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

        # Run value iteration
        for iteration in range(max_iterations):
            old_values = value_function.copy()

            # Vectorized Q-value computation
            q_values_all = self._compute_q_values_vectorized(
                value_function,
                next_positions,
                valid_moves,
                walkable_mask,
                consumption_mask,
                opponent_pos,
                target_pos,
                x_coords,
                y_coords,
            )

            # Update value function
            new_values = np.max(q_values_all, axis=2)
            new_values[target_pos[0], target_pos[1]] = 10.0
            value_function = np.where(walkable_mask, new_values, -self.wall_penalty)

            # Check convergence
            if np.max(np.abs(value_function - old_values)) < convergence_threshold:
                break

        # Get optimal action for current position
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
        if self.grid is None or self._preferred_door_color is None:
            return np.zeros((width, height), dtype=np.float32)

        consumption_mask = np.zeros((width, height), dtype=np.float32)
        for x in range(width):
            for y in range(height):
                if self._is_non_preferred_key_at_position((x, y)):
                    consumption_mask[x, y] = self.consumption_penalty
        return consumption_mask

    def _compute_q_values_vectorized(
        self,
        value_function,
        next_positions,
        valid_moves,
        walkable_mask,
        consumption_mask,
        opponent_pos,
        target_pos,
        x_coords,
        y_coords,
    ):
        """Vectorized Q-value computation for all positions and actions"""
        width, height, n_actions = next_positions.shape[:3]
        q_values = np.zeros((width, height, n_actions), dtype=np.float32)

        for action_idx in range(n_actions):
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
                self.gamma * value_function[x_coords, y_coords],
            )

            # Apply penalties vectorized
            rewards = np.where(valid, rewards, rewards - self.wall_penalty)

            # Walkability penalty
            next_walkable = np.where(
                valid, walkable_mask[valid_next_x, valid_next_y], True
            )
            rewards = np.where(next_walkable, rewards, rewards - self.wall_penalty)
            next_values = np.where(
                next_walkable,
                next_values,
                self.gamma * value_function[x_coords, y_coords],
            )

            # Consumption penalty
            if consumption_mask is not None:
                consumption_penalty = np.where(
                    valid, consumption_mask[valid_next_x, valid_next_y], 0
                )
                rewards -= consumption_penalty

            # Opponent conflict penalty
            if opponent_pos is not None:
                opponent_conflict = (
                    valid
                    & (valid_next_x == opponent_pos[0])
                    & (valid_next_y == opponent_pos[1])
                )
                rewards = np.where(
                    opponent_conflict, rewards - self.conflict_penalty, rewards
                )
                next_values = np.where(
                    opponent_conflict,
                    self.gamma * value_function[x_coords, y_coords],
                    next_values,
                )

            # Target bonus
            target_bonus = (
                valid
                & (valid_next_x == target_pos[0])
                & (valid_next_y == target_pos[1])
            )
            rewards = np.where(target_bonus, rewards + 10.0, rewards)

            # Store Q-values
            q_values[:, :, action_idx] = rewards + next_values

        return q_values

    def _select_action_vectorized(self, value_function, target_pos, opponent_pos):
        """Vectorized action selection using precomputed Q-values"""
        current_pos = self.agent_pos
        x, y = current_pos

        # Compute Q-values for current position
        q_values = np.zeros(4, dtype=np.float32)

        for action in range(4):
            dx, dy = self.actions[action]
            new_pos = (x + dx, y + dy)

            # Base reward
            reward = -self.movement_cost

            # Check bounds
            if (
                new_pos[0] < 0
                or new_pos[0] >= self.width
                or new_pos[1] < 0
                or new_pos[1] >= self.height
            ):
                reward -= self.wall_penalty
                next_value = self.gamma * value_function[x, y]
            else:
                # Check walkability
                if not self._is_walkable(new_pos):
                    reward -= self.wall_penalty
                    next_value = self.gamma * value_function[x, y]
                else:
                    # Consumption penalty
                    if (
                        self.role == "achiever"
                        and self._is_non_preferred_key_at_position(new_pos)
                    ):
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

        # Action selection with softmax
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

    def _get_opponent_position(self, obs):
        """Get opponent position - to be overridden by subclasses"""
        return None

    def _navigate_with_value_iteration(self, target_pos, obs=None):
        """Navigate using value iteration"""
        optimal_action = self._plan_value_iteration(target_pos, obs)
        if optimal_action is None:
            return 4  # Stay
        return optimal_action

    def _global_to_local_coords(self, global_pos):
        """Convert global maze coordinates to local partial view coordinates"""
        if self.grid is None or self.agent_pos is None:
            return None
            
        gx, gy = global_pos
        ax, ay = self.agent_pos
        
        # In partial observation, the grid is centered around the agent
        # For a 6x6 grid, the agent is typically at position (2,2) or (3,3) in local coordinates
        # This depends on the specific implementation, but commonly:
        # - Grid size 6: agent at local position (2,2) [0-indexed]
        
        grid_size = self.grid.width  # Should be 6 for partial view
        center = grid_size // 2  # For 6x6, center = 3, but 0-indexed means agent at (2,2)
        
        # Calculate local coordinates
        local_x = gx - ax + center
        local_y = gy - ay + center
        
        # Check if within local grid bounds
        if 0 <= local_x < grid_size and 0 <= local_y < grid_size:
            return (local_x, local_y)
        else:
            return None  # Outside partial view

    def _is_walkable(self, pos):
        """Check if position is walkable - walls are not walkable, but doors are walkable"""
        assert self.width is not None and self.height is not None, \
            f"Grid dimensions not initialized! width={self.width}, height={self.height}"
        
        if os.getenv("DEBUG_MODE") and pos == (1, 0):
            print(f"DEBUG _is_walkable: Checking global position {pos}")
            print(f"DEBUG _is_walkable: Agent at global position {self.agent_pos}")
            if self.grid is not None:
                print(f"DEBUG _is_walkable: Local grid size = {self.grid.width}x{self.grid.height}")
        
        # Check global boundary first
        if pos[0] < 0 or pos[0] >= self.width or pos[1] < 0 or pos[1] >= self.height:
            if os.getenv("DEBUG_MODE") and pos == (1, 0):
                print(f"DEBUG _is_walkable: {pos} is out of global bounds")
            return False

        # Check wall memory first (for positions we've seen before)
        if tuple(pos) in self.wall_positions:
            if os.getenv("DEBUG_MODE") and pos == (1, 0):
                print(f"DEBUG _is_walkable: {pos} found in wall_positions memory")
            return False

        if self.grid is not None:
            # Convert global coordinates to local partial view coordinates
            local_pos = self._global_to_local_coords(pos)
            
            if os.getenv("DEBUG_MODE") and pos == (1, 0):
                print(f"DEBUG _is_walkable: Global {pos} -> Local {local_pos}")
            
            if local_pos is None:
                # Position is outside partial view - treat as unknown/not walkable for safety
                if os.getenv("DEBUG_MODE") and pos == (1, 0):
                    print(f"DEBUG _is_walkable: {pos} outside partial view - treating as unwalkable")
                return False
            
            local_x, local_y = local_pos
            cell = self.grid.get(local_x, local_y)
            
            if os.getenv("DEBUG_MODE") and pos == (1, 0):
                print(f"DEBUG _is_walkable: Local grid cell at {local_pos} = {cell}, type = {type(cell)}")
            
            # Import Wall and Door classes
            from beliefrl.env.minigrid import Wall, Door
            
            # Walls are not walkable
            if cell is not None and isinstance(cell, Wall):
                if os.getenv("DEBUG_MODE") and pos == (1, 0):
                    print(f"DEBUG _is_walkable: {pos} is a Wall - NOT walkable")
                return False
            # Doors are walkable (agents can move to door positions to break them)
            if cell is not None and isinstance(cell, Door):
                if os.getenv("DEBUG_MODE") and pos == (1, 0):
                    print(f"DEBUG _is_walkable: {pos} is a Door - walkable")
                return True
            # Empty spaces and other objects are walkable
            if os.getenv("DEBUG_MODE") and pos == (1, 0):
                print(f"DEBUG _is_walkable: {pos} is empty/other - walkable")
            return True

        if os.getenv("DEBUG_MODE") and pos == (1, 0):
            print(f"DEBUG _is_walkable: {pos} no grid available - treating as unwalkable")
        # If no grid available, be conservative and treat as unwalkable
        return False

    def _is_non_preferred_key_at_position(self, pos):
        """Check if there's a non-preferred key at the given position"""
        if self.grid is None or self._preferred_door_color is None:
            return False

        width = self.width if self.width is not None else 9
        height = self.height if self.height is not None else 9
        if pos[0] < 0 or pos[0] >= width or pos[1] < 0 or pos[1] >= height:
            return False

        cell = self.grid.get(pos[0], pos[1])
        if isinstance(cell, Key):
            return cell.color != self._preferred_door_color

        return False

    def set_target_door_color(self, color):
        """Set the target door color (preferred key)"""
        self._preferred_door_color = color

    def reset(self):
        """Reset agent state for new episode"""
        self.agent_pos = None
        self.grid = None
        self.memory = {}
        self.discovered_positions = set()
        self.wall_positions = set()
        self.last_direction = None
        self.exploration_mode = True
