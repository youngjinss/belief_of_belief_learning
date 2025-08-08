import os
import sys
import numpy as np

# Add the lib directory to the path
lib_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "lib")
sys.path.insert(0, lib_path)

# Add the env directory to the path
env_path = os.path.join(lib_path, "env")
sys.path.insert(0, env_path)

try:
    from gym_minigrid.minigrid import Key, Door, Wall
except ImportError:
    # Fallback for different import structure
    import sys
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    lib_path = os.path.join(project_root, "lib", "env")
    sys.path.insert(0, lib_path)
    from gym_minigrid.minigrid import Key, Door, Wall


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

        # Target door color for consumption penalty (preferred key)
        self._preferred_door_color = None

        # Memory system for partial observation
        self.memory = {}  # Stores discovered key/door positions
        self.discovered_positions = set()  # Track all discovered positions
        
        # Exploration state for clockwise wall-following
        self.last_direction = None  # Last attempted direction
        self.exploration_mode = True
        self.stuck_counter = 0  # Counter for stuck detection
        
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
            
        # In partial observation, scan the visible area around agent
        if self.observability == "partial":
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
                            
                            # Store keys in memory
                            if isinstance(cell, Key):
                                key_color = cell.color
                                self.memory[f"key_{key_color}"] = pos
                                
                            # Store doors in memory  
                            elif isinstance(cell, Door):
                                door_color = cell.color
                                self.memory[f"door_{door_color}"] = pos
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

    def _find_preferred_targets(self, obs):
        """
        Find preferred targets considering already collected keys
        
        Returns preferred target positions that the agent should pursue,
        taking into account keys already collected to avoid redundant collection.
        """
        targets = []
        
        # Check if we have key inventory information
        collected_keys = set()
        if self.role == "achiever" and "achiever_keys" in obs:
            achiever_keys = obs["achiever_keys"]
            color_map = ["red", "green", "blue", "yellow"]
            for i, has_key in enumerate(achiever_keys):
                if has_key > 0 and i < len(color_map):
                    collected_keys.add(color_map[i])
        
        # Find target key (avoid already collected keys)
        if self._preferred_door_color:
            # First priority: target key if not already collected
            if self._preferred_door_color not in collected_keys:
                key_pos = self.memory.get(f"key_{self._preferred_door_color}")
                if key_pos:
                    targets.append(key_pos)
            
            # Second priority: target door if we have the key
            if self._preferred_door_color in collected_keys:
                door_pos = self.memory.get(f"door_{self._preferred_door_color}")
                if door_pos:
                    targets.append(door_pos)
        
        return targets

    def _get_clockwise_direction(self, current_direction):
        """
        Get next direction in clockwise rotation: up�right�down�left�up
        
        Includes validation to ensure the new direction is walkable.
        """
        if current_direction is None:
            return 0  # Start with up
            
        # Get next direction clockwise
        next_dir = (current_direction + 1) % 4
        
        # Verify the new direction is walkable
        if self.agent_pos is not None:
            dx, dy = self.direction_deltas[next_dir]
            new_pos = (self.agent_pos[0] + dx, self.agent_pos[1] + dy)
            
            if self._is_walkable(new_pos):
                return next_dir
        
        # If not walkable, continue rotating
        return self._get_clockwise_direction(next_dir)

    def _get_blocked_directions(self):
        """Get list of directions that are blocked (walls, boundaries, obstacles)"""
        blocked = []
        if self.agent_pos is None:
            return blocked
            
        x, y = self.agent_pos
        
        for direction, (dx, dy) in enumerate(self.direction_deltas):
            new_pos = (x + dx, y + dy)
            
            # Check boundaries
            if (new_pos[0] < 0 or new_pos[0] >= (self.width or 9) or 
                new_pos[1] < 0 or new_pos[1] >= (self.height or 9)):
                blocked.append(direction)
                continue
                
            # Check walkability
            if not self._is_walkable(new_pos):
                blocked.append(direction)
                
        return blocked

    def _should_change_direction(self):
        """
        Enhanced direction change detection with improved obstacle detection
        
        Returns True if agent should change direction due to obstacles or being stuck.
        """
        if self.agent_pos is None or self.last_direction is None:
            return True
            
        # Check if current direction is blocked
        dx, dy = self.direction_deltas[self.last_direction]
        next_pos = (self.agent_pos[0] + dx, self.agent_pos[1] + dy)
        
        # Check boundaries and walkability
        if (next_pos[0] < 0 or next_pos[0] >= (self.width or 9) or 
            next_pos[1] < 0 or next_pos[1] >= (self.height or 9) or
            not self._is_walkable(next_pos)):
            return True
            
        return False

    def _explore_with_clockwise_pattern(self):
        """
        Enhanced clockwise exploration with stuck detection and recovery
        
        Uses systematic clockwise wall-following: up�right�down�left�up
        """
        if self._should_change_direction():
            self.last_direction = self._get_clockwise_direction(self.last_direction)
            self.stuck_counter = 0
        else:
            # Continue in current direction but check for being stuck
            self.stuck_counter += 1
            if self.stuck_counter > 3:  # If stuck for too long
                self.last_direction = self._get_clockwise_direction(self.last_direction)
                self.stuck_counter = 0
        
        return self.last_direction if self.last_direction is not None else 0

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

    def _is_walkable(self, pos):
        """Check if position is walkable"""
        width = self.width if self.width is not None else 9
        height = self.height if self.height is not None else 9
        if pos[0] < 0 or pos[0] >= width or pos[1] < 0 or pos[1] >= height:
            return False

        if self.grid is not None:
            cell = self.grid.get(pos[0], pos[1])
            if cell is not None and isinstance(cell, Wall):
                return False
            return True

        return True

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
        self.last_direction = None
        self.exploration_mode = True
        self.stuck_counter = 0