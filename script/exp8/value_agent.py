import os
import sys
import numpy as np

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
        self.role = kwargs.get("role", "achiever")  # Default to achiever

        # Target door color for consumption penalty (preferred key)
        self._preferred_door_color = kwargs.get("target_door_color", None)

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

        # Memory system for partial observation
        self.memory = {
            'door_positions': {},  # color -> (x, y) position
            'key_positions': {},   # color -> (x, y) position (None if consumed)
            'wall_positions': set(),  # Set of (x, y) wall positions
            'visited_positions': set(),  # Set of (x, y) visited positions
            'walkable_positions': set(),  # Set of (x, y) known walkable cells
            'unwalkable_positions': set(),  # Set of (x, y) known unwalkable cells
        }
        
        # Exploration state
        self.exploration_mode = False
        self.exploration_direction = None  # Current exploration direction
        self.exploration_steps = 0  # Steps taken in current direction

    def act(self, obs):
        """
        Main action selection method with strategy coordination
        
        Strategy:
        1. Updates memory from observation
        2. Finds preferred targets  
        3. Decides whether to explore or use value iteration:
           - If preferred targets found in obs or memory: use value iteration (exploration_mode = False)
           - Else: exploration mode (exploration_mode = True)
           - If no preferred target suddenly disappears: exploration_mode = True
        4. Returns the appropriate action
        """
        # 1. Update memory from observation
        self.update_observation(obs)
        agent_pos = tuple(obs.get('achiever_pos', obs.get('blocker_pos', [0, 0])))
        self._update_memory(obs, agent_pos)
        
        # 2. Find preferred targets
        preferred_targets = self._find_preferred_targets()
        
        # Debug print
        if hasattr(self, '_debug_step_count'):
            self._debug_step_count += 1
        else:
            self._debug_step_count = 1
            
        if self._debug_step_count % 20 == 0:  # Print every 20 steps
            print(f"DEBUG: Step {self._debug_step_count}, Role: {getattr(self, 'role', 'unknown')}")
            print(f"  Memory keys: {self.memory['key_positions']}")
            print(f"  Memory doors: {self.memory['door_positions']}")
            print(f"  Preferred targets: {preferred_targets}")
            print(f"  Exploration mode: {self.exploration_mode}")
        
        # 3. Decide strategy based on preferred targets
        if preferred_targets:
            # Found preferred targets - use value iteration
            self.exploration_mode = False
            target_pos = preferred_targets[0]  # Take first preferred target
            
            # Try value iteration navigation
            action = self._navigate_with_value_iteration(target_pos, obs)
            if action is not None:
                return action
            else:
                # Value iteration failed, fall back to exploration
                self.exploration_mode = True
                return self._explore_action()
        else:
            # No preferred targets - use exploration mode
            self.exploration_mode = True
            return self._explore_action()

    def _find_preferred_targets(self):
        """
        Find preferred targets based on agent type and current strategy
        Returns list of preferred target positions
        """
        preferred_targets = []
        
        # Check for keys and doors in memory based on agent role and strategy
        if self.role == "achiever":
            # Achiever strategy: find keys first, then doors
            if hasattr(self, '_preferred_door_color') and self._preferred_door_color:
                # Look for preferred key first (only if not already collected)
                key_pos = self.memory['key_positions'].get(self._preferred_door_color)
                if (key_pos is not None and 
                    hasattr(self, 'collected_keys') and 
                    self._preferred_door_color not in self.collected_keys):
                    preferred_targets.append(key_pos)
                    
                # Look for preferred door (prioritize if key already collected)
                door_pos = self.memory['door_positions'].get(self._preferred_door_color)
                if door_pos is not None:
                    preferred_targets.append(door_pos)
            else:
                # Look for any keys/doors if no specific preference
                for color, pos in self.memory['key_positions'].items():
                    if pos is not None:
                        preferred_targets.append(pos)
                for color, pos in self.memory['door_positions'].items():
                    if pos is not None:
                        preferred_targets.append(pos)
                        
        elif self.role == "blocker":
            # Blocker strategy: find achiever position
            if hasattr(self, 'opponent_pos') and self.opponent_pos is not None:
                preferred_targets.append(self.opponent_pos)
        
        return preferred_targets

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
        
        # Update memory with current observations
        self._update_memory(obs)

    def _update_agent_position(self, obs):
        """Update agent position - to be overridden by subclasses"""
        raise NotImplementedError("Subclasses must implement _update_agent_position")

    def _update_grid_reference(self, obs):
        """Update grid reference - to be overridden by subclasses"""
        raise NotImplementedError("Subclasses must implement _update_grid_reference")
    
    def _update_memory(self, obs, agent_pos=None):
        """Update memory with information from current observation"""
        if obs is None:
            return
            
        # Extract agent position from obs if not provided
        if agent_pos is None:
            agent_pos = tuple(obs.get('achiever_pos', obs.get('blocker_pos', [0, 0])))
            
        # Update agent position
        if agent_pos is not None:
            self.agent_pos = agent_pos
            self.memory['visited_positions'].add(tuple(agent_pos))
            
        # Parse structured observation data (for achiever)
        if self.role == "achiever":
            if 'achiever_visible_keys' in obs and obs['achiever_visible_keys']:
                for color, pos in obs['achiever_visible_keys'].items():
                    if pos is not None:
                        self.memory['key_positions'][color] = tuple(pos)
                        print(f"DEBUG: Found {color} key at {pos}")
                        
            if 'achiever_visible_doors' in obs and obs['achiever_visible_doors']:
                for color, pos in obs['achiever_visible_doors'].items():
                    if pos is not None:
                        self.memory['door_positions'][color] = tuple(pos)
                        print(f"DEBUG: Found {color} door at {pos}")
        
        # Parse structured observation data (for blocker) 
        elif self.role == "blocker":
            if 'blocker_visible_keys' in obs and obs['blocker_visible_keys']:
                for color, pos in obs['blocker_visible_keys'].items():
                    if pos is not None:
                        self.memory['key_positions'][color] = tuple(pos)
                        
            if 'blocker_visible_doors' in obs and obs['blocker_visible_doors']:
                for color, pos in obs['blocker_visible_doors'].items():
                    if pos is not None:
                        self.memory['door_positions'][color] = tuple(pos)
                        
            # Update opponent position for blocker strategy
            if 'blocker_achiever_pos' in obs and obs['blocker_achiever_pos'] is not None:
                self.opponent_pos = tuple(obs['blocker_achiever_pos'])
            
        # Update door positions if visible (structured data)
        if isinstance(obs, dict) and 'door_positions' in obs:
            door_positions = obs['door_positions']
            if isinstance(door_positions, dict):
                for color, pos in door_positions.items():
                    if pos is not None and len(pos) >= 2:
                        self.memory['door_positions'][color] = tuple(pos[:2])
        
        # Update key positions if visible (structured data)
        if isinstance(obs, dict) and 'key_positions' in obs:
            key_positions = obs['key_positions']
            if isinstance(key_positions, dict):
                for color, pos in key_positions.items():
                    if pos is not None and len(pos) >= 2:
                        self.memory['key_positions'][color] = tuple(pos[:2])
        
        # Update wall positions if visible (structured data)
        if isinstance(obs, dict) and 'wall_positions' in obs:
            wall_positions = obs['wall_positions']
            if isinstance(wall_positions, (list, tuple)):
                for pos in wall_positions:
                    if pos is not None and len(pos) >= 2:
                        self.memory['wall_positions'].add(tuple(pos[:2]))
            
        # Update walkability memory from current grid observation
        self._update_walkability_memory()

    def _parse_partial_observation(self, image, agent_pos):
        """Parse partial observation image to extract keys and doors"""
        if image is None or agent_pos is None:
            return
            
        # Get partial view size from config
        partial_view_size = 5  # Default 5x5 for 9x9 environment
        if hasattr(self, 'config') and self.config:
            partial_view_size = self.config.get_partial_view_size()
        
        # Calculate offset from agent position to world coordinates
        offset_x = agent_pos[0] - partial_view_size // 2
        offset_y = agent_pos[1] - partial_view_size // 2
        
        # Parse each cell in the partial view
        height, width = image.shape[:2] if len(image.shape) >= 2 else (partial_view_size, partial_view_size)
        
        for y in range(min(height, partial_view_size)):
            for x in range(min(width, partial_view_size)):
                # Convert to world coordinates
                world_x = offset_x + x
                world_y = offset_y + y
                
                # Skip if out of bounds
                if world_x < 0 or world_y < 0:
                    continue
                    
                # Parse cell content from image
                cell_info = self._parse_cell(image, x, y)
                if cell_info:
                    obj_type, color = cell_info
                    world_pos = (world_x, world_y)
                    
                    if obj_type == 'key':
                        self.memory['key_positions'][color] = world_pos
                        self.memory['walkable_positions'].add(world_pos)
                        print(f"DEBUG: Found {color} key at {world_pos}")
                    elif obj_type == 'door':
                        self.memory['door_positions'][color] = world_pos
                        self.memory['walkable_positions'].add(world_pos)  # Doors are walkable in MiniGrid
                        print(f"DEBUG: Found {color} door at {world_pos}")
                    elif obj_type == 'wall':
                        self.memory['wall_positions'].add(world_pos)
                        self.memory['unwalkable_positions'].add(world_pos)
                    else:
                        # Empty space
                        self.memory['walkable_positions'].add(world_pos)

    def _parse_cell(self, image, x, y):
        """Parse individual cell from partial observation image"""
        try:
            # MiniGrid uses different encodings - this is a simplified approach
            # In practice, you'd need to understand the exact encoding used
            if len(image.shape) == 3 and image.shape[2] >= 3:
                # RGB image format
                pixel = image[y, x]
                return self._pixel_to_object(pixel)
            elif len(image.shape) == 3:
                # Channel-based format (more likely for MiniGrid)
                cell = image[y, x]
                return self._encoded_cell_to_object(cell)
        except (IndexError, ValueError):
            return None
        return None

    def _pixel_to_object(self, pixel):
        """Convert RGB pixel to object type and color"""
        # This would need to be customized based on MiniGrid's color scheme
        # Returning None for now as we need the actual encoding
        return None

    def _encoded_cell_to_object(self, cell):
        """Convert encoded cell to object type and color"""
        # MiniGrid typically uses integer encoding where:
        # cell[0] = object type, cell[1] = color, cell[2] = state
        if len(cell) >= 2:
            obj_type_id = cell[0] if hasattr(cell, '__getitem__') else cell
            color_id = cell[1] if len(cell) > 1 and hasattr(cell, '__getitem__') else 0
            
            # Map object type IDs to types (these are MiniGrid constants)
            obj_type_map = {
                0: None,      # Empty
                1: 'wall',    # Wall
                2: 'door',    # Door
                5: 'key',     # Key
            }
            
            # Map color IDs to colors
            color_map = {
                0: 'red',
                1: 'green', 
                2: 'blue',
                3: 'purple',
                4: 'yellow',
                5: 'grey'
            }
            
            obj_type = obj_type_map.get(obj_type_id)
            color = color_map.get(color_id, 'unknown')
            
            if obj_type in ['key', 'door']:
                return (obj_type, color)
            elif obj_type == 'wall':
                return ('wall', None)
                
        return None
    
    def _update_walkability_memory(self):
        """Update walkability memory from current grid observation"""
        if self.grid is None or self.agent_pos is None:
            return
            
        # Get current partial view bounds - use config or fallback
        view_size = 5  # Default fallback, could get from config
        if hasattr(self, 'width') and self.width is not None:
            if self.width <= 5:
                view_size = 3
            elif self.width <= 9:
                view_size = 5
            else:
                view_size = 7
                
        agent_x, agent_y = self.agent_pos[:2]
        half_view = view_size // 2
        
        # Update walkability for currently visible cells
        for dx in range(-half_view, half_view + 1):
            for dy in range(-half_view, half_view + 1):
                x, y = agent_x + dx, agent_y + dy
                if (self.width is not None and self.height is not None and 
                    0 <= x < self.width and 0 <= y < self.height):
                    cell = self.grid.get(x, y)
                    if cell is not None:
                        pos_tuple = (x, y)
                        if isinstance(cell, Wall):
                            self.memory['unwalkable_positions'].add(pos_tuple)
                            # Remove from walkable if it was there
                            self.memory['walkable_positions'].discard(pos_tuple)
                        else:  # Non-wall object or empty space
                            self.memory['walkable_positions'].add(pos_tuple)
                            # Remove from unwalkable if it was there  
                            self.memory['unwalkable_positions'].discard(pos_tuple)
    
    def _get_memory_augmented_obs(self, obs):
        """Combine current observation with memory for complete view"""
        if obs is None:
            return None
            
        # Start with current observation
        augmented_obs = obs.copy()
        
        # Merge memory data
        if 'door_positions' not in augmented_obs:
            augmented_obs['door_positions'] = {}
        if 'key_positions' not in augmented_obs:
            augmented_obs['key_positions'] = {}
        if 'wall_positions' not in augmented_obs:
            augmented_obs['wall_positions'] = []
            
        # Add memory data that's not already in observation
        for color, pos in self.memory['door_positions'].items():
            if color not in augmented_obs['door_positions']:
                augmented_obs['door_positions'][color] = pos
                
        for color, pos in self.memory['key_positions'].items():
            if color not in augmented_obs['key_positions']:
                augmented_obs['key_positions'][color] = pos
                
        # Add wall positions from memory
        current_walls = set(tuple(pos) for pos in augmented_obs['wall_positions']) if augmented_obs['wall_positions'] else set()
        all_walls = current_walls.union(self.memory['wall_positions'])
        augmented_obs['wall_positions'] = list(all_walls)
        
        return augmented_obs
    
    def _explore_action(self):
        """Execute exploration behavior - clockwise direction change when hitting walls"""
        if self.exploration_direction is None or self._should_change_direction():
            # Pick new direction using clockwise logic
            self.exploration_direction = self._get_clockwise_direction()
            self.exploration_steps = 0
            
        self.exploration_steps += 1
        # Ensure direction is valid
        if self.exploration_direction not in [0, 1, 2, 3]:
            self.exploration_direction = 0
        return self.exploration_direction
    
    def _get_clockwise_direction(self):
        """Get next clockwise direction when hitting obstacles"""
        if self.agent_pos is None or self.width is None or self.height is None:
            # Default to up if no position info
            return 0
            
        # Check which directions are blocked
        blocked_directions = self._get_blocked_directions()
        
        # If current exploration direction is set, use clockwise logic
        if self.exploration_direction is not None:
            # Implement clockwise direction change based on current direction
            # Current: up(0), right(1), down(2), left(3) 
            # Clockwise: up->right->down->left->up
            
            current_dir = self.exploration_direction
            
            # Try clockwise rotations until we find an unblocked direction
            # Start from next clockwise direction
            clockwise_order = [0, 1, 2, 3]  # up, right, down, left
            start_idx = (current_dir + 1) % 4
            
            # Try each direction in clockwise order starting from next direction
            for i in range(4):
                next_dir = clockwise_order[(start_idx + i) % 4]
                if next_dir not in blocked_directions:
                    return next_dir
            
            # If all directions are blocked (shouldn't happen), return current
            return current_dir
        
        # Initial direction selection - find first unblocked direction in clockwise order
        for direction in [0, 1, 2, 3]:  # up, right, down, left
            if direction not in blocked_directions:
                return direction
        
        # Default fallback - return right if all else fails
        return 1  # right
        
    def _get_blocked_directions(self):
        """Get list of directions that are blocked by walls or boundaries"""
        if self.agent_pos is None or self.width is None or self.height is None:
            return []
            
        blocked = []
        x, y = self.agent_pos
        
        # Check each direction: up(0), right(1), down(2), left(3)
        directions_to_check = [
            (0, (x, y - 1)),  # up
            (1, (x + 1, y)),  # right  
            (2, (x, y + 1)),  # down
            (3, (x - 1, y))   # left
        ]
        
        for direction, next_pos in directions_to_check:
            # Check boundaries
            if (next_pos[0] < 0 or next_pos[0] >= self.width or 
                next_pos[1] < 0 or next_pos[1] >= self.height):
                blocked.append(direction)
                continue
                
            # Check walls from memory
            if tuple(next_pos) in self.memory['wall_positions']:
                blocked.append(direction)
                continue
                
            # Check unwalkable positions
            if tuple(next_pos) in self.memory['unwalkable_positions']:
                blocked.append(direction)
                
        return blocked

    def _should_change_direction(self):
        """Check if agent should change exploration direction"""
        if self.agent_pos is None or self.width is None or self.height is None:
            return False
            
        # Change direction if hit wall or boundary
        next_pos = self._get_next_position(self.agent_pos, self.exploration_direction)
        
        # Check boundaries
        if (next_pos[0] < 0 or next_pos[0] >= self.width or 
            next_pos[1] < 0 or next_pos[1] >= self.height):
            return True
            
        # Check walls
        if tuple(next_pos) in self.memory['wall_positions']:
            return True
            
        # Check unwalkable positions
        if tuple(next_pos) in self.memory['unwalkable_positions']:
            return True
            
        # Change direction after too many steps
        return self.exploration_steps > 5
    
    def _get_next_position(self, pos, action):
        """Get next position given current position and action"""
        if pos is None or len(pos) < 2:
            return None
        if action is None or action < 0 or action >= len(self.actions):
            return pos  # Return current position if invalid action
        dx, dy = self.actions[action]
        return (pos[0] + dx, pos[1] + dy)
    
    def _is_entire_map_observed(self):
        """Check if entire map has been observed (heuristic)"""
        if self.width is None or self.height is None:
            return False
        total_cells = self.width * self.height
        visited_ratio = len(self.memory['visited_positions']) / total_cells
        return visited_ratio > 0.8  # Consider 80% visited as "complete"

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
        # Handle None target position
        if target_pos is None:
            return 4  # Stay action
            
        # Use memory-augmented observations for partial observability
        if self.observability == "partial":
            obs = self._get_memory_augmented_obs(obs)
        # Get grid size and constants
        width = self.width if self.width is not None else 9
        height = self.height if self.height is not None else 9
        n_actions = 4

        # Get opponent position once
        opponent_pos = self._get_opponent_position(obs)

        # Precompute all grid coordinates and action deltas
        x_coords, y_coords = np.meshgrid(
            np.arange(width), np.arange(height), indexing="ij"
        )
        actions = np.array([(0, -1), (1, 0), (0, 1), (-1, 0)])  # up, right, down, left

        # Precompute walkability mask using actual grid data
        walkable_mask = self._compute_walkability_mask(width, height)

        # Initialize value function with target reward
        value_function = np.zeros((width, height), dtype=np.float32)
        value_function[target_pos[0], target_pos[1]] = 10.0

        # Precompute consumption penalty mask for achievers
        consumption_mask = (
            self._compute_consumption_mask(width, height)
            if self.role == "achiever"
            else None
        )

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

            # Keep target value high and penalize unwalkable positions
            new_values[target_pos[0], target_pos[1]] = 10.0
            value_function = np.where(walkable_mask, new_values, -self.wall_penalty)

            # Check convergence with vectorized operations
            if np.max(np.abs(value_function - old_values)) < convergence_threshold:
                break

        # Get optimal action for current position using vectorized evaluation
        if self.agent_pos is None:
            return None

        # Always try to compute action - let the Q-value computation handle obstacles
        return self._select_action_vectorized(value_function, target_pos, opponent_pos)

    def _compute_walkability_mask(self, width, height):
        """Compute walkability mask using current observation + memory"""
        walkable_mask = np.ones((width, height), dtype=bool)
        
        # Apply known unwalkable positions from memory
        for pos in self.memory['unwalkable_positions']:
            x, y = pos
            if 0 <= x < width and 0 <= y < height:
                walkable_mask[x, y] = False
        
        # Apply current observation data if available
        if self.grid is not None:
            for x in range(width):
                for y in range(height):
                    cell = self.grid.get(x, y)
                    if cell is not None and isinstance(cell, Wall):
                        walkable_mask[x, y] = False
                        self.memory['unwalkable_positions'].add((x, y))
                    elif cell is not None:  # Non-wall object
                        self.memory['walkable_positions'].add((x, y))
        
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
                self.gamma * value_function[x_coords, y_coords],
            )

            # Apply penalties vectorized
            # Invalid move penalty
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

            # Consumption penalty for non-preferred keys
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

        # Compute Q-values for current position only
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
        if (
            new_pos[0] < 0
            or new_pos[0] >= width
            or new_pos[1] < 0
            or new_pos[1] >= height
        ):
            return reward - self.wall_penalty + self.gamma * value_function[x, y]

        # Check walkability
        if not self._is_walkable(new_pos):
            return reward - self.wall_penalty + self.gamma * value_function[x, y]

        # Check opponent conflict
        if opponent_pos is not None and new_pos == opponent_pos:
            return reward - self.conflict_penalty + self.gamma * value_function[x, y]

        # Apply consumption penalty for non-preferred keys (achiever only)
        if self.role == "achiever" and self._preferred_door_color is not None:
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
        """Check if position is walkable using current observation + memory"""
        # Check bounds using grid dimensions from observations
        width = self.width if self.width is not None else 9  # fallback
        height = self.height if self.height is not None else 9  # fallback
        if pos[0] < 0 or pos[0] >= width or pos[1] < 0 or pos[1] >= height:
            return False
        
        # Check memory first (most reliable)
        pos_tuple = tuple(pos)
        if pos_tuple in self.memory['unwalkable_positions']:
            return False
        if pos_tuple in self.memory['walkable_positions']:
            return True
        
        # Check current observation if available
        if self.grid is not None:
            cell = self.grid.get(pos[0], pos[1])
            if cell is not None and isinstance(cell, Wall):
                # Update memory and return
                self.memory['unwalkable_positions'].add(pos_tuple)
                return False
            elif cell is not None:  # Some non-wall object or empty space
                self.memory['walkable_positions'].add(pos_tuple)
                return True
        
        # Conservative assumption: unknown positions are walkable
        # (This prevents value iteration from failing due to incomplete information)
        return True

    def _navigate_with_value_iteration(self, target_pos, obs=None):
        """Navigate using value iteration and convert to MiniGrid actions"""
        import os
        
        if os.getenv('DEBUG_MODE') == 'true':
            print(f"DEBUG: Navigate to {target_pos}, agent at {self.agent_pos}")
            print(f"DEBUG: Memory has {len(self.memory['walkable_positions'])} walkable, "
                  f"{len(self.memory['unwalkable_positions'])} unwalkable positions")
        
        # Run value iteration to get optimal action
        optimal_action = self._plan_value_iteration(target_pos, obs)

        if optimal_action is None:
            if os.getenv('DEBUG_MODE') == 'true':
                print(f"DEBUG: Value iteration failed, falling back to exploration")
            return self._explore_action()  # Fallback to exploration instead of staying

        # Convert value iteration action to MiniGrid action
        return self._convert_to_minigrid_action(optimal_action)

    def set_target_door_color(self, color):
        """Set the target door color (preferred key)"""
        self._preferred_door_color = color

    def _is_non_preferred_key_at_position(self, pos):
        """Check if there's a non-preferred key at the given position"""
        if self.grid is None or self._preferred_door_color is None:
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
            return cell.color != self._preferred_door_color

        return False

    def reset(self):
        """Reset agent state for new episode"""
        self.agent_pos = None
        self.grid = None
        self.value_function = None
        self.policy = None
        self.converged = False
        # Note: Don't reset width/height as they don't change between episodes
