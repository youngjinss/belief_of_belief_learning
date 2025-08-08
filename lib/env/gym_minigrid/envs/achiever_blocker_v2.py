from ..minigrid import *
from ..register import register
from .base_v2 import BaseEnvV2
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class AchieverBlockerEnvV2(BaseEnvV2):
    """
    2-agent environment with achiever and blocker agents.
    Version 2: Adds support for partial observation mode.
    
    Achiever: Pursues their goal by seeking a color key and door based on preference
    Blocker: Tries to block the door by inferring achiever's preference and blocking access
    
    Both agents can have either full or partial observability.
    """

    def __init__(self, size=9, max_keys=4, preference=None, cost=None, max_steps=None, 
                 observability="full", partial_view_size=7):
        self.size = size
        self.max_keys = max_keys

        # Default preference and cost if not provided
        if preference is None:
            preference = {"red": 1.0, "green": 0.8, "blue": 0.6, "yellow": 0.4}
        if cost is None:
            cost = {"red": 0.1, "green": 0.2, "blue": 0.3, "yellow": 0.4}

        self.preference = preference
        self.cost = cost

        # 4 door colors
        self.door_colors = ["red", "green", "blue", "yellow"]

        # Track agents' states
        self.achiever_keys = []
        self.achiever_pos = None
        self.blocker_pos = None
        self.achiever_dir = 0  # Direction facing (0=right, 1=down, 2=left, 3=up)
        self.blocker_dir = 0
        
        # Previous positions for collision handling
        self.achiever_prev_pos = None
        self.blocker_prev_pos = None
        
        self.target_door_color = None
        
        # Storage for objects by color to avoid for-loops
        self.doors_by_color = {}  # color -> door_obj
        self.keys_by_color = {}   # color -> key_obj (None if consumed)
        self.door_positions = {}  # color -> (x, y) position
        self.key_positions = {}   # color -> (x, y) position (None if consumed)
        self.wall_positions = []  # Cached wall positions
        self.position_to_door_color = {}  # (x, y) -> door_color for fast lookup
        
        # Track state changes
        self.last_door_opened = None  # color of last door opened
        self.last_key_consumed = None  # color of last key consumed

        # Allow custom max_steps
        if max_steps is None:
            max_steps = 4 * size**2

        # Call base class with observation parameters
        super().__init__(
            grid_size=size,
            max_steps=max_steps,
            observability=observability,
            partial_view_size=partial_view_size
        )

        # Separate action spaces for achiever and blocker
        # Achiever: up=0, right=1, down=2, left=3, stay=4, pickup=5, toggle=6
        self.achiever_action_space = spaces.Discrete(7)
        
        # Blocker: up=0, right=1, down=2, left=3, stay=4, broken=5
        self.blocker_action_space = spaces.Discrete(6)
        
        # Combined action space: tuple of actions for (achiever, blocker)
        self.action_space = spaces.Tuple((
            self.achiever_action_space,  # achiever actions (7)
            self.blocker_action_space    # blocker actions (6)
        ))
        

    def reset(self, **kwargs):
        """Reset environment for both agents"""
        # Call parent reset but don't use its agent placement
        obs = super().reset(**kwargs)
        
        # Place both agents randomly (ensuring they don't overlap)
        self.place_agent_pair()
        
        # Generate observations for both agents
        obs_dict = self._get_observations()
        
        info = {
            'target_door_color': self.target_door_color,
            'achiever_keys': self.achiever_keys.copy()
        }
        return obs_dict, info

    def place_agent_pair(self):
        """Place both agents randomly ensuring they don't overlap"""
        # Place achiever first
        self.place_agent()
        self.achiever_pos = self.agent_pos.copy()
        self.achiever_dir = self.agent_dir
        
        # Place blocker (avoid achiever position)
        attempts = 0
        while attempts < 100:  # Prevent infinite loop
            blocker_pos = self.place_agent()
            if not np.array_equal(blocker_pos, self.achiever_pos):
                self.blocker_pos = blocker_pos
                self.blocker_dir = self.agent_dir
                break
            attempts += 1
        
        if attempts >= 100:
            # Fallback: place blocker at a fixed offset from achiever
            offset_x = 1 if self.achiever_pos[0] < self.size - 2 else -1
            self.blocker_pos = np.array([self.achiever_pos[0] + offset_x, self.achiever_pos[1]])
            self.blocker_dir = 0
            
        # Store previous positions
        self.achiever_prev_pos = self.achiever_pos.copy()
        self.blocker_prev_pos = self.blocker_pos.copy()

    def _gen_grid(self, width, height):
        """Generate grid same as KeyDoorEnv but for 2 agents"""
        # Create empty grid
        self.grid = Grid(width, height)

        # Generate surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Clear tracking
        self.achiever_keys = []
        self.last_door_opened = None
        self.last_key_consumed = None
        self.doors_by_color = {}
        self.keys_by_color = {}
        self.door_positions = {}
        self.key_positions = {}
        self.wall_positions = []
        self.position_to_door_color = {}

        # Generate 4 keys and 4 doors with matching colors
        key_positions = []
        door_positions = []

        # Generate random positions for keys
        for color in self.door_colors:
            # Place key
            key = Key(color)
            # Make keys overlappable so achiever can step on them
            key.can_overlap = lambda: True
            key_pos = self.place_obj(
                key, reject_fn=lambda env, pos: tuple(pos) in key_positions
            )
            key_positions.append(tuple(key_pos))
            
            # Store key info
            self.keys_by_color[color] = key
            self.key_positions[color] = tuple(key_pos)

            # Place door on walls
            door_pos = self._place_door_on_wall(color, door_positions)
            door_positions.append(door_pos)
            
            # Store door info
            door = self.grid.get(*door_pos)
            self.doors_by_color[color] = door
            self.door_positions[color] = door_pos
            self.position_to_door_color[tuple(door_pos)] = color  # Add reverse lookup with tuple conversion
            
            # Make doors overlappable for achiever if they have the key
            if isinstance(door, Door):
                # Create a closure that captures the door's color
                def make_door_can_overlap(door_color):
                    def door_can_overlap():
                        return not door.is_locked or door_color in self.achiever_keys
                    return door_can_overlap
                door.can_overlap = make_door_can_overlap(color)

        # Place agent randomly (will be replaced in reset by place_agent_pair)
        self.place_agent()

        # Set target door based on highest preference
        self.target_door_color = max(self.preference, key=self.preference.get)
        
        # Cache wall positions after grid generation
        self._cache_wall_positions()

        # Generate mission
        self.mission = f"achiever: collect {self.target_door_color} key and open {self.target_door_color} door"

    def _place_door_on_wall(self, color, existing_doors):
        """Place door at the center of a wall (same as V1)"""
        width, height = self.grid.width, self.grid.height

        # Calculate center positions for each wall
        center_positions = [
            (width // 2, 0),           # Top wall center
            (width // 2, height - 1),  # Bottom wall center
            (0, height // 2),          # Left wall center
            (width - 1, height // 2),  # Right wall center
        ]

        # Filter out existing positions
        available_positions = [
            pos for pos in center_positions if pos not in existing_doors
        ]

        # Choose random position from available centers
        door_pos = self._rand_elem(available_positions)

        # Place door
        self.grid.set(*door_pos, Door(color, is_locked=True))

        return door_pos
    
    def _cache_wall_positions(self):
        """Cache wall positions during grid generation to avoid repeated scanning"""
        self.wall_positions = []
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                obj = self.grid.get(x, y)
                if obj is not None and obj.type == "wall":
                    self.wall_positions.append((x, y))

    def step(self, actions):
        """Execute actions for both agents simultaneously"""
        achiever_action, blocker_action = actions
        
        # Reset state change tracking
        self.last_door_opened = None
        self.last_key_consumed = None
        
        # Store previous positions
        self.achiever_prev_pos = self.achiever_pos.copy()
        self.blocker_prev_pos = self.blocker_pos.copy()
        
        # Calculate new positions
        achiever_new_pos, achiever_new_dir = self._get_new_position(
            self.achiever_pos, self.achiever_dir, achiever_action, "achiever"
        )
        blocker_new_pos, blocker_new_dir = self._get_new_position(
            self.blocker_pos, self.blocker_dir, blocker_action, "blocker"
        )
        
        # Handle collision (agents cannot overlap)
        achiever_new_pos, blocker_new_pos = self._handle_collision(
            achiever_new_pos, blocker_new_pos
        )
        
        # Update positions
        self.achiever_pos = achiever_new_pos
        self.achiever_dir = achiever_new_dir
        self.blocker_pos = blocker_new_pos
        self.blocker_dir = blocker_new_dir
        
        # Handle achiever picking up keys (auto pickup when stepping on them)
        achiever_reward = self._auto_pickup_key(self.achiever_pos)
        
        # Handle achiever opening doors (auto open when stepping on them with key)
        door_reward = self._auto_open_door(self.achiever_pos)
        achiever_reward += door_reward
        
        # Handle blocker breaking doors (action 5)
        blocker_broke_door = False
        blocker_reward = 0
        if blocker_action == 5:
            # Check if blocker is at a door position using fast lookup
            blocker_pos_tuple = tuple(self.blocker_pos)
            door_color = self.position_to_door_color.get(blocker_pos_tuple)
            if door_color is not None:
                # Blocker successfully breaks the door
                blocker_broke_door = True
                blocker_reward = 1.0  # Large reward for breaking door
        else:
            # Regular blocker reward (positive for blocking achiever from target door)
            if np.array_equal(self.blocker_pos, self.door_positions.get(self.target_door_color, [-1, -1])):
                blocker_reward = 0.1  # Small reward for blocking target door
        
        # Check termination - end when target door is opened OR blocker breaks any door
        terminated = (door_reward > 0 and self.last_door_opened == self.target_door_color) or blocker_broke_door
        truncated = False
        
        # Update step count
        self.step_count += 1
        if self.step_count >= self.max_steps:
            truncated = True
        
        
        # Generate observations
        obs = self._get_observations()
        
        # Combined rewards
        rewards = {
            'achiever': achiever_reward,
            'blocker': blocker_reward
        }
        
        info = {
            'achiever_keys': self.achiever_keys.copy(),
            'target_door_color': self.target_door_color,
            'step_count': self.step_count
        }
        
        return obs, rewards, terminated, truncated, info

    def _get_new_position(self, current_pos, current_dir, action, agent_type="achiever"):
        """Calculate new position based on action (direct movement)"""
        new_pos = current_pos.copy()
        new_dir = current_dir  # Direction doesn't change in this environment
        
        # Define movement vectors for direct movement (same as original MiniGridEnv)
        move_vectors = {
            MiniGridEnv.Actions.up: np.array([0, -1]),
            MiniGridEnv.Actions.right: np.array([1, 0]),
            MiniGridEnv.Actions.down: np.array([0, 1]),
            MiniGridEnv.Actions.left: np.array([-1, 0]),
        }
        
        # Handle movement actions
        if action in move_vectors:
            new_pos = current_pos + move_vectors[action]
        elif action == MiniGridEnv.Actions.stay:
            pass  # No movement
        elif action == 5:  # "broken" action
            pass  # No movement
        # pickup and toggle actions don't change position
        
        # Check bounds
        if (new_pos[0] < 0 or new_pos[0] >= self.grid.width or 
            new_pos[1] < 0 or new_pos[1] >= self.grid.height):
            new_pos = current_pos  # Stay in place if out of bounds
            
        # Check walls and objects
        cell = self.grid.get(*new_pos)
        if cell is not None:
            # Special handling for doors based on agent type
            if isinstance(cell, Door):
                if agent_type == "blocker":
                    # Blockers can always walk on doors to block them
                    pass  # Allow movement
                elif agent_type == "achiever":
                    # Achievers follow original door overlap logic
                    if not cell.can_overlap():
                        new_pos = current_pos  # Stay in place if cannot overlap
                else:
                    # Default behavior for other objects
                    if not cell.can_overlap():
                        new_pos = current_pos  # Stay in place if cannot overlap
            else:
                # Non-door objects use standard overlap check
                if not cell.can_overlap():
                    new_pos = current_pos  # Stay in place if cannot overlap
            
        return new_pos, new_dir

    def _handle_collision(self, achiever_new_pos, blocker_new_pos):
        """Handle collision between agents (agents cannot overlap or swap positions)"""
        if np.array_equal(achiever_new_pos, blocker_new_pos):
            return self.achiever_prev_pos, blocker_new_pos
        elif (np.array_equal(achiever_new_pos, self.blocker_prev_pos) and 
              np.array_equal(blocker_new_pos, self.achiever_prev_pos)):
            return self.achiever_prev_pos, self.blocker_prev_pos
        else:
            return achiever_new_pos, blocker_new_pos

    def _auto_pickup_key(self, agent_pos):
        """Auto pickup key for achiever when stepping on it (same coordinate only)"""
        # Check exact position only
        obj = self.grid.get(*agent_pos)
        if isinstance(obj, Key):
            if len(self.achiever_keys) < self.max_keys:
                key_color = obj.color
                self.achiever_keys.append(key_color)
                self.grid.set(*agent_pos, None)
                
                # Track state change
                self.last_key_consumed = key_color
                self.keys_by_color[key_color] = None  # Mark as consumed
                self.key_positions[key_color] = None  # Mark position as consumed
                
                if key_color == self.target_door_color:
                    return 0.5
                else:
                    return -self.cost[key_color]

        return 0

    def _auto_open_door(self, agent_pos):
        """Auto open door for achiever when stepping on it and having the key"""
        obj = self.grid.get(*agent_pos)

        if isinstance(obj, Door) and obj.is_locked:
            door_color = obj.color
            
            # Check if achiever has the key for this door
            if door_color in self.achiever_keys:
                # Open the door automatically
                obj.is_open = True
                obj.is_locked = False
                
                # Track state change
                self.last_door_opened = door_color
                
                # Give reward based on preference
                return self.preference[door_color]

        return 0

    def _get_door_positions(self):
        """Get all door positions"""
        return list(self.door_positions.values())
    
    def _get_door_positions_with_colors(self):
        """Get door positions with their colors"""
        return self.door_positions.copy()
    
    def _get_key_positions(self):
        """Get key positions with their colors"""
        # Return only non-consumed keys
        return {color: pos for color, pos in self.key_positions.items() if pos is not None}

    def _get_target_door_position(self):
        """Get target door position"""
        return self.door_positions.get(self.target_door_color, None)
    
    def _get_wall_positions(self):
        """Get all wall positions (optimized with caching)"""
        return self.wall_positions

    
    def _get_full_observations(self):
        """Generate full observations for both agents (original implementation)"""
        # Set agent position for observation generation
        temp_pos = self.agent_pos
        temp_dir = self.agent_dir
        
        # Get achiever observation
        self.agent_pos = self.achiever_pos
        self.agent_dir = self.achiever_dir
        achiever_obs = super().gen_obs()
        
        # Get blocker observation  
        self.agent_pos = self.blocker_pos
        self.agent_dir = self.blocker_dir
        blocker_obs = super().gen_obs()
        
        # Restore original agent position
        self.agent_pos = temp_pos
        self.agent_dir = temp_dir
        
        # Create achiever keys array
        achiever_keys_array = np.zeros(len(self.door_colors), dtype=np.int32)
        for i, color in enumerate(self.door_colors):
            if color in self.achiever_keys:
                achiever_keys_array[i] = 1
        
        # Get key positions
        key_positions = self._get_key_positions()
        
        # Get door positions  
        door_positions = self._get_door_positions_with_colors()
        
        # Get wall positions
        wall_positions = self._get_wall_positions()
        
        # Get grid size info
        grid_info = {
            'width': self.width,
            'height': self.height
        }
        
        return {
            'achiever': achiever_obs,
            'blocker': blocker_obs,
            'achiever_keys': achiever_keys_array,
            'achiever_pos': self.achiever_pos.astype(np.int32),
            'blocker_pos': self.blocker_pos.astype(np.int32),
            'target_door_color': self.target_door_color,
            'key_positions': key_positions,
            'door_positions': door_positions,
            'wall_positions': wall_positions,
            'grid_info': grid_info,
            'observability': self.observability
        }
    
    def _get_partial_observations(self):
        """Generate partial observations for both agents"""
        # Store original agent state
        temp_pos = self.agent_pos
        temp_dir = self.agent_dir
        
        # Get achiever's partial observation
        self.agent_pos = self.achiever_pos
        self.agent_dir = self.achiever_dir
        achiever_obs = super().gen_obs()
        achiever_grid, achiever_vis_mask = self.gen_obs_grid()
        
        # Get visible keys and doors for achiever
        achiever_visible_keys = {}
        achiever_visible_doors = {}
        
        for color, pos in self.key_positions.items():
            if pos is not None:
                rel_coords = self.relative_coords(*pos)
                if rel_coords is not None:
                    vx, vy = rel_coords
                    if achiever_vis_mask[vx, vy]:
                        achiever_visible_keys[color] = pos
        
        for color, pos in self.door_positions.items():
            rel_coords = self.relative_coords(*pos)
            if rel_coords is not None:
                vx, vy = rel_coords
                if achiever_vis_mask[vx, vy]:
                    achiever_visible_doors[color] = pos
        
        # Check if blocker is visible to achiever
        achiever_sees_blocker = False
        blocker_rel_pos = None
        blocker_rel_coords = self.relative_coords(*self.blocker_pos)
        if blocker_rel_coords is not None:
            vx, vy = blocker_rel_coords
            if achiever_vis_mask[vx, vy]:
                achiever_sees_blocker = True
                blocker_rel_pos = self.blocker_pos.astype(np.int32)
        
        # Get blocker's partial observation
        self.agent_pos = self.blocker_pos
        self.agent_dir = self.blocker_dir
        blocker_obs = super().gen_obs()
        blocker_grid, blocker_vis_mask = self.gen_obs_grid()
        
        # Get visible keys and doors for blocker
        blocker_visible_keys = {}
        blocker_visible_doors = {}
        
        for color, pos in self.key_positions.items():
            if pos is not None:
                rel_coords = self.relative_coords(*pos)
                if rel_coords is not None:
                    vx, vy = rel_coords
                    if blocker_vis_mask[vx, vy]:
                        blocker_visible_keys[color] = pos
        
        for color, pos in self.door_positions.items():
            rel_coords = self.relative_coords(*pos)
            if rel_coords is not None:
                vx, vy = rel_coords
                if blocker_vis_mask[vx, vy]:
                    blocker_visible_doors[color] = pos
        
        # Check if achiever is visible to blocker
        blocker_sees_achiever = False
        achiever_rel_pos = None
        achiever_rel_coords = self.relative_coords(*self.achiever_pos)
        if achiever_rel_coords is not None:
            vx, vy = achiever_rel_coords
            if blocker_vis_mask[vx, vy]:
                blocker_sees_achiever = True
                achiever_rel_pos = self.achiever_pos.astype(np.int32)
        
        # Restore original agent position
        self.agent_pos = temp_pos
        self.agent_dir = temp_dir
        
        # Create achiever keys array
        achiever_keys_array = np.zeros(len(self.door_colors), dtype=np.int32)
        for i, color in enumerate(self.door_colors):
            if color in self.achiever_keys:
                achiever_keys_array[i] = 1
        
        # Get grid size info
        grid_info = {
            'width': self.width,
            'height': self.height
        }
        
        return {
            'achiever': achiever_obs,
            'blocker': blocker_obs,
            'achiever_keys': achiever_keys_array,
            'achiever_pos': self.achiever_pos.astype(np.int32),
            'blocker_pos': self.blocker_pos.astype(np.int32),
            'target_door_color': self.target_door_color,
            'achiever_visible_keys': achiever_visible_keys,
            'achiever_visible_doors': achiever_visible_doors,
            'achiever_sees_blocker': achiever_sees_blocker,
            'achiever_blocker_pos': blocker_rel_pos,  # Blocker position if visible to achiever
            'blocker_visible_keys': blocker_visible_keys,
            'blocker_visible_doors': blocker_visible_doors,
            'blocker_sees_achiever': blocker_sees_achiever,
            'blocker_achiever_pos': achiever_rel_pos,  # Achiever position if visible to blocker
            'grid_info': grid_info,
            'observability': self.observability
        }

    def render(self, mode="human"):
        """Render environment with both agents"""
        if mode == "human" and self.grid_render is None:
            from gym_minigrid.rendering import Renderer
            from gym_minigrid.minigrid import CELL_PIXELS
            self.grid_render = Renderer(
                self.width * CELL_PIXELS,
                self.height * CELL_PIXELS,
                True if mode == "human" else False,
            )
        
        if mode in ["human", "rgb_array"]:
            from gym_minigrid.minigrid import CELL_PIXELS
            
            # Initialize renderer if needed
            if self.grid_render is None:
                from gym_minigrid.rendering import Renderer
                self.grid_render = Renderer(
                    self.width * CELL_PIXELS,
                    self.height * CELL_PIXELS,
                    False,  # Not human mode for rgb_array
                )
            
            r = self.grid_render
            if r.window:
                r.window.setText(self.mission)
            
            r.beginFrame()
            
            # Render the whole grid
            self.grid.render(r, CELL_PIXELS)
            
            # Draw the achiever agent (red triangle)
            if hasattr(self, 'achiever_pos') and self.achiever_pos is not None:
                r.push()
                r.translate(
                    CELL_PIXELS * (self.achiever_pos[0] + 0.5),
                    CELL_PIXELS * (self.achiever_pos[1] + 0.5),
                )
                r.rotate(getattr(self, 'achiever_dir', 0) * 90)
                r.setLineColor(255, 0, 0)  # Red
                r.setColor(255, 0, 0)      # Red
                r.drawPolygon([(-12, 10), (12, 0), (-12, -10)])
                r.pop()
            
            # Draw the blocker agent (blue triangle)
            if hasattr(self, 'blocker_pos') and self.blocker_pos is not None:
                r.push()
                r.translate(
                    CELL_PIXELS * (self.blocker_pos[0] + 0.5),
                    CELL_PIXELS * (self.blocker_pos[1] + 0.5),
                )
                r.rotate(getattr(self, 'blocker_dir', 0) * 90)
                r.setLineColor(0, 0, 255)  # Blue
                r.setColor(0, 0, 255)      # Blue
                r.drawPolygon([(-12, 10), (12, 0), (-12, -10)])
                r.pop()
            
            r.endFrame()
            
            if mode == "rgb_array":
                return r.getArray()
            elif mode == "human":
                return r
        else:
            # Fallback to parent render for other modes
            return super().render(mode)


# Size variants
class AchieverBlocker5x5EnvV2(AchieverBlockerEnvV2):
    def __init__(self, max_keys=4, preference=None, cost=None, max_steps=None, 
                 observability="full", partial_view_size=5):
        super().__init__(size=5, max_keys=max_keys, preference=preference, cost=cost, 
                         max_steps=max_steps, observability=observability, 
                         partial_view_size=partial_view_size)


class AchieverBlocker9x9EnvV2(AchieverBlockerEnvV2):
    def __init__(self, max_keys=4, preference=None, cost=None, max_steps=None,
                 observability="full", partial_view_size=7):
        super().__init__(size=9, max_keys=max_keys, preference=preference, cost=cost, 
                         max_steps=max_steps, observability=observability,
                         partial_view_size=partial_view_size)


class AchieverBlocker11x11EnvV2(AchieverBlockerEnvV2):
    def __init__(self, max_keys=4, preference=None, cost=None, max_steps=None,
                 observability="full", partial_view_size=7):
        super().__init__(size=11, max_keys=max_keys, preference=preference, cost=cost, 
                         max_steps=max_steps, observability=observability,
                         partial_view_size=partial_view_size)


# Register environments
register(id="MiniGrid-AchieverBlocker-5x5-v2", entry_point="gym_minigrid.envs:AchieverBlocker5x5EnvV2")
register(id="MiniGrid-AchieverBlocker-9x9-v2", entry_point="gym_minigrid.envs:AchieverBlocker9x9EnvV2")  
register(id="MiniGrid-AchieverBlocker-11x11-v2", entry_point="gym_minigrid.envs:AchieverBlocker11x11EnvV2")