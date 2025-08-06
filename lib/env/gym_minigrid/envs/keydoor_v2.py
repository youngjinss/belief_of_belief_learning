from ..minigrid import *
from ..register import register
import numpy as np


class KeyDoorEnvV2(MiniGridEnv):
    """
    Environment with 4 keys and 4 doors where agent must collect keys to open doors.
    Agent has preferences and costs for different colored doors.
    Version 2: Adds support for partial observation mode.
    """

    def __init__(self, size=9, max_keys=4, preference=None, cost=None, max_steps=None,
                 observability="full", partial_view_size=7):
        self.size = size
        self.max_keys = max_keys
        self.observability = observability
        self.partial_view_size = partial_view_size

        # Default preference and cost if not provided
        if preference is None:
            preference = {"red": 1.0, "green": 0.8, "blue": 0.6, "yellow": 0.4}
        if cost is None:
            cost = {"red": 0.1, "green": 0.2, "blue": 0.3, "yellow": 0.4}

        self.preference = preference
        self.cost = cost

        # 4 door colors
        self.door_colors = ["red", "green", "blue", "yellow"]

        # Track agent's inventory
        self.agent_keys = []
        self.target_door_color = None
        
        # Track state changes for compatibility with AchieverBlocker
        self.last_door_opened = None  # color of last door opened
        self.last_key_consumed = None  # color of last key consumed

        # Allow custom max_steps
        if max_steps is None:
            max_steps = 4 * size**2

        # Set observation parameters based on observability mode
        if observability == "partial":
            see_through_walls = False
            agent_view_size = partial_view_size
        else:  # full
            see_through_walls = True
            agent_view_size = size

        super().__init__(
            grid_size=size,
            max_steps=max_steps,
            see_through_walls=see_through_walls,
            agent_view_size=agent_view_size,
        )

        # Use standard MiniGrid actions: up=0, right=1, down=2, left=3, stay=4, pickup=5, toggle=6
        self.actions = MiniGridEnv.Actions
        self.action_space = spaces.Discrete(7)

    def _gen_grid(self, width, height):
        # Create empty grid
        self.grid = Grid(width, height)

        # Generate surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Clear agent keys and tracking
        self.agent_keys = []
        self.last_door_opened = None
        self.last_key_consumed = None

        # Generate 4 keys and 4 doors with matching colors
        key_positions = []
        door_positions = []

        # Generate random positions for keys
        for color in self.door_colors:
            # Place key
            key = Key(color)
            # Make keys overlappable so agent can step on them
            key.can_overlap = lambda: True
            key_pos = self.place_obj(
                key, reject_fn=lambda env, pos: tuple(pos) in key_positions
            )
            key_positions.append(tuple(key_pos))

            # Place door on walls
            door_pos = self._place_door_on_wall(color, door_positions)
            door_positions.append(door_pos)
            
            # Make doors overlappable if agent has the key
            door = self.grid.get(*door_pos)
            if isinstance(door, Door):
                # Create a closure that captures the door's color
                def make_door_can_overlap(door_color):
                    def door_can_overlap():
                        return not door.is_locked or door_color in self.agent_keys
                    return door_can_overlap
                door.can_overlap = make_door_can_overlap(color)

        # Place agent randomly
        self.place_agent()

        # Set target door based on highest preference
        self.target_door_color = max(self.preference, key=self.preference.get)

        # Generate mission
        self.mission = f"collect {self.target_door_color} key and open {self.target_door_color} door"

    def _place_door_on_wall(self, color, existing_doors):
        """Place a door on a wall position"""
        walls = []
        for x in range(1, self.width - 1):
            walls.append((x, 0))  # Top wall
            walls.append((x, self.height - 1))  # Bottom wall
        for y in range(1, self.height - 1):
            walls.append((0, y))  # Left wall
            walls.append((self.width - 1, y))  # Right wall
        
        # Filter out positions with existing doors
        available_walls = [w for w in walls if w not in existing_doors]
        
        # Randomly select a wall position
        door_pos = available_walls[self._rand_int(0, len(available_walls))]
        
        # Place door
        door = Door(color, is_locked=True)
        self.grid.set(*door_pos, door)
        
        return door_pos

    def reset(self, **kwargs):
        """Reset environment and return observation"""
        # Call parent reset
        obs = super().reset(**kwargs)
        
        # Generate observation based on observability mode
        obs_dict = self._get_observations()
        
        info = {
            'target_door_color': self.target_door_color,
            'agent_keys': self.agent_keys.copy()
        }
        
        return obs_dict, info

    def step(self, action):
        # Reset state change tracking
        self.last_door_opened = None
        self.last_key_consumed = None
        
        # Store previous position
        prev_pos = self.agent_pos.copy()
        
        # Execute action
        obs, reward, done, stuck, _ = super().step(action)
        
        # Handle auto pickup of keys when stepping on them
        key_reward = self._auto_pickup_key(self.agent_pos)
        reward += key_reward
        
        # Handle auto open door when stepping on it with key
        door_reward = self._auto_open_door(self.agent_pos)
        reward += door_reward
        
        # Check termination
        terminated = door_reward > 0  # Episode ends when any door is opened
        truncated = self.step_count >= self.max_steps
        
        # Generate observations
        obs_dict = self._get_observations()
        
        info = {
            'agent_keys': self.agent_keys.copy(),
            'target_door_color': self.target_door_color,
            'step_count': self.step_count
        }
        
        return obs_dict, reward, terminated, truncated, info

    def _auto_pickup_key(self, agent_pos):
        """Auto pickup key when stepping on it"""
        obj = self.grid.get(*agent_pos)
        if isinstance(obj, Key):
            if len(self.agent_keys) < self.max_keys:
                key_color = obj.color
                self.agent_keys.append(key_color)
                self.grid.set(*agent_pos, None)
                
                # Track state change
                self.last_key_consumed = key_color
                
                if key_color == self.target_door_color:
                    return 0.5
                else:
                    return -self.cost[key_color]
        return 0

    def _auto_open_door(self, agent_pos):
        """Auto open door when stepping on it and having the key"""
        obj = self.grid.get(*agent_pos)

        if isinstance(obj, Door) and obj.is_locked:
            door_color = obj.color
            
            # Check if agent has the key for this door
            if door_color in self.agent_keys:
                # Open the door automatically
                obj.is_open = True
                obj.is_locked = False
                
                # Track state change
                self.last_door_opened = door_color
                
                # Give reward based on preference
                return self.preference[door_color]

        return 0

    def _get_key_positions(self):
        """Get key positions with their colors"""
        key_positions = {}
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                obj = self.grid.get(x, y)
                if obj is not None and obj.type == "key":
                    key_positions[obj.color] = (x, y)
        return key_positions

    def _get_door_positions_with_colors(self):
        """Get door positions with their colors"""
        door_positions = {}
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                obj = self.grid.get(x, y)
                if obj is not None and obj.type == "door":
                    door_positions[obj.color] = (x, y)
        return door_positions
    
    def _get_door_positions(self):
        """Get all door positions"""
        door_positions = []
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                obj = self.grid.get(x, y)
                if obj is not None and obj.type == "door":
                    door_positions.append((x, y))
        return door_positions

    def _get_wall_positions(self):
        """Get all wall positions"""
        wall_positions = []
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                obj = self.grid.get(x, y)
                if obj is not None and obj.type == "wall":
                    wall_positions.append((x, y))
        return wall_positions

    def _get_observations(self):
        """Generate observations based on observability mode"""
        if self.observability == "partial":
            return self._get_partial_observations()
        else:
            return self._get_full_observations()
    
    def _get_full_observations(self):
        """Generate full observations (original implementation)"""
        # Get standard MiniGrid observation
        obs = super().gen_obs()
        
        # Create agent keys array
        agent_keys_array = np.zeros(len(self.door_colors), dtype=np.int32)
        for i, color in enumerate(self.door_colors):
            if color in self.agent_keys:
                agent_keys_array[i] = 1
        
        # Get positions
        key_positions = self._get_key_positions()
        door_positions = self._get_door_positions_with_colors()
        wall_positions = self._get_wall_positions()
        
        # Get grid size info
        grid_info = {
            'width': self.width,
            'height': self.height
        }
        
        return {
            'agent': obs,
            'agent_keys': agent_keys_array,
            'agent_pos': self.agent_pos.astype(np.int32),
            'target_door_color': self.target_door_color,
            'key_positions': key_positions,
            'door_positions': door_positions,
            'wall_positions': wall_positions,
            'grid_info': grid_info,
            'observability': self.observability
        }
    
    def _get_partial_observations(self):
        """Generate partial observations"""
        # Get standard MiniGrid observation
        obs = super().gen_obs()
        
        # Generate the partial grid view
        grid, vis_mask = self.gen_obs_grid()
        
        # Get all keys and doors positions
        all_key_positions = self._get_key_positions()
        all_door_positions = self._get_door_positions_with_colors()
        
        # Get visible keys and doors
        visible_keys = {}
        visible_doors = {}
        
        for color, pos in all_key_positions.items():
            if pos is not None:
                rel_coords = self.relative_coords(*pos)
                if rel_coords is not None:
                    vx, vy = rel_coords
                    if vis_mask[vx, vy]:
                        visible_keys[color] = pos
        
        for color, pos in all_door_positions.items():
            rel_coords = self.relative_coords(*pos)
            if rel_coords is not None:
                vx, vy = rel_coords
                if vis_mask[vx, vy]:
                    visible_doors[color] = pos
        
        # Create agent keys array
        agent_keys_array = np.zeros(len(self.door_colors), dtype=np.int32)
        for i, color in enumerate(self.door_colors):
            if color in self.agent_keys:
                agent_keys_array[i] = 1
        
        # Get grid size info
        grid_info = {
            'width': self.width,
            'height': self.height
        }
        
        return {
            'agent': obs,
            'agent_keys': agent_keys_array,
            'agent_pos': self.agent_pos.astype(np.int32),
            'target_door_color': self.target_door_color,
            'visible_keys': visible_keys,
            'visible_doors': visible_doors,
            'grid_info': grid_info,
            'observability': self.observability
        }


# Size variants
class KeyDoor5x5EnvV2(KeyDoorEnvV2):
    def __init__(self, max_keys=4, preference=None, cost=None, max_steps=None,
                 observability="full", partial_view_size=5):
        super().__init__(size=5, max_keys=max_keys, preference=preference, cost=cost,
                         max_steps=max_steps, observability=observability,
                         partial_view_size=partial_view_size)


class KeyDoor9x9EnvV2(KeyDoorEnvV2):
    def __init__(self, max_keys=4, preference=None, cost=None, max_steps=None,
                 observability="full", partial_view_size=7):
        super().__init__(size=9, max_keys=max_keys, preference=preference, cost=cost,
                         max_steps=max_steps, observability=observability,
                         partial_view_size=partial_view_size)


class KeyDoor11x11EnvV2(KeyDoorEnvV2):
    def __init__(self, max_keys=4, preference=None, cost=None, max_steps=None,
                 observability="full", partial_view_size=7):
        super().__init__(size=11, max_keys=max_keys, preference=preference, cost=cost,
                         max_steps=max_steps, observability=observability,
                         partial_view_size=partial_view_size)


# Register environments
register(id="MiniGrid-KeyDoor-5x5-v2", entry_point="gym_minigrid.envs:KeyDoor5x5EnvV2")
register(id="MiniGrid-KeyDoor-9x9-v2", entry_point="gym_minigrid.envs:KeyDoor9x9EnvV2")
register(id="MiniGrid-KeyDoor-11x11-v2", entry_point="gym_minigrid.envs:KeyDoor11x11EnvV2")