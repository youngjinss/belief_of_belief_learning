from ..minigrid import *
from ..register import register
import numpy as np


class KeyDoorEnv(MiniGridEnv):
    """
    Environment with 4 keys and 4 doors where agent must collect keys to open doors.
    Agent has preferences and costs for different colored doors.
    """

    def __init__(self, size=9, max_keys=4, preference=None, cost=None, max_steps=None):
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

        # Track agent's inventory
        self.agent_keys = []
        self.target_door_color = None

        # Allow custom max_steps
        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            grid_size=size,
            max_steps=max_steps,
            see_through_walls=True,  # Full observability
            agent_view_size=size,  # Agent can see entire grid
        )

        # Use standard MiniGrid actions: left=0, right=1, forward=2, pickup=3, drop=4, toggle=5, done=6
        self.actions = MiniGridEnv.Actions
        self.action_space = spaces.Discrete(7)

    def _gen_grid(self, width, height):
        # Create empty grid
        self.grid = Grid(width, height)

        # Generate surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Clear agent keys
        self.agent_keys = []

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

    def _place_door_on_wall(self, color, existing_positions):
        """Place door at the center of a wall (similar to GoToDoor-5x5-v0)"""
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
            pos for pos in center_positions if pos not in existing_positions
        ]

        # Choose random position from available centers
        door_pos = self._rand_elem(available_positions)

        # Place door
        self.grid.set(*door_pos, Door(color, is_locked=True))

        return door_pos

    def step(self, action):
        # Handle regular movement actions first using parent class
        obs, reward, done, stuck, info = super().step(action)
        
        # Convert old API format to new Gymnasium format
        terminated = done
        truncated = False  # Don't terminate early due to stuck condition

        # Check for automatic key pickup when agent moves to a key position
        pickup_reward = self._auto_pickup_key()
        reward += pickup_reward
        
        # Check for automatic door opening when agent moves to a door position
        door_reward = self._auto_open_door()
        reward += door_reward
        
        # Check if agent opened target door (end condition)
        door_positions = self._get_door_positions()
        agent_pos_tuple = tuple(self.agent_pos)
        if agent_pos_tuple in door_positions:
            door = self.grid.get(*self.agent_pos)
            if (
                isinstance(door, Door)
                and door.color == self.target_door_color
                and door.is_open
            ):
                terminated = True

        return obs, reward, terminated, truncated, info

    def _auto_pickup_key(self):
        """Automatically pick up key when agent steps on it"""
        obj = self.grid.get(*self.agent_pos)

        if isinstance(obj, Key):
            # Check if agent can carry more keys
            if len(self.agent_keys) < self.max_keys:
                key_color = obj.color

                # Pick up key automatically
                self.agent_keys.append(key_color)
                self.grid.set(*self.agent_pos, None)

                # Give reward if it's target color key
                if key_color == self.target_door_color:
                    return 0.5  # Reward for collecting target key
                else:
                    return -self.cost[key_color]  # Cost for collecting wrong key

        return 0
    
    def _auto_open_door(self):
        """Automatically open door when agent steps on it and has the key"""
        obj = self.grid.get(*self.agent_pos)

        if isinstance(obj, Door) and obj.is_locked:
            door_color = obj.color
            
            # Check if agent has the key for this door
            if door_color in self.agent_keys:
                # Open the door automatically
                obj.is_open = True
                obj.is_locked = False
                
                # Give reward based on preference
                return self.preference[door_color]

        return 0

    def _get_door_positions(self):
        """Get all door positions"""
        door_positions = []
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                obj = self.grid.get(x, y)
                if isinstance(obj, Door):
                    door_positions.append((x, y))
        return door_positions

    def can_overlap(self, pos, obj):
        """Allow agent to overlap with keys and open doors"""
        cell = self.grid.get(*pos)
        return (
            cell is None
            or isinstance(cell, Key)
            or (isinstance(cell, Door) and cell.is_open)
        )

    def render(self, mode="human"):
        """Custom render"""
        img = super().render(mode)
        return img


class KeyDoor3x3Env(KeyDoorEnv):
    def __init__(self, max_keys=4, preference=None, cost=None, max_steps=None):
        super().__init__(size=3, max_keys=max_keys, preference=preference, cost=cost, max_steps=max_steps)


class KeyDoor5x5Env(KeyDoorEnv):
    def __init__(self, max_keys=4, preference=None, cost=None, max_steps=None):
        super().__init__(size=5, max_keys=max_keys, preference=preference, cost=cost, max_steps=max_steps)


class KeyDoor9x9Env(KeyDoorEnv):
    def __init__(self, max_keys=4, preference=None, cost=None, max_steps=None):
        super().__init__(size=9, max_keys=max_keys, preference=preference, cost=cost, max_steps=max_steps)


class KeyDoor11x11Env(KeyDoorEnv):
    def __init__(self, max_keys=4, preference=None, cost=None, max_steps=None):
        super().__init__(size=11, max_keys=max_keys, preference=preference, cost=cost, max_steps=max_steps)


# Register environments
register(id="MiniGrid-KeyDoor-3x3-v0", entry_point="gym_minigrid.envs:KeyDoor3x3Env")
register(id="MiniGrid-KeyDoor-5x5-v0", entry_point="gym_minigrid.envs:KeyDoor5x5Env")
register(id="MiniGrid-KeyDoor-9x9-v0", entry_point="gym_minigrid.envs:KeyDoor9x9Env")
register(
    id="MiniGrid-KeyDoor-11x11-v0", entry_point="gym_minigrid.envs:KeyDoor11x11Env"
)
