from gym_minigrid.minigrid import *
from gym_minigrid.register import register
import numpy as np


class KeyDoorEnv(MiniGridEnv):
    """
    Environment with 4 keys and 4 doors where agent must collect keys to open doors.
    Agent has preferences and costs for different colored doors.
    """

    def __init__(self, size=9, max_keys=4, preference=None, cost=None):
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

        super().__init__(
            grid_size=size,
            max_steps=4 * size**2,
            see_through_walls=True,  # Full observability
            agent_view_size=size,  # Agent can see entire grid
        )

        # Custom action space: up, down, left, right, stay, pickup
        self.actions = MiniGridEnv.Actions
        self.action_space = spaces.Discrete(6)

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
            key_pos = self.place_obj(
                Key(color), reject_fn=lambda env, pos: tuple(pos) in key_positions
            )
            key_positions.append(tuple(key_pos))

            # Place door on walls
            door_pos = self._place_door_on_wall(color, door_positions)
            door_positions.append(door_pos)

        # Place agent randomly
        self.place_agent()

        # Set target door based on highest preference
        self.target_door_color = max(self.preference, key=self.preference.get)

        # Generate mission
        self.mission = f"collect {self.target_door_color} key and open {self.target_door_color} door"

    def _place_door_on_wall(self, color, existing_positions):
        """Place door on one of the walls"""
        width, height = self.grid.width, self.grid.height

        # Possible wall positions (excluding corners)
        wall_positions = []
        # Top wall
        wall_positions.extend([(x, 0) for x in range(1, width - 1)])
        # Bottom wall
        wall_positions.extend([(x, height - 1) for x in range(1, width - 1)])
        # Left wall
        wall_positions.extend([(0, y) for y in range(1, height - 1)])
        # Right wall
        wall_positions.extend([(width - 1, y) for y in range(1, height - 1)])

        # Filter out existing positions
        available_positions = [
            pos for pos in wall_positions if pos not in existing_positions
        ]

        # Choose random position
        door_pos = self._rand_elem(available_positions)

        # Place door
        self.grid.set(*door_pos, Door(color, is_locked=True))

        return door_pos

    def step(self, action):
        # Custom pickup action
        if action == 5:  # pickup action
            pickup_reward = self._pickup_action()
            # Return observation without moving
            obs = self.gen_obs()
            return obs, pickup_reward, False, False, {}

        # Handle regular movement actions
        obs, reward, terminated, truncated, info = super().step(action)

        # Check if agent opened target door
        door_positions = self._get_door_positions()
        agent_pos_tuple = tuple(self.agent_pos)
        if agent_pos_tuple in door_positions:
            door = self.grid.get(*self.agent_pos)
            if (
                isinstance(door, Door)
                and door.color == self.target_door_color
                and door.is_open
            ):
                reward += self.preference[self.target_door_color]
                terminated = True

        return obs, reward, terminated, truncated, info

    def _pickup_action(self):
        """Handle pickup action"""
        # Check if there's a key at current position
        obj = self.grid.get(*self.agent_pos)

        if isinstance(obj, Key):
            # Check if agent can carry more keys
            if len(self.agent_keys) < self.max_keys:
                key_color = obj.color

                # Pick up key
                self.agent_keys.append(key_color)
                self.grid.set(*self.agent_pos, None)

                # Give reward if it's target color key
                if key_color == self.target_door_color:
                    return 0.5  # Reward for collecting target key
                else:
                    return -self.cost[key_color]  # Cost for collecting wrong key

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
        """Custom render to show agent's keys"""
        img = super().render(mode)

        if mode == "human":
            print(f"Agent keys: {self.agent_keys}")
            print(f"Target door: {self.target_door_color}")
            print(f"Keys capacity: {len(self.agent_keys)}/{self.max_keys}")

        return img


class KeyDoor3x3Env(KeyDoorEnv):
    def __init__(self, max_keys=4, preference=None, cost=None):
        super().__init__(size=3, max_keys=max_keys, preference=preference, cost=cost)


class KeyDoor5x5Env(KeyDoorEnv):
    def __init__(self, max_keys=4, preference=None, cost=None):
        super().__init__(size=5, max_keys=max_keys, preference=preference, cost=cost)


class KeyDoor9x9Env(KeyDoorEnv):
    def __init__(self, max_keys=4, preference=None, cost=None):
        super().__init__(size=9, max_keys=max_keys, preference=preference, cost=cost)


class KeyDoor11x11Env(KeyDoorEnv):
    def __init__(self, max_keys=4, preference=None, cost=None):
        super().__init__(size=11, max_keys=max_keys, preference=preference, cost=cost)


# Register environments
register(id="MiniGrid-KeyDoor-3x3-v0", entry_point="gym_minigrid.envs:KeyDoor3x3Env")
register(id="MiniGrid-KeyDoor-5x5-v0", entry_point="gym_minigrid.envs:KeyDoor5x5Env")
register(id="MiniGrid-KeyDoor-9x9-v0", entry_point="gym_minigrid.envs:KeyDoor9x9Env")
register(
    id="MiniGrid-KeyDoor-11x11-v0", entry_point="gym_minigrid.envs:KeyDoor11x11Env"
)
