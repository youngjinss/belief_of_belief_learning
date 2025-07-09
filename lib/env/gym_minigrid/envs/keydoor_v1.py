from ..minigrid import *
from ..register import register
import numpy as np


class KeyDoorV1Env(MiniGridEnv):
    """
    KeyDoor Environment Version 1 - Manual Interaction Required
    
    This version requires explicit actions for all interactions:
    - Keys must be picked up using the pickup action (not automatic)
    - Doors must be opened using the toggle action (not automatic)
    - Episodes must be ended using the done action (not automatic)
    - Inventory management with drop action for limited carrying capacity
    
    This addresses the action density problem by making all 7 actions necessary.
    """

    def __init__(self, size=9, max_keys=2, preference=None, cost=None, max_steps=None):
        self.size = size
        self.max_keys = max_keys  # Reduced from 4 to 2 to require inventory management

        # Default preference and cost if not provided
        if preference is None:
            preference = {"red": 1.0, "green": 0.8, "blue": 0.6, "yellow": 0.4}
        if cost is None:
            cost = {"red": 0.1, "green": 0.2, "blue": 0.3, "yellow": 0.4}

        self.preference = preference
        self.cost = cost

        # 4 door colors
        self.door_colors = ["red", "green", "blue", "yellow"]

        # Track agent's inventory (manual management)
        self.agent_keys = []
        self.target_door_color = None
        self.target_door_reached = False  # Track if agent reached target door

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
        self.target_door_reached = False

        # Generate 4 keys and 4 doors with matching colors
        key_positions = []
        door_positions = []

        # Create lists of available positions (avoid walls and corners)
        available_positions = []
        for x in range(2, width - 2):
            for y in range(2, height - 2):
                available_positions.append((x, y))

        # Shuffle positions
        np.random.shuffle(available_positions)

        # Place 4 keys
        for i, color in enumerate(self.door_colors):
            if i < len(available_positions):
                pos = available_positions[i]
                key_positions.append(pos)
                self.grid.set(*pos, Key(color))

        # Place 4 doors (avoid key positions)
        door_available = available_positions[4:]
        for i, color in enumerate(self.door_colors):
            if i < len(door_available):
                pos = door_available[i]
                door_positions.append(pos)
                door = Door(color, is_locked=True)  # All doors start locked
                self.grid.set(*pos, door)

        # Place agent in a random empty position
        remaining_positions = available_positions[8:]
        if remaining_positions:
            agent_pos = remaining_positions[0]
            self.agent_pos = agent_pos
        else:
            # Fallback to center
            self.agent_pos = (width // 2, height // 2)

        # Set random agent direction
        self.agent_dir = np.random.randint(0, 4)

        # Set target door based on highest preference
        self.target_door_color = max(self.preference, key=self.preference.get)

        # Set mission
        self.mission = f"collect {self.target_door_color} key, open {self.target_door_color} door, and use done action"

    def reset(self, **kwargs):
        """Reset environment and clear target door flag"""
        obs = super().reset(**kwargs)
        self.target_door_reached = False
        return obs

    def step(self, action):
        """
        Execute action - NO automatic interactions
        All interactions must be done through explicit actions
        """
        self.step_count += 1
        reward = 0
        terminated = False
        truncated = False

        # Handle different actions
        if action == self.actions.left:
            self.agent_dir = (self.agent_dir - 1) % 4
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4
        elif action == self.actions.forward:
            reward = self._move_forward()
        elif action == self.actions.pickup:
            reward = self._pickup_action()
        elif action == self.actions.drop:
            reward = self._drop_action()
        elif action == self.actions.toggle:
            reward = self._toggle_action()
        elif action == self.actions.done:
            reward, terminated = self._done_action()

        # Check if max steps reached
        if self.step_count >= self.max_steps:
            truncated = True

        # Generate observation
        obs = self.gen_obs()

        return obs, reward, terminated, truncated, {}

    def _move_forward(self):
        """Move forward (no automatic interactions)"""
        # Get the position in front of the agent
        fwd_pos = self.front_pos
        
        # Check if we can move forward
        fwd_cell = self.grid.get(*fwd_pos)
        
        if fwd_cell is None or fwd_cell.can_overlap():
            self.agent_pos = fwd_pos
            return -0.01  # Small movement cost
        else:
            return -0.05  # Penalty for trying to move into obstacle

    def _pickup_action(self):
        """Pick up key in front of agent (explicit action required)"""
        fwd_pos = self.front_pos
        fwd_cell = self.grid.get(*fwd_pos)
        
        if isinstance(fwd_cell, Key):
            # Check if agent has inventory space
            if len(self.agent_keys) >= self.max_keys:
                return -0.1  # Penalty for trying to pick up when inventory is full
            
            # Pick up the key
            key_color = fwd_cell.color
            self.agent_keys.append(key_color)
            self.grid.set(*fwd_pos, None)
            
            # Give reward/penalty based on key color
            if key_color == self.target_door_color:
                return 0.5  # Positive reward for target key
            else:
                return -self.cost[key_color]  # Penalty for wrong key
        
        return -0.05  # Penalty for trying to pick up when no key is present

    def _drop_action(self):
        """Drop a key (explicit action required for inventory management)"""
        if not self.agent_keys:
            return -0.05  # Penalty for trying to drop when no keys held
        
        # Drop the last key picked up
        dropped_key_color = self.agent_keys.pop()
        
        # Place the key at agent's current position
        current_cell = self.grid.get(*self.agent_pos)
        if current_cell is None:
            self.grid.set(*self.agent_pos, Key(dropped_key_color))
            return -0.02  # Small penalty for dropping (inventory management cost)
        else:
            # Can't drop here, add key back
            self.agent_keys.append(dropped_key_color)
            return -0.1  # Penalty for trying to drop in occupied cell

    def _toggle_action(self):
        """Toggle door in front of agent (explicit action required)"""
        fwd_pos = self.front_pos
        fwd_cell = self.grid.get(*fwd_pos)
        
        if isinstance(fwd_cell, Door):
            door_color = fwd_cell.color
            
            # Check if agent has the matching key
            if door_color in self.agent_keys:
                if fwd_cell.is_locked:
                    # Open the door and consume the key
                    fwd_cell.is_open = True
                    fwd_cell.is_locked = False
                    self.agent_keys.remove(door_color)
                    
                    # Check if this is the target door
                    if door_color == self.target_door_color:
                        self.target_door_reached = True
                        return self.preference[door_color]  # High reward for target door
                    else:
                        return self.preference[door_color] * 0.5  # Reduced reward for non-target doors
                else:
                    return -0.02  # Small penalty for trying to toggle already open door
            else:
                return -0.2  # Penalty for trying to open door without key
        
        return -0.05  # Penalty for trying to toggle when no door is present

    def _done_action(self):
        """End episode (explicit action required)"""
        # Check if agent has completed the task
        if self.target_door_reached:
            # Calculate reward based on efficiency
            efficiency_bonus = 1 - 0.9 * (self.step_count / self.max_steps)
            return max(0.1, efficiency_bonus), True  # Success with efficiency bonus
        else:
            return -0.5, True  # Large penalty for ending episode without completing task

    def render(self, mode="human"):
        """Enhanced render to show inventory and target"""
        img = super().render(mode)
        
        # Add inventory and target info to the rendered image
        if hasattr(self, 'window') and self.window:
            import pygame
            
            # Display inventory
            inventory_text = f"Keys: {self.agent_keys}"
            target_text = f"Target: {self.target_door_color}"
            reached_text = f"Target reached: {self.target_door_reached}"
            
            # You could add text rendering here if needed
            
        return img


class KeyDoor3x3V1Env(KeyDoorV1Env):
    def __init__(self, max_keys=2, preference=None, cost=None, max_steps=None):
        super().__init__(size=3, max_keys=max_keys, preference=preference, cost=cost, max_steps=max_steps)


class KeyDoor5x5V1Env(KeyDoorV1Env):
    def __init__(self, max_keys=2, preference=None, cost=None, max_steps=None):
        super().__init__(size=5, max_keys=max_keys, preference=preference, cost=cost, max_steps=max_steps)


class KeyDoor9x9V1Env(KeyDoorV1Env):
    def __init__(self, max_keys=2, preference=None, cost=None, max_steps=None):
        super().__init__(size=9, max_keys=max_keys, preference=preference, cost=cost, max_steps=max_steps)


class KeyDoor11x11V1Env(KeyDoorV1Env):
    def __init__(self, max_keys=2, preference=None, cost=None, max_steps=None):
        super().__init__(size=11, max_keys=max_keys, preference=preference, cost=cost, max_steps=max_steps)


# Register v1 environments
register(id="MiniGrid-KeyDoor-3x3-v1", entry_point="gym_minigrid.envs:KeyDoor3x3V1Env")
register(id="MiniGrid-KeyDoor-5x5-v1", entry_point="gym_minigrid.envs:KeyDoor5x5V1Env")
register(id="MiniGrid-KeyDoor-9x9-v1", entry_point="gym_minigrid.envs:KeyDoor9x9V1Env")
register(id="MiniGrid-KeyDoor-11x11-v1", entry_point="gym_minigrid.envs:KeyDoor11x11V1Env")