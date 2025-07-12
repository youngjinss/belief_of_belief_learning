from ..minigrid import *
from ..register import register
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class AchieverBlockerEnv(MiniGridEnv):
    """
    2-agent environment with achiever and blocker agents.
    
    Achiever: Pursues their goal by seeking a color key and door based on preference
    Blocker: Tries to block the door by inferring achiever's preference and blocking access
    
    Both agents have full observability but cannot overlap on the same position.
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

        # Allow custom max_steps
        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            grid_size=size,
            max_steps=max_steps,
            see_through_walls=True,  # Full observability
            agent_view_size=size,  # Agents can see entire grid
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

        # Clear achiever keys
        self.achiever_keys = []

        # Generate 4 keys and 4 doors with matching colors
        key_positions = []
        door_positions = []

        # Generate random positions for keys
        for color in self.door_colors:
            # Place key
            key = Key(color)
            # Make keys overlappable so agents can step on them
            key.can_overlap = lambda: True
            key_pos = self.place_obj(
                key, reject_fn=lambda env, pos: tuple(pos) in key_positions
            )
            key_positions.append(tuple(key_pos))

            # Place door on walls
            door_pos = self._place_door_on_wall(color, door_positions)
            door_positions.append(door_pos)
            
            # Make doors overlappable if achiever has the key
            door = self.grid.get(*door_pos)
            if isinstance(door, Door):
                # Create a closure that captures the door's color
                def make_door_can_overlap(door_color):
                    def door_can_overlap():
                        return not door.is_locked or door_color in self.achiever_keys
                    return door_can_overlap
                door.can_overlap = make_door_can_overlap(color)

        # Place agents using parent's place_agent method to set start_pos and start_dir
        self.place_agent()

        # Set target door based on highest preference
        self.target_door_color = max(self.preference, key=self.preference.get)

        # Generate mission
        self.mission = f"Achiever: collect {self.target_door_color} key and open door. Blocker: block achiever's target door."

    def _place_door_on_wall(self, color, existing_positions):
        """Place door at the center of a wall (same as KeyDoorEnv)"""
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

    def step(self, action_pair):
        """Execute actions for both agents"""
        achiever_action, blocker_action = action_pair
        
        # Store previous positions for collision handling
        self.achiever_prev_pos = self.achiever_pos.copy()
        self.blocker_prev_pos = self.blocker_pos.copy()
        
        # Get proposed new positions
        achiever_new_pos, achiever_new_dir = self._get_new_position(
            self.achiever_pos, self.achiever_dir, achiever_action
        )
        blocker_new_pos, blocker_new_dir = self._get_new_position(
            self.blocker_pos, self.blocker_dir, blocker_action
        )
        
        # Handle collision (agents cannot overlap)
        achiever_final_pos, blocker_final_pos = self._handle_collision(
            achiever_new_pos, blocker_new_pos
        )
        
        # Update positions and directions
        self.achiever_pos = achiever_final_pos
        self.achiever_dir = achiever_new_dir
        self.blocker_pos = blocker_final_pos
        self.blocker_dir = blocker_new_dir
        
        # Handle achiever-specific actions (pickup, door opening)
        achiever_reward = 0
        blocker_reward = 0
        
        # Auto pickup key for achiever (auto pickup when moving to key position)
        if np.array_equal(self.achiever_pos, achiever_new_pos):
            pickup_reward = self._auto_pickup_key(self.achiever_pos)
            achiever_reward += pickup_reward
        
        # Auto open door for achiever
        door_reward = self._auto_open_door(self.achiever_pos)
        achiever_reward += door_reward
        
        # Check termination conditions
        terminated = False
        truncated = False
        
        # Check if blocker chose "broken" action (action 5)
        if blocker_action == 5:  # "broken" action
            blocker_pos_tuple = tuple(self.blocker_pos)
            door_positions = self._get_door_positions()
            
            # Check if blocker is at a door position
            if blocker_pos_tuple in door_positions:
                door = self.grid.get(*self.blocker_pos)
                if isinstance(door, Door):
                    terminated = True
                    
                    # Check if blocker blocked the correct target door
                    if door.color == self.target_door_color:
                        blocker_reward += 1.0   # Success: blocked target door
                        achiever_reward -= 1.0  # Penalty for achiever
                    else:
                        blocker_reward -= 1.0   # Failure: blocked wrong door
                        achiever_reward += 0.5  # Small reward for achiever
        
        # Check if achiever opened target door
        if not terminated:  # Only check if game hasn't ended due to blocker's "done"
            door_positions = self._get_door_positions()
            achiever_pos_tuple = tuple(self.achiever_pos)
            if achiever_pos_tuple in door_positions:
                door = self.grid.get(*self.achiever_pos)
                if (
                    isinstance(door, Door)
                    and door.color == self.target_door_color
                    and door.is_open
                ):
                    terminated = True
                    achiever_reward += 10.0  # Big reward for achiever success
                    blocker_reward -= 5.0    # Penalty for blocker failure
        
        # Small reward for blocker being near target door
        target_door_pos = self._get_target_door_position()
        if target_door_pos is not None and np.linalg.norm(np.array(self.blocker_pos) - np.array(target_door_pos)) <= 1.0:
            blocker_reward += 0.1
        
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

    def _get_new_position(self, current_pos, current_dir, action):
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
        if cell is not None and not cell.can_overlap():
            new_pos = current_pos  # Stay in place if cannot overlap
            
        return new_pos, new_dir

    def _handle_collision(self, achiever_new_pos, blocker_new_pos):
        """Handle collision between agents (agents cannot overlap or swap positions)"""
        # Check if agents are trying to occupy the same cell
        if np.array_equal(achiever_new_pos, blocker_new_pos):
            # Collision detected: both agents stay in their previous positions
            return self.achiever_prev_pos, self.blocker_prev_pos
        
        # Check if agents are trying to swap positions
        elif (np.array_equal(achiever_new_pos, self.blocker_prev_pos) and 
              np.array_equal(blocker_new_pos, self.achiever_prev_pos)):
            # Position swap detected: both agents stay in their previous positions
            return self.achiever_prev_pos, self.blocker_prev_pos
        else:
            # No collision
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

    def _get_target_door_position(self):
        """Get target door position"""
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                obj = self.grid.get(x, y)
                if isinstance(obj, Door) and obj.color == self.target_door_color:
                    return (x, y)
        return None

    def _get_observations(self):
        """Generate observations for both agents"""
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
        
        return {
            'achiever': achiever_obs,
            'blocker': blocker_obs,
            'achiever_keys': achiever_keys_array,
            'achiever_pos': self.achiever_pos.astype(np.int32),
            'blocker_pos': self.blocker_pos.astype(np.int32)
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
class AchieverBlocker5x5Env(AchieverBlockerEnv):
    def __init__(self, max_keys=4, preference=None, cost=None, max_steps=None):
        super().__init__(size=5, max_keys=max_keys, preference=preference, cost=cost, max_steps=max_steps)


class AchieverBlocker9x9Env(AchieverBlockerEnv):
    def __init__(self, max_keys=4, preference=None, cost=None, max_steps=None):
        super().__init__(size=9, max_keys=max_keys, preference=preference, cost=cost, max_steps=max_steps)


class AchieverBlocker11x11Env(AchieverBlockerEnv):
    def __init__(self, max_keys=4, preference=None, cost=None, max_steps=None):
        super().__init__(size=11, max_keys=max_keys, preference=preference, cost=cost, max_steps=max_steps)


# Register environments
register(id="MiniGrid-AchieverBlocker-5x5-v1", entry_point="gym_minigrid.envs:AchieverBlocker5x5Env")
register(id="MiniGrid-AchieverBlocker-9x9-v1", entry_point="gym_minigrid.envs:AchieverBlocker9x9Env")  
register(id="MiniGrid-AchieverBlocker-11x11-v1", entry_point="gym_minigrid.envs:AchieverBlocker11x11Env")