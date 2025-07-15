import numpy as np
import heapq
from collections import deque
import sys
import os

# Add the lib directory to the path
lib_path = os.path.join(os.path.dirname(__file__), "..", "..", "lib")
sys.path.insert(0, lib_path)

# Add the env directory to the path
env_path = os.path.join(lib_path, "env")
sys.path.insert(0, env_path)

from gym_minigrid.minigrid import Key, Door, Wall
from lib.utils.seed import set_seed

# Add current directory for config import
sys.path.append(os.path.dirname(__file__))
from config import Config

# Set seed using Config default value
config = Config()
set_seed(config.seed)


class Node:
    """
    A node class for A* Pathfinding
    """

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

    def __repr__(self):
        return f"{self.position} - g: {self.g} h: {self.h} f: {self.f}"

    def __lt__(self, other):
        return self.f < other.f

    def __gt__(self, other):
        return self.f > other.f


class AStarAgent:
    """
    A* Agent adapted for KeyDoor environments
    """

    def __init__(self, observability="full"):
        self.observability = observability
        self.agent_pos = None
        self.grid = None
        self.current_target = None
        self.path = []

        # Grid dimensions - will be set from observations
        self.width = None
        self.height = None

        # Track agent's collected keys
        self.collected_keys = set()

        # Strategy: first collect target key, then go to target door
        self.strategy_phase = "collect_key"  # "collect_key" or "open_door"
        self.target_door_color = None

        # Adjacent squares for movement (up, down, left, right)
        # Note: In grid coordinates (row, col), positive row is down
        self.adjacent_moves = [
            (-1, 0),  # up (row - 1)
            (1, 0),  # down (row + 1)
            (0, -1),  # left (col - 1)
            (0, 1),  # right (col + 1)
        ]

        # MiniGrid action mapping: 0=up, 1=right, 2=down, 3=left, 4=stay, 5=pickup, 6=toggle

    def update_observation(self, obs):
        """Update agent's understanding of the environment"""
        if obs is None:
            return

        # Update grid dimensions from observations
        if "grid_info" in obs and self.width is None:
            self.width = obs["grid_info"]["width"]
            self.height = obs["grid_info"]["height"]

        # Get current achiever position from observations
        new_pos = tuple(obs["achiever_pos"])

        if new_pos != self.agent_pos:
            # If we have a path and we moved to the expected next position, advance the path
            if self.path and len(self.path) >= 2 and self.path[1] == new_pos:
                self.path.pop(0)
            else:
                # Position changed unexpectedly, clear path
                self.path = []
            self.agent_pos = new_pos
        else:
            # Position didn't change - could be turning or stuck
            pass

        # Get the grid from achiever's visual observation
        self.grid = obs["achiever"]

        # Update collected keys based on achiever's key inventory
        achiever_keys_array = obs["achiever_keys"]
        color_map = ["red", "green", "blue", "yellow"]
        self.collected_keys = set()
        for i, has_key in enumerate(achiever_keys_array):
            if has_key > 0 and i < len(color_map):
                self.collected_keys.add(color_map[i])

    def get_action(self, obs):
        """
        Get the next action for the agent
        """
        self.update_observation(obs)

        # Infer target door color from observations
        if self.target_door_color is None:
            self.target_door_color = self._infer_target_door_color(obs)

        target_key_color = self.target_door_color

        if self.strategy_phase == "collect_key":
            # Check if we already have the target key
            if target_key_color in self.collected_keys:
                self.strategy_phase = "open_door"
                self.path = []  # Clear path to recalculate
            else:
                # Find and collect the target key
                return self._collect_target_key(target_key_color, obs)

        elif self.strategy_phase == "open_door":
            # Go to target door and open it
            return self._open_target_door(target_key_color, obs)

        # Better fallback: try random movement instead of staying stuck
        return np.random.choice([0, 1, 2, 3])  # random movement

    def _infer_target_door_color(self, obs=None):
        """Infer target door color from observations."""
        # Use target door color from observations if available
        if obs and "target_door_color" in obs:
            return obs["target_door_color"]

        # Fallback: use first available door color
        if obs and "door_positions" in obs:
            door_colors = list(obs["door_positions"].keys())
            if door_colors:
                return door_colors[0]

        # Final fallback
        return "red"

    def _collect_target_key(self, target_key_color, obs=None):
        """Strategy to collect the target key"""
        # Find target key position
        target_key_pos = self._find_object_position(Key, target_key_color, obs)
        if target_key_pos is None:
            return 4  # Stay if key not found

        # Check if we're already at the key position
        if self.agent_pos == target_key_pos:
            return 4  # Stay - key pickup is automatic

        # Navigate to the key position (key will be picked up automatically when agent steps on it)
        return self._navigate_to_position(target_key_pos)

    def _open_target_door(self, target_door_color, obs=None):
        """Strategy to open the target door"""
        # Find target door position
        target_door_pos = self._find_object_position(Door, target_door_color, obs)
        if target_door_pos is None:
            return 4  # Stay if door not found

        # Check if we're at the door position
        if self.agent_pos == target_door_pos:
            door = self.grid.get(*target_door_pos)
            if isinstance(door, Door) and door.is_open:
                # Already opened, episode should end
                return 4  # Stay on the opened door
            elif (
                isinstance(door, Door)
                and door.is_locked
                and target_door_color in self.collected_keys
            ):
                # Door will be opened automatically when agent steps on it
                return 4  # Stay - door opening is automatic

        # Navigate to door (door will open automatically when agent steps on it)
        return self._navigate_to_position(target_door_pos)

    def _find_object_position(self, obj_type, color, obs=None):
        """Find position of specific object type and color"""
        if obs is None:
            return None

        if obj_type.__name__ == "Key":
            # Get key positions from observations
            if "key_positions" in obs:
                return obs["key_positions"].get(color, None)
        elif obj_type.__name__ == "Door":
            # Get door positions from observations
            if "door_positions" in obs:
                return obs["door_positions"].get(color, None)

        return None

    def _navigate_to_position(self, target_pos):
        """Navigate to target position using A* pathfinding and direct movement"""
        # Always recalculate if path is empty or if we're not on the expected path
        if (
            not self.path
            or len(self.path) < 2
            or (len(self.path) >= 1 and self.path[0] != self.agent_pos)
        ):
            # Calculate new path if needed
            self.path = self._astar_pathfind(self.agent_pos, target_pos)

        if len(self.path) >= 2:
            # Get next step in path
            next_pos = self.path[1]  # path[0] should be current position

            # Verify current position matches path start
            if self.path[0] != self.agent_pos:
                self.path = self._astar_pathfind(self.agent_pos, target_pos)
                if len(self.path) < 2:
                    return 4  # Stay if no path found
                next_pos = self.path[1]

            # Calculate movement direction needed
            dx = next_pos[0] - self.agent_pos[0]
            dy = next_pos[1] - self.agent_pos[1]

            # Map movement to direct action
            if dx == 0 and dy == -1:
                return 0  # Up
            elif dx == 1 and dy == 0:
                return 1  # Right
            elif dx == 0 and dy == 1:
                return 2  # Down
            elif dx == -1 and dy == 0:
                return 3  # Left
            else:
                # Shouldn't happen with proper A* path
                return 4  # Stay

        return 4  # Stay if no path found

    def _astar_pathfind(self, start_pos, goal_pos):
        """
        A* pathfinding algorithm adapted for KeyDoor environment
        """
        # Create start and end nodes
        start_node = Node(None, start_pos)
        end_node = Node(None, goal_pos)

        # Initialize open and closed lists
        open_list = []
        closed_list = []

        # Add start node to open list
        heapq.heappush(open_list, start_node)

        # Main A* loop
        iteration_count = 0
        while open_list:
            iteration_count += 1
            if iteration_count > 200:  # Prevent infinite loops
                break

            # Get node with lowest f cost
            current_node = heapq.heappop(open_list)

            # Add current node to closed list
            closed_list.append(current_node)

            # Check if we reached the goal
            if current_node.position == end_node.position:
                path = []
                current = current_node
                while current is not None:
                    path.append(current.position)
                    current = current.parent
                return path[::-1]  # Return reversed path

            # Generate children
            children = []
            for move in self.adjacent_moves:
                node_pos = (
                    current_node.position[0] + move[0],
                    current_node.position[1] + move[1],
                )

                # Check if position is within grid bounds
                # Use grid dimensions from observations
                width = self.width if self.width is not None else 9  # fallback
                height = self.height if self.height is not None else 9  # fallback

                if (
                    node_pos[0] < 0
                    or node_pos[0] >= width
                    or node_pos[1] < 0
                    or node_pos[1] >= height
                ):
                    continue

                # Check if position is walkable
                if not self._is_walkable(node_pos):
                    continue

                # Create new node
                new_node = Node(current_node, node_pos)
                children.append(new_node)

            # Process children
            for child in children:
                # Skip if child is in closed list
                if any(
                    closed_child.position == child.position
                    for closed_child in closed_list
                ):
                    continue

                # Calculate g, h, and f values
                child.g = current_node.g + 1
                child.h = self._heuristic(child.position, end_node.position)
                child.f = child.g + child.h

                # Skip if child is already in open list with better g value
                if any(
                    open_node.position == child.position and child.g > open_node.g
                    for open_node in open_list
                ):
                    continue

                # Add child to open list
                heapq.heappush(open_list, child)

        return []  # Return empty path if no path found

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

    def _heuristic(self, pos1, pos2):
        """Manhattan distance heuristic"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def reset(self):
        """Reset agent state for new episode"""
        self.agent_pos = None
        self.grid = None
        self.current_target = None
        self.path = []
        self.collected_keys = set()
        self.strategy_phase = "collect_key"


class Level0ValueAchiever:
    """
    Level-0 Value-based Achiever Agent with stochastic action selection using value iteration
    Updated for KeyDoor environment with automatic key pickup and door opening

    Strategy:
    - Direct approach to target door color
    - Uses value iteration for optimal path planning
    - Accounts for automatic key pickup when stepping on keys
    - Accounts for automatic door opening when stepping on doors with correct key
    - Compatible with 7-action MiniGrid action space
    - Stochastic policy with temperature-based action selection
    """

    def __init__(
        self,
        observability="full",
        movement_cost=0.01,
        wall_penalty=2.0,
        gamma=0.99,
        temperature=0.1,
    ):
        self.observability = observability
        self.agent_pos = None
        self.grid = None
        self.current_target = None
        self.path = []
        self.target_door_color = None

        # Grid dimensions - will be set from observations
        self.width = None
        self.height = None

        # Track agent's collected keys
        self.collected_keys = set()

        # Strategy: first collect target key, then go to target door
        self.strategy_phase = "collect_key"  # "collect_key" or "open_door"

        # Value iteration parameters
        self.movement_cost = movement_cost
        self.wall_penalty = wall_penalty
        self.gamma = gamma
        self.temperature = temperature

        # Value function and policy
        self.value_function = None
        self.policy = None
        self.converged = False

        # MiniGrid action mapping: 0=up, 1=right, 2=down, 3=left, 4=stay, 5=pickup, 6=toggle
        # Value iteration uses 4 movement actions: 0=up, 1=right, 2=down, 3=left
        # Grid coordinate system: (x, y) where x=column, y=row, positive y is down
        self.actions = [
            (0, -1),
            (1, 0),
            (0, 1),
            (-1, 0),
        ]  # dx, dy for up, right, down, left

    def update_observation(self, obs):
        """Update agent's understanding of the environment"""
        if obs is None:
            return

        # Update grid dimensions from observations
        if "grid_info" in obs and self.width is None:
            self.width = obs["grid_info"]["width"]
            self.height = obs["grid_info"]["height"]

        # Get current achiever position from observations
        new_pos = tuple(obs["achiever_pos"])

        if new_pos != self.agent_pos:
            self.agent_pos = new_pos

        # Get the grid from achiever's visual observation
        self.grid = obs["achiever"]

        # Update collected keys based on achiever's key inventory
        achiever_keys_array = obs["achiever_keys"]
        color_map = ["red", "green", "blue", "yellow"]
        self.collected_keys = set()
        for i, has_key in enumerate(achiever_keys_array):
            if has_key > 0 and i < len(color_map):
                self.collected_keys.add(color_map[i])

    def get_action(self, obs):
        """
        Get the next action for the agent using value iteration
        """
        self.update_observation(obs)

        # Infer target door color from observations
        if self.target_door_color is None:
            self.target_door_color = self._infer_target_door_color(obs)

        target_key_color = self.target_door_color

        if self.strategy_phase == "collect_key":
            # Check if we already have the target key
            if target_key_color in self.collected_keys:
                self.strategy_phase = "open_door"
            else:
                # Find and collect the target key
                return self._collect_target_key(target_key_color, obs)

        elif self.strategy_phase == "open_door":
            # Go to target door and open it
            return self._open_target_door(target_key_color, obs)

        # Better fallback: try random movement instead of staying stuck
        return np.random.choice([0, 1, 2, 3])  # random movement

    def _collect_target_key(self, target_key_color, obs=None):
        """Strategy to collect the target key using value iteration"""
        # Find target key position
        target_key_pos = self._find_object_position(Key, target_key_color, obs)
        if target_key_pos is None:
            return 4  # Stay if key not found

        # Check if we're already at the key position
        if self.agent_pos == target_key_pos:
            return 4  # Stay - key pickup is automatic when agent steps on it

        # Use value iteration to navigate to key (key will be picked up automatically)
        return self._navigate_with_value_iteration(target_key_pos, obs)

    def _open_target_door(self, target_door_color, obs=None):
        """Strategy to open the target door using value iteration"""
        # Find target door position
        target_door_pos = self._find_object_position(Door, target_door_color, obs)
        if target_door_pos is None:
            return 4  # Stay if door not found

        # Check if we're at the door position
        if self.agent_pos == target_door_pos:
            door = self.grid.get(*target_door_pos)
            if isinstance(door, Door) and door.is_open:
                # Already opened, episode should end
                return 4  # Stay on the opened door
            elif (
                isinstance(door, Door)
                and door.is_locked
                and target_door_color in self.collected_keys
            ):
                # Door will be opened automatically when agent steps on it
                return 4  # Stay - door opening is automatic

        # Navigate to door (door will open automatically when agent steps on it)
        return self._navigate_with_value_iteration(target_door_pos, obs)

    def _find_object_position(self, obj_type, color, obs=None):
        """Find position of specific object type and color"""
        if obs is None:
            return None

        if obj_type.__name__ == "Key":
            # Get key positions from observations
            if "key_positions" in obs:
                return obs["key_positions"].get(color, None)
        elif obj_type.__name__ == "Door":
            # Get door positions from observations
            if "door_positions" in obs:
                return obs["door_positions"].get(color, None)

        return None

    def _navigate_with_value_iteration(self, target_pos, obs=None):
        """Navigate using value iteration and convert to MiniGrid actions"""
        # Run value iteration to get optimal action
        optimal_action = self._plan_value_iteration(target_pos, obs)

        if optimal_action is None:
            return 4  # Stay if no action found

        # Convert value iteration action to MiniGrid action
        return self._convert_to_minigrid_action(optimal_action)

    def _plan_value_iteration(
        self, target_pos, obs=None, max_iterations=100, convergence_threshold=0.01
    ):
        """
        Vectorized value iteration to compute optimal action for reaching target
        """
        # Get grid size from instance dimensions
        width = self.width if self.width is not None else 9  # fallback
        height = self.height if self.height is not None else 9  # fallback
        n_actions = 4

        # Get blocker position from observations
        blocker_pos = None
        if obs is not None and "blocker_pos" in obs:
            blocker_pos = tuple(obs["blocker_pos"])

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
        # that matches the current _is_walkable implementation
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

                # Apply blocker position penalty
                if blocker_pos is not None:
                    blocker_conflict = (
                        valid_moves
                        & (valid_next_x == blocker_pos[0])
                        & (valid_next_y == blocker_pos[1])
                    )
                    rewards = np.where(
                        blocker_conflict, rewards - self.wall_penalty, rewards
                    )
                    next_values = np.where(
                        blocker_conflict,
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
                blocker_pos,
            )
            q_values.append(q_val)

        # Choose action with softmax policy
        if self.temperature > 0:
            q_values = np.array(q_values)
            q_values_clipped = np.clip(q_values, -100, 100)
            exp_q = np.exp(q_values_clipped / self.temperature)
            action_probs = exp_q / np.sum(exp_q)

            # Sample action stochastically
            action = np.random.choice(n_actions, p=action_probs)
        else:
            # Deterministic policy
            action = np.argmax(q_values)

        return action

    def _evaluate_action(
        self, pos, action, value_function, target_pos, width, height, blocker_pos=None
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
                # Check if new position conflicts with blocker position
                if blocker_pos is not None and new_pos == blocker_pos:
                    reward -= (
                        self.wall_penalty
                    )  # Heavy penalty for trying to move to blocker's position
                    next_value = (
                        self.gamma * value_function[x, y]
                    )  # Stay in current position
                else:
                    # Bonus for reaching target
                    if new_pos == target_pos:
                        reward += 10.0

                    next_value = self.gamma * value_function[new_pos[0], new_pos[1]]

        return reward + next_value

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

    def _infer_target_door_color(self, obs=None):
        """Infer target door color from observations."""
        # Use target door color from observations if available
        if obs and "target_door_color" in obs:
            return obs["target_door_color"]

        # Fallback: use first available door color
        if obs and "door_positions" in obs:
            door_colors = list(obs["door_positions"].keys())
            if door_colors:
                return door_colors[0]

        # Final fallback
        return "red"

    def reset(self):
        """Reset agent state for new episode"""
        self.agent_pos = None
        self.grid = None
        self.current_target = None
        self.path = []
        self.collected_keys = set()
        self.strategy_phase = "collect_key"
        self.value_function = None
        self.policy = None
        self.converged = False
        self.target_door_color = None


class Level1ValueAchiever:
    """
    Level-1 Value-based Achiever Agent with rule-based deception strategies
    Updated for KeyDoor environment with automatic key pickup and door opening

    Strategy:
    1. Randomly-selected color except "self.target_door_color"
    2. Go to randomly-selected color key
    3. After randomly-selected key, go to the "real" target color key
    4. After collecting target color key, go to "open_door"
    
    Uses value iteration for optimal path planning with deceptive behavior
    """

    def __init__(
        self,
        observability="full",
        movement_cost=0.01,
        wall_penalty=2.0,
        gamma=0.99,
        temperature=0.1,
    ):
        self.observability = observability
        self.agent_pos = None
        self.grid = None
        self.current_target = None
        self.path = []
        self.target_door_color = None

        # Grid dimensions - will be set from observations
        self.width = None
        self.height = None

        # Track agent's collected keys
        self.collected_keys = set()

        # Strategy phases: "collect_decoy_key", "collect_target_key", "open_door"
        self.strategy_phase = "collect_decoy_key"
        self.decoy_key_color = None
        self.decoy_key_collected = False

        # Value iteration parameters
        self.movement_cost = movement_cost
        self.wall_penalty = wall_penalty
        self.gamma = gamma
        self.temperature = temperature

        # Value function and policy
        self.value_function = None
        self.policy = None
        self.converged = False

        # MiniGrid action mapping: 0=up, 1=right, 2=down, 3=left, 4=stay, 5=pickup, 6=toggle
        # Value iteration uses 4 movement actions: 0=up, 1=right, 2=down, 3=left
        # Grid coordinate system: (x, y) where x=column, y=row, positive y is down
        self.actions = [
            (0, -1),
            (1, 0),
            (0, 1),
            (-1, 0),
        ]  # dx, dy for up, right, down, left

    def update_observation(self, obs):
        """Update agent's understanding of the environment"""
        if obs is None:
            return

        # Update grid dimensions from observations
        if "grid_info" in obs and self.width is None:
            self.width = obs["grid_info"]["width"]
            self.height = obs["grid_info"]["height"]

        # Get current achiever position from observations
        new_pos = tuple(obs["achiever_pos"])

        if new_pos != self.agent_pos:
            self.agent_pos = new_pos

        # Get the grid from achiever's visual observation
        self.grid = obs["achiever"]

        # Update collected keys based on achiever's key inventory
        achiever_keys_array = obs["achiever_keys"]
        color_map = ["red", "green", "blue", "yellow"]
        self.collected_keys = set()
        for i, has_key in enumerate(achiever_keys_array):
            if has_key > 0 and i < len(color_map):
                self.collected_keys.add(color_map[i])

    def get_action(self, obs):
        """
        Get the next action for the agent using value iteration with deceptive strategy
        """
        self.update_observation(obs)

        # Infer target door color from observations
        if self.target_door_color is None:
            self.target_door_color = self._infer_target_door_color(obs)

        # Select decoy key color if not already selected
        if self.decoy_key_color is None:
            self._select_decoy_key_color(obs)

        # Phase 1: Collect decoy key first
        if self.strategy_phase == "collect_decoy_key":
            if self.decoy_key_color in self.collected_keys:
                self.decoy_key_collected = True
                self.strategy_phase = "collect_target_key"
            else:
                return self._collect_key(self.decoy_key_color, obs)

        # Phase 2: Collect target key
        elif self.strategy_phase == "collect_target_key":
            if self.target_door_color in self.collected_keys:
                self.strategy_phase = "open_door"
            else:
                return self._collect_key(self.target_door_color, obs)

        # Phase 3: Open target door
        elif self.strategy_phase == "open_door":
            return self._open_target_door(self.target_door_color, obs)

        # Better fallback: try random movement instead of staying stuck
        return np.random.choice([0, 1, 2, 3])  # random movement

    def _select_decoy_key_color(self, obs):
        """Select a decoy key color that is different from target door color"""
        all_colors = ["red", "green", "blue", "yellow"]
        available_colors = [color for color in all_colors if color != self.target_door_color]
        
        if available_colors:
            self.decoy_key_color = np.random.choice(available_colors)
        else:
            # Fallback: if somehow no other colors available, use a random color
            self.decoy_key_color = np.random.choice(all_colors)

    def _collect_key(self, key_color, obs=None):
        """Strategy to collect a specific key using value iteration"""
        # Find target key position
        target_key_pos = self._find_object_position(Key, key_color, obs)
        if target_key_pos is None:
            return 4  # Stay if key not found

        # Check if we're already at the key position
        if self.agent_pos == target_key_pos:
            return 4  # Stay - key pickup is automatic when agent steps on it

        # Use value iteration to navigate to key (key will be picked up automatically)
        return self._navigate_with_value_iteration(target_key_pos, obs)

    def _open_target_door(self, target_door_color, obs=None):
        """Strategy to open the target door using value iteration"""
        # Find target door position
        target_door_pos = self._find_object_position(Door, target_door_color, obs)
        if target_door_pos is None:
            return 4  # Stay if door not found

        # Check if we're at the door position
        if self.agent_pos == target_door_pos:
            door = self.grid.get(*target_door_pos)
            if isinstance(door, Door) and door.is_open:
                # Already opened, episode should end
                return 4  # Stay on the opened door
            elif (
                isinstance(door, Door)
                and door.is_locked
                and target_door_color in self.collected_keys
            ):
                # Door will be opened automatically when agent steps on it
                return 4  # Stay - door opening is automatic

        # Navigate to door (door will open automatically when agent steps on it)
        return self._navigate_with_value_iteration(target_door_pos, obs)

    def _find_object_position(self, obj_type, color, obs=None):
        """Find position of specific object type and color"""
        if obs is None:
            return None

        if obj_type.__name__ == "Key":
            # Get key positions from observations
            if "key_positions" in obs:
                return obs["key_positions"].get(color, None)
        elif obj_type.__name__ == "Door":
            # Get door positions from observations
            if "door_positions" in obs:
                return obs["door_positions"].get(color, None)

        return None

    def _navigate_with_value_iteration(self, target_pos, obs=None):
        """Navigate using value iteration and convert to MiniGrid actions"""
        # Run value iteration to get optimal action
        optimal_action = self._plan_value_iteration(target_pos, obs)

        if optimal_action is None:
            return 4  # Stay if no action found

        # Convert value iteration action to MiniGrid action
        return self._convert_to_minigrid_action(optimal_action)

    def _plan_value_iteration(
        self, target_pos, obs=None, max_iterations=100, convergence_threshold=0.01
    ):
        """
        Vectorized value iteration to compute optimal action for reaching target
        """
        # Get grid size from instance dimensions
        width = self.width if self.width is not None else 9  # fallback
        height = self.height if self.height is not None else 9  # fallback
        n_actions = 4

        # Get blocker position from observations
        blocker_pos = None
        if obs is not None and "blocker_pos" in obs:
            blocker_pos = tuple(obs["blocker_pos"])

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
        # that matches the current _is_walkable implementation
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

                # Apply blocker position penalty
                if blocker_pos is not None:
                    blocker_conflict = (
                        valid_moves
                        & (valid_next_x == blocker_pos[0])
                        & (valid_next_y == blocker_pos[1])
                    )
                    rewards = np.where(
                        blocker_conflict, rewards - self.wall_penalty, rewards
                    )
                    next_values = np.where(
                        blocker_conflict,
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
                blocker_pos,
            )
            q_values.append(q_val)

        # Choose action with softmax policy
        if self.temperature > 0:
            q_values = np.array(q_values)
            q_values_clipped = np.clip(q_values, -100, 100)
            exp_q = np.exp(q_values_clipped / self.temperature)
            action_probs = exp_q / np.sum(exp_q)

            # Sample action stochastically
            action = np.random.choice(n_actions, p=action_probs)
        else:
            # Deterministic policy
            action = np.argmax(q_values)

        return action

    def _evaluate_action(
        self, pos, action, value_function, target_pos, width, height, blocker_pos=None
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
                # Check if new position conflicts with blocker position
                if blocker_pos is not None and new_pos == blocker_pos:
                    reward -= (
                        self.wall_penalty
                    )  # Heavy penalty for trying to move to blocker's position
                    next_value = (
                        self.gamma * value_function[x, y]
                    )  # Stay in current position
                else:
                    # Bonus for reaching target
                    if new_pos == target_pos:
                        reward += 10.0

                    next_value = self.gamma * value_function[new_pos[0], new_pos[1]]

        return reward + next_value

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

    def _infer_target_door_color(self, obs=None):
        """Infer target door color from observations."""
        # Use target door color from observations if available
        if obs and "target_door_color" in obs:
            return obs["target_door_color"]

        # Fallback: use first available door color
        if obs and "door_positions" in obs:
            door_colors = list(obs["door_positions"].keys())
            if door_colors:
                return door_colors[0]

        # Final fallback
        return "red"

    def reset(self):
        """Reset agent state for new episode"""
        self.agent_pos = None
        self.grid = None
        self.current_target = None
        self.path = []
        self.collected_keys = set()
        self.strategy_phase = "collect_decoy_key"
        self.decoy_key_color = None
        self.decoy_key_collected = False
        self.value_function = None
        self.policy = None
        self.converged = False
        self.target_door_color = None


class RandomAgent:
    """A random agent that explores the AchieverBlocker environment.

    Achiever uses 7-action space: up, right, down, left, stay, pickup, toggle
    (no "done" action - that's only for blockers)
    """

    def __init__(self, action_space=None, movement_prob=0.9):
        # Get action space from config if not provided
        if action_space is None:
            try:
                from config import Config

                config = Config()
                action_space = config.model_config["achiever_action_space"]
            except:
                action_space = 7  # Fallback default
        self.action_space = action_space
        self.movement_prob = movement_prob

    def get_action(self, obs):
        """Return a random action with bias towards movement (automatic pickup/door opening)."""
        rand = np.random.random()

        if rand < self.movement_prob:
            # Focus on movement actions: up, right, down, left, stay
            return np.random.choice(
                [0, 1, 2, 3, 4]
            )  # 0=up, 1=right, 2=down, 3=left, 4=stay
        else:
            # Rarely use pickup or toggle actions (mostly unnecessary due to automatic behavior)
            # But keep them for compatibility
            return np.random.choice([5, 6])  # 5=pickup, 6=toggle

    def analyze_feedback(self, reward, done):
        """Random agent doesn't learn from feedback."""
        pass

    def reset(self):
        """Reset agent state for new episode"""
        pass
