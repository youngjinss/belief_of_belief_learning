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

    def __init__(self, env, observability="full"):
        self.env = env
        self.observability = observability
        self.agent_pos = None
        self.grid = None
        self.current_target = None
        self.path = []

        # Track agent's collected keys
        self.collected_keys = set()

        # Strategy: first collect target key, then go to target door
        self.strategy_phase = "collect_key"  # "collect_key" or "open_door"

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
        # Get current achiever position from environment
        if hasattr(self.env, 'achiever_pos'):
            new_pos = tuple(self.env.achiever_pos)
        else:
            # Fallback to single agent mode
            new_pos = tuple(self.env.agent_pos) if hasattr(self.env, 'agent_pos') else self.agent_pos
            
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

        # Get the grid from environment
        self.grid = self.env.grid

        # Update collected keys based on achiever's inventory
        if hasattr(self.env, 'achiever_keys'):
            self.collected_keys = set(self.env.achiever_keys)
        else:
            # Fallback to single agent mode
            self.collected_keys = set(self.env.agent_keys) if hasattr(self.env, 'agent_keys') else set()

    def get_action(self, obs):
        """
        Get the next action for the agent
        """
        self.update_observation(obs)

        # Determine strategy based on current state
        target_key_color = self.env.target_door_color

        if self.strategy_phase == "collect_key":
            # Check if we already have the target key
            if target_key_color in self.collected_keys:
                self.strategy_phase = "open_door"
                self.path = []  # Clear path to recalculate
            else:
                # Find and collect the target key
                return self._collect_target_key(target_key_color)

        elif self.strategy_phase == "open_door":
            # Go to target door and open it
            return self._open_target_door(target_key_color)

        # Better fallback: try random movement instead of staying stuck
        return np.random.choice([0, 1, 2, 3])  # random movement

    def _collect_target_key(self, target_key_color):
        """Strategy to collect the target key"""
        # Find target key position
        target_key_pos = self._find_object_position(Key, target_key_color)
        if target_key_pos is None:
            return 4  # Stay if key not found

        # Check if we're already at the key position
        if self.agent_pos == target_key_pos:
            return 4  # Stay - key pickup is automatic

        # Navigate to the key position (key will be picked up automatically when agent steps on it)
        return self._navigate_to_position(target_key_pos)

    def _open_target_door(self, target_door_color):
        """Strategy to open the target door"""
        # Find target door position
        target_door_pos = self._find_object_position(Door, target_door_color)
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

    def _find_object_position(self, obj_type, color):
        """Find position of specific object type and color"""
        for i in range(self.grid.width):
            for j in range(self.grid.height):
                obj = self.grid.get(i, j)
                if obj is not None:
                    # Check both isinstance and class name for compatibility
                    if isinstance(obj, obj_type) and obj.color == color:
                        return (i, j)
                    elif (
                        hasattr(obj, "color")
                        and obj.color == color
                        and obj.__class__.__name__ == obj_type.__name__
                    ):
                        return (i, j)
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
                if (
                    node_pos[0] < 0
                    or node_pos[0] >= self.grid.width
                    or node_pos[1] < 0
                    or node_pos[1] >= self.grid.height
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
        """Check if position is walkable"""
        obj = self.grid.get(*pos)

        # Empty cells are walkable
        if obj is None:
            return True

        # Keys are walkable (can step on them)
        if isinstance(obj, Key):
            return True

        # Check using class name for compatibility
        if hasattr(obj, "__class__") and obj.__class__.__name__ == "Key":
            return True

        # Doors are walkable if they're open or we have the key
        if isinstance(obj, Door):
            if obj.is_open:
                return True
            # Check if we have the key for locked doors
            if obj.is_locked and obj.color in self.collected_keys:
                return True

        # Check using class name for compatibility
        if hasattr(obj, "__class__") and obj.__class__.__name__ == "Door":
            if obj.is_open:
                return True
            # Check if we have the key for locked doors
            if obj.is_locked and obj.color in self.collected_keys:
                return True

        # Walls and other objects are not walkable
        return False

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


class ValueAgent:
    """
    Value-based agent with stochastic action selection using value iteration
    Updated for KeyDoor environment with automatic key pickup and door opening

    Key features:
    - Uses value iteration for optimal path planning
    - Accounts for automatic key pickup when stepping on keys
    - Accounts for automatic door opening when stepping on doors with correct key
    - Compatible with 7-action MiniGrid action space
    - Stochastic policy with temperature-based action selection
    """

    def __init__(
        self,
        env,
        observability="full",
        movement_cost=0.01,
        wall_penalty=2.0,
        gamma=0.99,
        temperature=0.1,
    ):
        self.env = env
        self.observability = observability
        self.agent_pos = None
        self.grid = None
        self.current_target = None
        self.path = []

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
        # Get current achiever position from environment
        if hasattr(self.env, 'achiever_pos'):
            new_pos = tuple(self.env.achiever_pos)
        else:
            # Fallback to single agent mode
            new_pos = tuple(self.env.agent_pos) if hasattr(self.env, 'agent_pos') else self.agent_pos
            
        if new_pos != self.agent_pos:
            self.agent_pos = new_pos

        # Get the grid from environment
        self.grid = self.env.grid

        # Update collected keys based on achiever's inventory
        if hasattr(self.env, 'achiever_keys'):
            self.collected_keys = set(self.env.achiever_keys)
        else:
            # Fallback to single agent mode
            self.collected_keys = set(self.env.agent_keys) if hasattr(self.env, 'agent_keys') else set()

    def get_action(self, obs):
        """
        Get the next action for the agent using value iteration
        """
        self.update_observation(obs)

        # Determine strategy based on current state
        target_key_color = self.env.target_door_color

        if self.strategy_phase == "collect_key":
            # Check if we already have the target key
            if target_key_color in self.collected_keys:
                self.strategy_phase = "open_door"
            else:
                # Find and collect the target key
                return self._collect_target_key(target_key_color)

        elif self.strategy_phase == "open_door":
            # Go to target door and open it
            return self._open_target_door(target_key_color)

        # Better fallback: try random movement instead of staying stuck
        return np.random.choice([0, 1, 2, 3])  # random movement

    def _collect_target_key(self, target_key_color):
        """Strategy to collect the target key using value iteration"""
        # Find target key position
        target_key_pos = self._find_object_position(Key, target_key_color)
        if target_key_pos is None:
            return 4  # Stay if key not found

        # Check if we're already at the key position
        if self.agent_pos == target_key_pos:
            return 4  # Stay - key pickup is automatic when agent steps on it

        # Use value iteration to navigate to key (key will be picked up automatically)
        return self._navigate_with_value_iteration(target_key_pos)

    def _open_target_door(self, target_door_color):
        """Strategy to open the target door using value iteration"""
        # Find target door position
        target_door_pos = self._find_object_position(Door, target_door_color)
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
        return self._navigate_with_value_iteration(target_door_pos)

    def _find_object_position(self, obj_type, color):
        """Find position of specific object type and color"""
        for i in range(self.grid.width):
            for j in range(self.grid.height):
                obj = self.grid.get(i, j)
                if obj is not None:
                    # Check both isinstance and class name for compatibility
                    if isinstance(obj, obj_type) and obj.color == color:
                        return (i, j)
                    elif (
                        hasattr(obj, "color")
                        and obj.color == color
                        and obj.__class__.__name__ == obj_type.__name__
                    ):
                        return (i, j)
        return None

    def _navigate_with_value_iteration(self, target_pos):
        """Navigate using value iteration and convert to MiniGrid actions"""
        # Run value iteration to get optimal action
        optimal_action = self._plan_value_iteration(target_pos)

        if optimal_action is None:
            return 4  # Stay if no action found

        # Convert value iteration action to MiniGrid action
        return self._convert_to_minigrid_action(optimal_action)

    def _plan_value_iteration(
        self, target_pos, max_iterations=100, convergence_threshold=0.01
    ):
        """
        Run value iteration to compute optimal action for reaching target
        """
        width, height = self.grid.width, self.grid.height
        n_actions = 4

        # Initialize value function
        value_function = np.zeros((width, height))

        # Set high reward for target position
        value_function[target_pos[0], target_pos[1]] = 10.0

        # Run value iteration
        for iteration in range(max_iterations):
            old_values = value_function.copy()

            # Value iteration update
            for x in range(width):
                for y in range(height):
                    if (x, y) == target_pos:
                        continue  # Keep target value high

                    if not self._is_walkable((x, y)):
                        value_function[x, y] = -self.wall_penalty
                        continue

                    # Compute Q-values for each action
                    q_values = []
                    for action in range(n_actions):
                        q_val = self._evaluate_action(
                            (x, y), action, old_values, target_pos
                        )
                        q_values.append(q_val)

                    # Update value function
                    value_function[x, y] = max(q_values)

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
                current_pos, action, value_function, target_pos
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

    def _evaluate_action(self, pos, action, value_function, target_pos):
        """Evaluate expected value of taking action from position"""
        x, y = pos  # Grid coordinates (x=column, y=row)
        dx, dy = self.actions[action]
        new_pos = (x + dx, y + dy)

        # Base movement cost
        reward = -self.movement_cost

        # Check bounds
        if (
            new_pos[0] < 0
            or new_pos[0] >= self.grid.width
            or new_pos[1] < 0
            or new_pos[1] >= self.grid.height
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
        """Check if position is walkable (matches KeyDoor environment's can_overlap logic)"""
        obj = self.grid.get(*pos)

        # Empty cells are walkable
        if obj is None:
            return True

        # Keys are walkable (can step on them - automatic pickup)
        if isinstance(obj, Key):
            return True

        # Check using class name for compatibility
        if hasattr(obj, "__class__") and obj.__class__.__name__ == "Key":
            return True

        # Doors are walkable if they're open or we have the key
        if isinstance(obj, Door):
            if obj.is_open:
                return True
            # Check if we have the key for locked doors (automatic opening)
            if obj.is_locked and obj.color in self.collected_keys:
                return True

        # Check using class name for compatibility
        if hasattr(obj, "__class__") and obj.__class__.__name__ == "Door":
            if obj.is_open:
                return True
            # Check if we have the key for locked doors (automatic opening)
            if obj.is_locked and obj.color in self.collected_keys:
                return True

        # Walls and other objects are not walkable
        return False

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


class RandomAgent:
    """A random agent that explores the AchieverBlocker environment.
    
    Achiever uses 7-action space: up, right, down, left, stay, pickup, toggle
    (no "done" action - that's only for blockers)
    """

    def __init__(self, action_space=7, movement_prob=0.9):
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
