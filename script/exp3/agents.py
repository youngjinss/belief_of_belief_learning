import numpy as np
import heapq
from collections import deque
import sys
import os

# Add the lib directory to the path
lib_path = os.path.join(os.path.dirname(__file__), '..', '..', 'lib')
sys.path.insert(0, lib_path)

# Add the env directory to the path
env_path = os.path.join(lib_path, 'env')
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
            (1, 0),   # down (row + 1)
            (0, -1),  # left (col - 1)
            (0, 1),   # right (col + 1)
        ]
        
        # Action mapping for MiniGrid: 0=left turn, 1=right turn, 2=forward, 3=pickup, 4=drop, 5=toggle
        # Agent has direction: 0=right, 1=down, 2=left, 3=up
        self.action_mapping = {
            (1, 0): 0,   # right (direction 0)
            (0, 1): 1,   # down (direction 1)  
            (-1, 0): 2,  # left (direction 2)
            (0, -1): 3,  # up (direction 3)
        }
        
        # Direction vectors for MiniGrid agent directions
        self.direction_vectors = [
            (1, 0),   # 0: right
            (0, 1),   # 1: down
            (-1, 0),  # 2: left
            (0, -1),  # 3: up
        ]
    
    def update_observation(self, obs):
        """Update agent's understanding of the environment"""
        # Get current agent position from environment
        new_pos = tuple(self.env.agent_pos)
        if new_pos != self.agent_pos:
            print(f"DEBUG: Agent position updated from {self.agent_pos} to {new_pos}")
            # If we have a path and we moved to the expected next position, advance the path
            if self.path and len(self.path) >= 2 and self.path[1] == new_pos:
                print(f"DEBUG: Advanced path by removing {self.path[0]}")
                self.path.pop(0)
            else:
                # Position changed unexpectedly, clear path
                self.path = []
            self.agent_pos = new_pos
        else:
            # Position didn't change - could be turning or stuck
            if self.agent_pos is not None:
                print(f"DEBUG: Agent position unchanged at {self.agent_pos}")
        
        # Get the grid from environment
        self.grid = self.env.grid
        
        # Update collected keys based on agent's inventory
        self.collected_keys = set(self.env.agent_keys)
    
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
        
        return 4  # Stay if no action determined
    
    def _collect_target_key(self, target_key_color):
        """Strategy to collect the target key"""
        # Find target key position
        target_key_pos = self._find_object_position(Key, target_key_color)
        if target_key_pos is None:
            print(f"DEBUG: Could not find {target_key_color} key in grid")
            return 4  # Stay if key not found
        
        print(f"DEBUG: Found {target_key_color} key at {target_key_pos}, agent at {self.agent_pos}")
        
        # Check if we're already at the key position
        if self.agent_pos == target_key_pos:
            print(f"DEBUG: Agent is at key position {target_key_pos}, key will be picked up automatically")
            return 4  # Stay - key pickup is automatic
        
        # Navigate to the key position (key will be picked up automatically when agent steps on it)
        print(f"DEBUG: Agent needs to reach key position {target_key_pos}")
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
                print(f"DEBUG: Agent is at opened door position {target_door_pos}")
                return 4  # Stay on the opened door
            elif isinstance(door, Door) and door.is_locked and target_door_color in self.collected_keys:
                # Door will be opened automatically when agent steps on it
                print(f"DEBUG: Agent is at locked door position {target_door_pos}, door will open automatically")
                return 4  # Stay - door opening is automatic
        
        # Navigate to door (door will open automatically when agent steps on it)
        print(f"DEBUG: Agent needs to reach door position {target_door_pos}")
        return self._navigate_to_position(target_door_pos)
    
    def _find_object_position(self, obj_type, color):
        """Find position of specific object type and color"""
        for i in range(self.grid.width):
            for j in range(self.grid.height):
                obj = self.grid.get(i, j)
                if isinstance(obj, obj_type) and obj.color == color:
                    return (i, j)
        return None
    
    def _navigate_to_position(self, target_pos):
        """Navigate to target position using A* pathfinding and MiniGrid turn-based movement"""
        # Always recalculate if path is empty or if we're not on the expected path
        if not self.path or len(self.path) < 2 or (len(self.path) >= 1 and self.path[0] != self.agent_pos):
            # Calculate new path if needed
            self.path = self._astar_pathfind(self.agent_pos, target_pos)
            print(f"DEBUG: Calculated path: {self.path}")
        
        if len(self.path) >= 2:
            # Get next step in path
            next_pos = self.path[1]  # path[0] should be current position
            
            # Verify current position matches path start
            if self.path[0] != self.agent_pos:
                print(f"DEBUG: Position mismatch. Agent at {self.agent_pos}, path starts at {self.path[0]}. Recalculating.")
                self.path = self._astar_pathfind(self.agent_pos, target_pos)
                if len(self.path) < 2:
                    return 4  # Drop if no path found
                next_pos = self.path[1]
            
            # Calculate movement direction needed
            move_dir = (next_pos[0] - self.agent_pos[0], next_pos[1] - self.agent_pos[1])
            
            # Get current agent direction from environment
            agent_dir = self.env.agent_dir
            current_dir_vec = self.direction_vectors[agent_dir]
            
            print(f"DEBUG: Moving from {self.agent_pos} to {next_pos}, move_dir: {move_dir}, agent_dir: {agent_dir}, current_dir_vec: {current_dir_vec}")
            
            # Check if we're already facing the right direction
            if move_dir == current_dir_vec:
                # We're facing the right direction, move forward
                # Check if the target position is actually walkable
                if self._is_walkable(next_pos):
                    print(f"DEBUG: Already facing correct direction, moving forward to {next_pos}")
                    # Debug: check what's at the next position
                    obj_at_next = self.grid.get(*next_pos)
                    print(f"DEBUG: Object at target position {next_pos}: {obj_at_next}")
                    # Check if there's a wall at the next position in the direction we're going
                    wall_check_pos = (next_pos[0] + move_dir[0], next_pos[1] + move_dir[1])
                    if 0 <= wall_check_pos[0] < self.grid.width and 0 <= wall_check_pos[1] < self.grid.height:
                        wall_obj = self.grid.get(*wall_check_pos)
                        print(f"DEBUG: Object at position beyond target {wall_check_pos}: {wall_obj}")
                    return 2  # Forward
                else:
                    print(f"DEBUG: Target position {next_pos} is not walkable, clearing path")
                    self.path = []
                    return 4  # Drop
            else:
                # We need to turn first
                # Find which direction we need to face
                target_dir = None
                for i, dir_vec in enumerate(self.direction_vectors):
                    if dir_vec == move_dir:
                        target_dir = i
                        break
                
                if target_dir is not None:
                    # Calculate turn direction (left or right)
                    # MiniGrid uses 0=left turn, 1=right turn
                    turn_diff = (target_dir - agent_dir) % 4
                    if turn_diff == 1:  # Turn right
                        print(f"DEBUG: Need to turn right to face direction {target_dir}")
                        return 1
                    elif turn_diff == 3:  # Turn left (3 steps right = 1 step left)
                        print(f"DEBUG: Need to turn left to face direction {target_dir}")
                        return 0
                    elif turn_diff == 2:  # Turn around (choose right)
                        print(f"DEBUG: Need to turn around to face direction {target_dir}")
                        return 1
                else:
                    print(f"DEBUG: Invalid move direction {move_dir}")
                    self.path = []
                    return 4  # Drop
        
        print(f"DEBUG: No path found or path too short: {self.path}")
        return 4  # Drop if no path found
    
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
        while open_list:
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
                    current_node.position[1] + move[1]
                )
                
                # Check if position is within grid bounds
                if (node_pos[0] < 0 or node_pos[0] >= self.grid.width or
                    node_pos[1] < 0 or node_pos[1] >= self.grid.height):
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
                if any(closed_child.position == child.position for closed_child in closed_list):
                    continue
                
                # Calculate g, h, and f values
                child.g = current_node.g + 1
                child.h = self._heuristic(child.position, end_node.position)
                child.f = child.g + child.h
                
                # Skip if child is already in open list with better g value
                if any(open_node.position == child.position and child.g > open_node.g 
                       for open_node in open_list):
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
        
        # Doors are walkable if they're open or we have the key
        if isinstance(obj, Door):
            if obj.is_open:
                return True
            # Check if we have the key for locked doors
            if obj.is_locked and obj.color in self.collected_keys:
                return True
        
        # Walls and other objects are not walkable
        print(f"DEBUG: Position {pos} is not walkable, contains: {obj}")
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
    Adapted for KeyDoor environment
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
        
        # Action mapping for MiniGrid: 0=turn_left, 1=turn_right, 2=forward
        # We'll use a simplified 4-action space for value iteration: 0=up, 1=right, 2=down, 3=left
        self.actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left
        
        # Direction vectors for MiniGrid agent directions
        self.direction_vectors = [
            (1, 0),   # 0: right
            (0, 1),   # 1: down
            (-1, 0),  # 2: left
            (0, -1),  # 3: up
        ]
    
    def update_observation(self, obs):
        """Update agent's understanding of the environment"""
        # Get current agent position from environment
        new_pos = tuple(self.env.agent_pos)
        if new_pos != self.agent_pos:
            self.agent_pos = new_pos
        
        # Get the grid from environment
        self.grid = self.env.grid
        
        # Update collected keys based on agent's inventory
        self.collected_keys = set(self.env.agent_keys)
    
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
        
        return 4  # Stay if no action determined
    
    def _collect_target_key(self, target_key_color):
        """Strategy to collect the target key using value iteration"""
        # Find target key position
        target_key_pos = self._find_object_position(Key, target_key_color)
        if target_key_pos is None:
            return 4  # Stay if key not found
        
        # Use value iteration to navigate to key
        return self._navigate_with_value_iteration(target_key_pos)
    
    def _open_target_door(self, target_door_color):
        """Strategy to open the target door using value iteration"""
        # Find target door position
        target_door_pos = self._find_object_position(Door, target_door_color)
        if target_door_pos is None:
            return 4  # Stay if door not found
        
        # Use value iteration to navigate to door
        return self._navigate_with_value_iteration(target_door_pos)
    
    def _find_object_position(self, obj_type, color):
        """Find position of specific object type and color"""
        for i in range(self.grid.width):
            for j in range(self.grid.height):
                obj = self.grid.get(i, j)
                if isinstance(obj, obj_type) and obj.color == color:
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
    
    def _plan_value_iteration(self, target_pos, max_iterations=100, convergence_threshold=0.01):
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
            for i in range(width):
                for j in range(height):
                    if (i, j) == target_pos:
                        continue  # Keep target value high
                    
                    if not self._is_walkable((i, j)):
                        value_function[i, j] = -self.wall_penalty
                        continue
                    
                    # Compute Q-values for each action
                    q_values = []
                    for action in range(n_actions):
                        q_val = self._evaluate_action((i, j), action, old_values, target_pos)
                        q_values.append(q_val)
                    
                    # Update value function
                    value_function[i, j] = max(q_values)
            
            # Check convergence
            if np.max(np.abs(value_function - old_values)) < convergence_threshold:
                break
        
        # Get optimal action for current position
        if not self._is_walkable(self.agent_pos):
            return None
        
        current_pos = self.agent_pos
        q_values = []
        for action in range(n_actions):
            q_val = self._evaluate_action(current_pos, action, value_function, target_pos)
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
        i, j = pos
        delta = self.actions[action]
        new_pos = (i + delta[0], j + delta[1])
        
        # Base movement cost
        reward = -self.movement_cost
        
        # Check bounds
        if (new_pos[0] < 0 or new_pos[0] >= self.grid.width or
            new_pos[1] < 0 or new_pos[1] >= self.grid.height):
            reward -= self.wall_penalty
            next_value = self.gamma * value_function[i, j]
        else:
            # Check if position is walkable
            if not self._is_walkable(new_pos):
                reward -= self.wall_penalty
                next_value = self.gamma * value_function[i, j]
            else:
                # Bonus for reaching target
                if new_pos == target_pos:
                    reward += 10.0
                
                next_value = self.gamma * value_function[new_pos[0], new_pos[1]]
        
        return reward + next_value
    
    def _convert_to_minigrid_action(self, value_action):
        """Convert value iteration action to MiniGrid turn-based action"""
        if value_action is None:
            return 4  # Stay
        
        # Get movement direction from value action
        move_dir = self.actions[value_action]
        
        # Get current agent direction from environment
        agent_dir = self.env.agent_dir
        current_dir_vec = self.direction_vectors[agent_dir]
        
        # Check if we're already facing the right direction
        if move_dir == current_dir_vec:
            return 2  # Forward
        else:
            # We need to turn first
            # Find which direction we need to face
            target_dir = None
            for i, dir_vec in enumerate(self.direction_vectors):
                if dir_vec == move_dir:
                    target_dir = i
                    break
            
            if target_dir is not None:
                # Calculate turn direction (left or right)
                turn_diff = (target_dir - agent_dir) % 4
                if turn_diff == 1:  # Turn right
                    return 1
                elif turn_diff == 3:  # Turn left
                    return 0
                elif turn_diff == 2:  # Turn around (choose right)
                    return 1
        
        return 4  # Stay as fallback
    
    def _is_walkable(self, pos):
        """Check if position is walkable"""
        obj = self.grid.get(*pos)
        
        # Empty cells are walkable
        if obj is None:
            return True
        
        # Keys are walkable (can step on them)
        if isinstance(obj, Key):
            return True
        
        # Doors are walkable if they're open or we have the key
        if isinstance(obj, Door):
            if obj.is_open:
                return True
            # Check if we have the key for locked doors
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
    """A random agent that explores the KeyDoor environment."""
    
    def __init__(self, action_space, movement_prob=0.8):
        self.action_space = action_space
        self.movement_prob = movement_prob
    
    def get_action(self, obs):
        """Return a random action with bias towards movement."""
        rand = np.random.random()
        
        if rand < self.movement_prob:
            # Focus on movement actions: turn_left, turn_right, forward
            return np.random.choice([0, 1, 2])  # 0=turn_left, 1=turn_right, 2=forward
        else:
            # Toggle action (for opening doors)
            return 5  # toggle
    
    def analyze_feedback(self, reward, done):
        """Random agent doesn't learn from feedback."""
        pass
    
    def reset(self):
        """Reset agent state for new episode"""
        pass