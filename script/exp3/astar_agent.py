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
        
        # Action mapping: KeyDoor uses custom actions
        # 0: up, 1: down, 2: left, 3: right, 4: stay, 5: pickup
        self.action_mapping = {
            (-1, 0): 0,  # up
            (1, 0): 1,   # down
            (0, -1): 2,  # left
            (0, 1): 3,   # right
        }
    
    def update_observation(self, obs):
        """Update agent's understanding of the environment"""
        # Get current agent position from environment
        new_pos = tuple(self.env.agent_pos)
        if new_pos != self.agent_pos:
            print(f"DEBUG: Agent position updated from {self.agent_pos} to {new_pos}")
            self.agent_pos = new_pos
        
        # Get the grid from environment
        self.grid = self.env.grid
        
        # Update collected keys based on agent's inventory
        self.collected_keys = set(self.env.agent_keys)
    
    def get_action(self, obs):
        """
        Get the next action for the agent
        """
        self.update_observation(obs)
        
        # Simple test: try to move down first
        if self.agent_pos == (1, 7):
            print("DEBUG: Testing simple movement - trying to move down")
            # Check what's at the down position
            down_pos = (2, 7)
            if down_pos[0] < self.grid.width:
                down_obj = self.grid.get(*down_pos)
                print(f"DEBUG: Object at down position {down_pos}: {down_obj}")
            else:
                print(f"DEBUG: Down position {down_pos} is out of bounds")
            return 1  # down
        
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
        # Check if we're on a key position
        current_obj = self.grid.get(*self.agent_pos)
        if isinstance(current_obj, Key) and current_obj.color == target_key_color:
            print(f"DEBUG: Found {target_key_color} key at agent position, picking up")
            return 5  # Pickup action
        
        # Find target key position
        target_key_pos = self._find_object_position(Key, target_key_color)
        if target_key_pos is None:
            print(f"DEBUG: Could not find {target_key_color} key in grid")
            return 4  # Stay if key not found
        
        print(f"DEBUG: Found {target_key_color} key at {target_key_pos}, agent at {self.agent_pos}")
        
        # Navigate to key
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
            if isinstance(door, Door) and door.is_locked and target_door_color in self.collected_keys:
                # Agent will automatically unlock and open the door by stepping on it
                return 4  # Stay - environment will handle door opening
        
        # Navigate to door
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
        """Navigate to target position using A* pathfinding"""
        if not self.path or len(self.path) < 2:
            # Calculate new path if needed
            self.path = self._astar_pathfind(self.agent_pos, target_pos)
            print(f"DEBUG: Calculated path: {self.path}")
        
        if len(self.path) >= 2:
            # Get next step in path
            next_pos = self.path[1]  # path[0] is current position
            
            # Calculate movement direction
            move_dir = (next_pos[0] - self.agent_pos[0], next_pos[1] - self.agent_pos[1])
            
            # Only move one step at a time
            if abs(move_dir[0]) <= 1 and abs(move_dir[1]) <= 1 and (move_dir[0] != 0 or move_dir[1] != 0):
                # Remove current position from path
                self.path.pop(0)
                
                print(f"DEBUG: Moving from {self.agent_pos} to {next_pos}, direction: {move_dir}")
                
                # Return corresponding action
                if move_dir in self.action_mapping:
                    return self.action_mapping[move_dir]
            else:
                # Invalid move, recalculate path
                print(f"DEBUG: Invalid move direction {move_dir}, recalculating path")
                self.path = []
        
        print(f"DEBUG: No path found or path too short: {self.path}")
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