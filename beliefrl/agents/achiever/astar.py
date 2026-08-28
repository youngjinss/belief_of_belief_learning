import heapq
import os

import numpy as np

from beliefrl.env.minigrid import Door, Key, Wall
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
        # Handle both single-agent and multi-agent environments
        if "achiever_pos" in obs:
            new_pos = tuple(obs["achiever_pos"])
        elif "agent_pos" in obs:
            new_pos = tuple(obs["agent_pos"])
        else:
            # Fallback - no position update
            return

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
            if os.getenv('DEBUG_MODE'):
                print(f"DEBUG _infer_target_door_color: obs target_door_color = {obs['target_door_color']}")
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
