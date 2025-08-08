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
from utils import set_seed

# Add current directory for config import
sys.path.append(os.path.dirname(__file__))
from config import Config
from value_agent import BaseValueAgent

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


class Level0ValueAchiever(BaseValueAgent):
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
        conflict_penalty=2.0,
        consumption_penalty=1.0,
        gamma=0.99,
        temperature=0.1,
        q_value_clip=100,
    ):
        # Initialize base class
        super().__init__(
            observability=observability,
            movement_cost=movement_cost,
            wall_penalty=wall_penalty,
            conflict_penalty=conflict_penalty,
            consumption_penalty=consumption_penalty,
            gamma=gamma,
            temperature=temperature,
            q_value_clip=q_value_clip,
            role="achiever",
        )

        # Achiever-specific attributes
        self.current_target = None
        self.path = []
        self.target_door_color = None
        self.collected_keys = set()
        self.strategy_phase = "collect_key"  # "collect_key" or "open_door"

    def _update_agent_position(self, obs):
        """Update achiever position from observations"""
        # Handle both single-agent and multi-agent environments
        if "achiever_pos" in obs:
            new_pos = tuple(obs["achiever_pos"])
        elif "agent_pos" in obs:
            new_pos = tuple(obs["agent_pos"])
        else:
            # Fallback - no position update
            return
        
        if new_pos != self.agent_pos:
            self.agent_pos = new_pos

    def _update_grid_reference(self, obs):
        """Update grid reference from observations"""
        self.grid = obs["achiever"]

    def _get_opponent_position(self, obs):
        """Get blocker position for conflict penalty"""
        if obs and "blocker_pos" in obs and obs["blocker_pos"] is not None:
            return tuple(obs["blocker_pos"])
        return None

    def update_observation(self, obs):
        """Update agent's understanding of the environment"""
        if obs is None:
            return

        # Call base class update
        super().update_observation(obs)

        # Update collected keys based on achiever's key inventory
        achiever_keys_array = obs["achiever_keys"]
        color_map = ["red", "green", "blue", "yellow"]
        self.collected_keys = set()
        for i, has_key in enumerate(achiever_keys_array):
            if has_key > 0 and i < len(color_map):
                self.collected_keys.add(color_map[i])

    def get_action(self, obs):
        """
        Level0ValueAchiever with clockwise exploration strategy:
        
        Uses base class act() method for proper strategy coordination and clockwise exploration
        """
        # Infer target door color from observations (only once)
        if self.target_door_color is None:
            self.target_door_color = self._infer_target_door_color(obs)
            if self.target_door_color:
                # Set target door color in base class for consumption penalty
                self.set_target_door_color(self.target_door_color)
                if os.getenv('DEBUG_MODE'):
                    print(f"DEBUG Level0ValueAchiever: FIXED target door color to {self.target_door_color}")
                
        # Set preferred door color for base class target finding
        self._preferred_door_color = self.target_door_color
        
        # Use base class act method for strategy coordination and clockwise exploration
        return self.act(obs)

    def _collect_target_key_with_exploration(self, target_key_color, obs=None):
        """Collect target key using exploration strategy"""
        if target_key_color is None:
            # No target identified yet, continue exploring
            return self._explore_action()
            
        # Check memory first, then current observation
        target_key_pos = None
        if target_key_color in self.memory['key_positions']:
            target_key_pos = self.memory['key_positions'][target_key_color]
        else:
            target_key_pos = self._find_object_position(Key, target_key_color, obs)
            
        if target_key_pos is None:
            # Target key not found, continue exploration
            return self._explore_action()
            
        # Target key found, navigate to it using value iteration
        if self.agent_pos == target_key_pos:
            return 4  # Stay - key pickup is automatic
            
        return self._navigate_with_value_iteration(target_key_pos, obs)
        
    def _open_target_door_with_exploration(self, target_door_color, obs=None):
        """Open target door using exploration strategy"""
        if target_door_color is None:
            # No target identified, continue exploring
            return self._explore_action()
            
        # Check memory first, then current observation  
        target_door_pos = None
        if target_door_color in self.memory['door_positions']:
            target_door_pos = self.memory['door_positions'][target_door_color]
        else:
            target_door_pos = self._find_object_position(Door, target_door_color, obs)
            
        if target_door_pos is None:
            # Target door not found, continue exploration even after collecting key
            return self._explore_action()
            
        # Target door found, navigate to it
        if self.agent_pos == target_door_pos:
            return 4  # Stay - door opening is automatic
            
        return self._navigate_with_value_iteration(target_door_pos, obs)

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
        return super()._navigate_with_value_iteration(target_key_pos, obs)

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
        return super()._navigate_with_value_iteration(target_door_pos, obs)

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

    def reset(self):
        """Reset agent state for new episode"""
        super().reset()
        self.current_target = None
        self.path = []
        self.collected_keys = set()
        self.strategy_phase = "collect_key"
        self.target_door_color = None


class Level1ValueAchiever(BaseValueAgent):
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
        conflict_penalty=2.0,
        consumption_penalty=1.0,
        gamma=0.99,
        temperature=0.1,
        q_value_clip=100,
    ):
        # Initialize base class
        super().__init__(
            observability=observability,
            movement_cost=movement_cost,
            wall_penalty=wall_penalty,
            conflict_penalty=conflict_penalty,
            consumption_penalty=consumption_penalty,
            gamma=gamma,
            temperature=temperature,
            q_value_clip=q_value_clip,
            role="achiever",
        )

        # Achiever-specific attributes
        self.current_target = None
        self.path = []
        self.target_door_color = None
        self.collected_keys = set()

        # Strategy phases: "collect_decoy_key", "collect_target_key", "open_door"
        self.strategy_phase = "collect_decoy_key"
        self.decoy_key_color = None
        self.decoy_key_collected = False

        # Blocker observation attributes
        self.blocker_at_door_observed = False
        self.previous_blocker_pos = None

    def _update_agent_position(self, obs):
        """Update achiever position from observations"""
        # Handle both single-agent and multi-agent environments
        if "achiever_pos" in obs:
            new_pos = tuple(obs["achiever_pos"])
        elif "agent_pos" in obs:
            new_pos = tuple(obs["agent_pos"])
        else:
            # Fallback - no position update
            return
        
        if new_pos != self.agent_pos:
            self.agent_pos = new_pos

    def _update_grid_reference(self, obs):
        """Update grid reference from observations"""
        self.grid = obs["achiever"]

    def _get_opponent_position(self, obs):
        """Get blocker position for conflict penalty"""
        if obs and "blocker_pos" in obs and obs["blocker_pos"] is not None:
            return tuple(obs["blocker_pos"])
        return None

    def update_observation(self, obs):
        """Update agent's understanding of the environment"""
        if obs is None:
            return

        # Call base class update
        super().update_observation(obs)

        # Update collected keys based on achiever's key inventory
        achiever_keys_array = obs["achiever_keys"]
        color_map = ["red", "green", "blue", "yellow"]
        self.collected_keys = set()
        for i, has_key in enumerate(achiever_keys_array):
            if has_key > 0 and i < len(color_map):
                self.collected_keys.add(color_map[i])

        # Observe blocker position and check if it goes to any door
        if obs and "blocker_pos" in obs and obs["blocker_pos"] is not None:
            current_blocker_pos = tuple(obs["blocker_pos"])

            # Check if blocker moved to a door position
            if "door_positions" in obs:
                door_positions = obs["door_positions"]
                door_position_tuples = [
                    tuple(pos) for pos in door_positions.values() if pos is not None
                ]

                # If blocker is at any door position, mark as observed
                if current_blocker_pos in door_position_tuples:
                    self.blocker_at_door_observed = True

            self.previous_blocker_pos = current_blocker_pos

    def get_action(self, obs):
        """
        Level1ValueAchiever with enhanced deception strategy for partial observation:
        
        Uses base class act() method for strategy coordination
        """
        # Infer target door color from observations
        if self.target_door_color is None:
            self.target_door_color = self._infer_target_door_color(obs)
            if self.target_door_color:
                self.set_target_door_color(self.target_door_color)
                
        # Set preferred door color for base class target finding
        self._preferred_door_color = self.target_door_color
        
        # Use base class act method for strategy coordination
        return self.act(obs)
        
    def _is_blocker_visible(self, obs):
        """Check if blocker is visible in current observation"""
        return obs and "blocker_pos" in obs and obs["blocker_pos"] is not None
        
    def _is_blocker_nearby(self, obs, distance_threshold=3):
        """Check if blocker is nearby (within threshold distance)"""
        if not self._is_blocker_visible(obs) or self.agent_pos is None:
            return False
        blocker_pos = obs["blocker_pos"]
        distance = abs(self.agent_pos[0] - blocker_pos[0]) + abs(self.agent_pos[1] - blocker_pos[1])
        return distance <= distance_threshold
        
    def _has_seen_blocker(self, obs):
        """Check if blocker has been seen at any point"""
        return self.previous_blocker_pos is not None or self._is_blocker_visible(obs)
        
    def _level0_behavior(self, obs):
        """Behave like Level0ValueAchiever when no blocker seen"""
        # Use the same logic as Level0ValueAchiever
        if self.target_door_color is None:
            return self._explore_action()
            
        if self.strategy_phase == "collect_decoy_key":
            self.strategy_phase = "collect_target_key"  # Skip decoy phase
            
        if self.strategy_phase == "collect_target_key":
            if self.target_door_color in self.collected_keys:
                self.strategy_phase = "open_door"
                return self._open_target_door_with_exploration(self.target_door_color, obs)
            else:
                return self._collect_target_key_with_exploration(self.target_door_color, obs)
                
        return self._explore_action()
        
    def _select_decoy_key_color_from_discovered(self, obs):
        """Select decoy key from discovered keys, or explore to find one"""
        available_keys = set(self.memory['key_positions'].keys())
        if self.target_door_color:
            available_keys.discard(self.target_door_color)
            
        if available_keys:
            self.decoy_key_color = np.random.choice(list(available_keys))
        else:
            # No keys discovered yet, pick any color different from target
            all_colors = ["red", "green", "blue", "yellow"]
            available_colors = [c for c in all_colors if c != self.target_door_color]
            if available_colors:
                self.decoy_key_color = np.random.choice(available_colors)
                
    def _collect_decoy_key_with_deception(self, obs, blocker_visible, blocker_nearby):
        """Collect decoy key with deception behavior"""
        if self.decoy_key_color is None:
            return self._explore_action()
            
        # Check memory first, then observation
        decoy_key_pos = None
        if self.decoy_key_color in self.memory['key_positions']:
            decoy_key_pos = self.memory['key_positions'][self.decoy_key_color]
        else:
            decoy_key_pos = self._find_object_position(Key, self.decoy_key_color, obs)
            
        if decoy_key_pos is None:
            # Decoy key not found, explore
            return self._explore_action()
            
        # Only pretend to move towards decoy when blocker is observing
        if blocker_visible and blocker_nearby:
            # Move towards decoy to confuse blocker
            if self.agent_pos == decoy_key_pos:
                return 4  # Stay at decoy key
            return self._navigate_with_value_iteration(decoy_key_pos, obs)
        else:
            # Blocker not watching, explore for target key instead
            return self._explore_action()
            
    def _collect_target_key_with_deception(self, obs, blocker_visible, blocker_nearby):
        """Collect target key with deception behavior"""
        if self.target_door_color is None:
            return self._explore_action()
            
        # Check memory first, then observation
        target_key_pos = None
        if self.target_door_color in self.memory['key_positions']:
            target_key_pos = self.memory['key_positions'][self.target_door_color]
        else:
            target_key_pos = self._find_object_position(Key, self.target_door_color, obs)
            
        if target_key_pos is None:
            # Target key not found
            if blocker_nearby:
                # Move to misleading locations when blocker nearby  
                return self._move_misleadingly(obs)
            else:
                # Actively explore when blocker far/not visible
                return self._explore_action()
        else:
            # Target key found
            if blocker_nearby:
                # Move misleadingly instead of directly to target
                return self._move_misleadingly(obs)
            else:
                # Navigate to target when blocker not watching
                if self.agent_pos == target_key_pos:
                    return 4  # Stay - pickup automatic
                return self._navigate_with_value_iteration(target_key_pos, obs)
                
    def _move_misleadingly(self, obs):
        """Move to misleading locations when blocker is nearby"""
        # Find a position that's not the target key/door
        misleading_positions = []
        
        # Add discovered key positions (except target) as misleading locations
        for color, pos in self.memory['key_positions'].items():
            if color != self.target_door_color and pos:
                misleading_positions.append(pos)
                
        # Add discovered door positions (except target) as misleading locations  
        for color, pos in self.memory['door_positions'].items():
            if color != self.target_door_color and pos:
                misleading_positions.append(pos)
                
        if misleading_positions:
            misleading_target = np.random.choice(misleading_positions)
            return self._navigate_with_value_iteration(misleading_target, obs)
        else:
            # No misleading locations found, just explore
            return self._explore_action()
            
    def _deceptive_exploration(self, blocker_visible, blocker_nearby):
        """Exploration with deception considerations"""
        if blocker_nearby:
            # Move misleadingly when blocker is watching
            return self._move_misleadingly(None)
        else:
            # Normal exploration when blocker not nearby
            return self._explore_action()
            
    def _collect_target_key_with_exploration(self, target_key_color, obs=None):
        """Collect target key using exploration strategy (for Level0 behavior)"""
        if target_key_color is None:
            return self._explore_action()
            
        # Check memory first, then current observation
        target_key_pos = None
        if target_key_color in self.memory['key_positions']:
            target_key_pos = self.memory['key_positions'][target_key_color]
        else:
            target_key_pos = self._find_object_position(Key, target_key_color, obs)
            
        if target_key_pos is None:
            return self._explore_action()
            
        if self.agent_pos == target_key_pos:
            return 4  # Stay - key pickup is automatic
            
        return self._navigate_with_value_iteration(target_key_pos, obs)
        
    def _open_target_door_with_exploration(self, target_door_color, obs=None):
        """Open target door using exploration strategy"""  
        if target_door_color is None:
            return self._explore_action()
            
        # Check memory first, then current observation
        target_door_pos = None
        if target_door_color in self.memory['door_positions']:
            target_door_pos = self.memory['door_positions'][target_door_color]
        else:
            target_door_pos = self._find_object_position(Door, target_door_color, obs)
            
        if target_door_pos is None:
            return self._explore_action()
            
        if self.agent_pos == target_door_pos:
            return 4  # Stay - door opening is automatic
            
        return self._navigate_with_value_iteration(target_door_pos, obs)

    def _select_decoy_key_color(self, obs):
        """Select a decoy key color that is different from target door color"""
        all_colors = ["red", "green", "blue", "yellow"]
        available_colors = [
            color for color in all_colors if color != self.target_door_color
        ]

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
        return super()._navigate_with_value_iteration(target_key_pos, obs)

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
        return super()._navigate_with_value_iteration(target_door_pos, obs)

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

    def reset(self):
        """Reset agent state for new episode"""
        super().reset()
        self.current_target = None
        self.path = []
        self.collected_keys = set()
        self.strategy_phase = "collect_decoy_key"
        self.decoy_key_color = None
        self.decoy_key_collected = False
        self.target_door_color = None
        # Reset blocker observation attributes
        self.blocker_at_door_observed = False
        self.previous_blocker_pos = None


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
