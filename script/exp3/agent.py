import numpy as np
import heapq
from collections import deque

import sys

sys.path.append("../../")

from lib.env.gym_minigrid.minigrid import OBJECT_TO_IDX

"""
A* Agent for MiniGrid-LockedRoom-v0 environment
Adapted from ToMnetF_impl AgentStar class
"""

# MiniGrid Actions
# 0: Turn left
# 1: Turn right
# 2: Move forward
# 3: Pick up / Drop
# 4: Toggle (open/close doors)
# 5: Done


class Node:
    """
    A node class for A* Pathfinding
    """

    def __init__(self, parent=None, position=None, action=None):
        self.parent = parent
        self.position = position
        self.action = action

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

    def __lt__(self, other):
        return self.f < other.f

    def __gt__(self, other):
        return self.f > other.f


class MiniGridAStarAgent:
    """
    A* Agent adapted for MiniGrid-LockedRoom-v0 environment
    """

    def __init__(self, env, max_exploration_steps=1000, debug=False):
        self.env = env
        self.max_exploration_steps = max_exploration_steps
        self.debug = debug

        # State tracking
        self.step_count = 0
        self.trajectory = []
        self.position_trajectory = []
        self.rewards = []
        self.done = False

        # Memory for visited states
        self.visited_positions = set()
        self.explored_grid = {}

        # Mission tracking
        self.has_key = False
        self.key_color = None
        self.key_position = None
        self.door_position = None
        self.goal_position = None

        # Current plan
        self.current_plan = None
        self.plan_step = 0

        # Initialize with first observation
        self.last_obs = None

    def reset(self):
        """Reset agent state"""
        self.step_count = 0
        self.trajectory = []
        self.position_trajectory = []
        self.rewards = []
        self.done = False

        self.visited_positions = set()
        self.explored_grid = {}

        self.has_key = False
        self.key_color = None
        self.key_position = None
        self.door_position = None
        self.goal_position = None

        self.current_plan = None
        self.plan_step = 0

        self.last_obs = None

    def get_agent_position(self, obs):
        """Extract agent position from observation"""
        # Agent position is available in the env directly
        return tuple(self.env.agent_pos)

    def get_agent_direction(self, obs):
        """Extract agent direction from observation"""
        return self.env.agent_dir

    def update_exploration_memory(self, obs):
        """Update memory with visible objects"""
        if obs is None:
            return

        # Get agent position and direction
        agent_pos = self.get_agent_position(obs)
        agent_dir = self.get_agent_direction(obs)

        # Add agent position to visited
        self.visited_positions.add(agent_pos)

        # Parse visible grid from observation
        image = obs["image"]
        height, width, _ = image.shape

        # Convert relative positions to absolute positions
        agent_x, agent_y = agent_pos

        for i in range(height):
            for j in range(width):
                obj_type = image[i, j, 0]
                obj_color = image[i, j, 1] if obj_type != OBJECT_TO_IDX["empty"] else 0

                # Convert relative position to absolute position
                # This is a simplified conversion - in reality we'd need to account for agent direction
                # For now, assume the agent is looking forward and the grid is centered
                abs_x = agent_x + j - width // 2
                abs_y = agent_y + i - height // 2

                if (
                    abs_x >= 0
                    and abs_x < self.env.width
                    and abs_y >= 0
                    and abs_y < self.env.height
                ):
                    self.explored_grid[(abs_x, abs_y)] = {
                        "type": obj_type,
                        "color": obj_color,
                    }

                    # Track specific objects
                    if obj_type == OBJECT_TO_IDX["key"]:
                        self.key_position = (abs_x, abs_y)
                        self.key_color = obj_color
                    elif obj_type == OBJECT_TO_IDX["door"]:
                        self.door_position = (abs_x, abs_y)
                    elif obj_type == OBJECT_TO_IDX["goal"]:
                        self.goal_position = (abs_x, abs_y)

        # Check if agent is carrying key
        if obs.get("carrying") is not None:
            carrying = obs["carrying"]
            if carrying["type"] == OBJECT_TO_IDX["key"]:
                self.has_key = True
                self.key_color = carrying["color"]

    def find_path_to_target(self, target_pos):
        """Find A* path to target position"""
        if target_pos is None:
            return None

        start_pos = self.get_agent_position(self.last_obs)
        if start_pos == target_pos:
            return []

        # A* implementation
        open_list = []
        closed_set = set()

        start_node = Node(None, start_pos)
        start_node.g = 0
        start_node.h = self.manhattan_distance(start_pos, target_pos)
        start_node.f = start_node.g + start_node.h

        heapq.heappush(open_list, start_node)

        while open_list:
            current_node = heapq.heappop(open_list)

            if current_node.position in closed_set:
                continue

            closed_set.add(current_node.position)

            if current_node.position == target_pos:
                # Reconstruct path
                path = []
                while current_node:
                    if current_node.position != start_pos:
                        path.append(current_node.position)
                    current_node = current_node.parent
                return path[::-1]

            # Get neighbors
            neighbors = self.get_valid_neighbors(current_node.position)

            for neighbor_pos in neighbors:
                if neighbor_pos in closed_set:
                    continue

                neighbor_node = Node(current_node, neighbor_pos)
                neighbor_node.g = current_node.g + 1
                neighbor_node.h = self.manhattan_distance(neighbor_pos, target_pos)
                neighbor_node.f = neighbor_node.g + neighbor_node.h

                # Check if this path is better
                better_path = True
                for open_node in open_list:
                    if (
                        open_node.position == neighbor_pos
                        and open_node.f <= neighbor_node.f
                    ):
                        better_path = False
                        break

                if better_path:
                    heapq.heappush(open_list, neighbor_node)

        return None  # No path found

    def get_valid_neighbors(self, position):
        """Get valid neighboring positions"""
        x, y = position
        neighbors = []

        # Check all 4 directions
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_x, new_y = x + dx, y + dy

            # Check bounds
            if 0 <= new_x < self.env.width and 0 <= new_y < self.env.height:
                # Check if position is walkable
                if self.is_walkable((new_x, new_y)):
                    neighbors.append((new_x, new_y))

        return neighbors

    def is_walkable(self, position):
        """Check if position is walkable"""
        if position not in self.explored_grid:
            return True  # Unknown positions are assumed walkable

        obj_info = self.explored_grid[position]
        obj_type = obj_info["type"]

        # Walls are not walkable
        if obj_type == OBJECT_TO_IDX["wall"]:
            return False

        # Locked doors are not walkable unless we have the key
        if obj_type == OBJECT_TO_IDX["door"]:
            if not self.has_key:
                return False

        return True

    def manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def get_action_to_move_to(self, target_pos):
        """Get the action needed to move to target position"""
        current_pos = self.get_agent_position(self.last_obs)
        current_dir = self.get_agent_direction(self.last_obs)

        # Calculate direction to target
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]

        # Determine required direction
        if dx > 0:
            required_dir = 0  # Right
        elif dx < 0:
            required_dir = 2  # Left
        elif dy > 0:
            required_dir = 1  # Down
        elif dy < 0:
            required_dir = 3  # Up
        else:
            return 2  # Move forward if already at target

        # Calculate turns needed
        turn_diff = (required_dir - current_dir) % 4

        if turn_diff == 0:
            return 2  # Move forward
        elif turn_diff == 1:
            return 1  # Turn right
        elif turn_diff == 3:
            return 0  # Turn left
        else:
            return 1  # Turn right (2 turns needed, start with right)

    def choose_action(self, obs):
        """Choose next action based on observation"""
        self.last_obs = obs
        self.update_exploration_memory(obs)

        current_pos = self.get_agent_position(obs)
        self.position_trajectory.append(current_pos)

        # Mission logic: get key -> unlock door -> reach goal
        if not self.has_key and self.key_position is not None:
            # Go to key
            if current_pos == self.key_position:
                action = 3  # Pick up key
            else:
                path = self.find_path_to_target(self.key_position)
                if path:
                    action = self.get_action_to_move_to(path[0])
                else:
                    action = self.explore_action()
        elif self.has_key and self.door_position is not None:
            # Go to door and unlock it
            if self.manhattan_distance(current_pos, self.door_position) == 1:
                # Face the door and toggle it
                action = self.get_action_to_move_to(self.door_position)
                if action == 2:  # If already facing door
                    action = 4  # Toggle door
            else:
                path = self.find_path_to_target(self.door_position)
                if path:
                    action = self.get_action_to_move_to(path[0])
                else:
                    action = self.explore_action()
        elif self.goal_position is not None:
            # Go to goal
            if current_pos == self.goal_position:
                action = 5  # Done
            else:
                path = self.find_path_to_target(self.goal_position)
                if path:
                    action = self.get_action_to_move_to(path[0])
                else:
                    action = self.explore_action()
        else:
            # Explore to find objects
            action = self.explore_action()

        self.trajectory.append(action)
        self.step_count += 1

        if self.debug:
            print(f"Step {self.step_count}: Pos {current_pos}, Action {action}")
            print(f"  Has key: {self.has_key}, Key pos: {self.key_position}")
            print(f"  Door pos: {self.door_position}, Goal pos: {self.goal_position}")

        return action

    def explore_action(self):
        """Simple exploration strategy"""
        # Try to move to unexplored areas
        current_pos = self.get_agent_position(self.last_obs)

        # Find unvisited neighbors
        neighbors = self.get_valid_neighbors(current_pos)
        unvisited = [pos for pos in neighbors if pos not in self.visited_positions]

        if unvisited:
            # Go to closest unvisited neighbor
            target = min(
                unvisited, key=lambda p: self.manhattan_distance(current_pos, p)
            )
            return self.get_action_to_move_to(target)
        else:
            # Random exploration
            if neighbors:
                target = np.random.choice(len(neighbors))
                return self.get_action_to_move_to(neighbors[target])
            else:
                return np.random.choice(6)  # Random action

    def get_trajectory_data(self):
        """Get trajectory data for saving"""
        return {
            "trajectory": self.trajectory,
            "position_trajectory": self.position_trajectory,
            "rewards": self.rewards,
            "step_count": self.step_count,
            "success": self.done and len(self.rewards) > 0 and self.rewards[-1] > 0,
        }
