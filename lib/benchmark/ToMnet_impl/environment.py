import numpy as np
from typing import Tuple, List, Dict, Optional
import random

class GridWorld:
    """
    11x11 GridWorld environment for ToMnet experiments
    Features: random walls (0-4), 4 consumable objects, agent position
    """
    
    def __init__(self, size: int = 11, max_walls: int = 4, max_steps: int = 31):
        self.size = size
        self.max_walls = max_walls
        self.max_steps = max_steps
        
        # Object colors (4 different objects)
        self.n_objects = 4
        
        # Action space: up, down, left, right, stay
        self.actions = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: (0, 0)}
        self.n_actions = len(self.actions)
        
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset environment with new random layout"""
        # Initialize empty grid
        self.walls = np.zeros((self.size, self.size), dtype=bool)
        self.objects = np.zeros((self.size, self.size), dtype=int)  # 0=empty, 1-4=object types
        
        # Add random walls (0-4)
        n_walls = np.random.randint(0, self.max_walls + 1)
        for _ in range(n_walls):
            wall_pos = self._get_random_empty_position()
            if wall_pos is not None:
                self.walls[wall_pos] = True
        
        # Add 4 objects at random positions
        object_positions = []
        for obj_id in range(1, self.n_objects + 1):
            obj_pos = self._get_random_empty_position()
            if obj_pos is not None:
                self.objects[obj_pos] = obj_id
                object_positions.append(obj_pos)
        
        # Place agent at random position
        self.agent_pos = self._get_random_empty_position()
        if self.agent_pos is None:
            self.agent_pos = (0, 0)  # Fallback
        
        self.step_count = 0
        self.done = False
        self.consumed_objects = []
        
        return self.get_state()
    
    def _get_random_empty_position(self) -> Optional[Tuple[int, int]]:
        """Get random empty position on grid"""
        empty_positions = []
        for i in range(self.size):
            for j in range(self.size):
                if not self.walls[i, j] and self.objects[i, j] == 0 and (i, j) != self.agent_pos:
                    empty_positions.append((i, j))
        
        if empty_positions:
            return random.choice(empty_positions)
        return None
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action and return next state, reward, done, info"""
        if self.done:
            return self.get_state(), 0.0, True, {}
        
        # Get movement delta
        delta = self.actions.get(action, (0, 0))
        new_pos = (self.agent_pos[0] + delta[0], self.agent_pos[1] + delta[1])
        
        reward = -0.01  # Movement penalty
        
        # Check bounds and walls
        if (0 <= new_pos[0] < self.size and 
            0 <= new_pos[1] < self.size and 
            not self.walls[new_pos]):
            self.agent_pos = new_pos
        else:
            reward -= 0.05  # Wall penalty
        
        # Check object consumption
        if self.objects[self.agent_pos] > 0:
            consumed_obj = self.objects[self.agent_pos]
            self.consumed_objects.append(consumed_obj)
            self.objects[self.agent_pos] = 0  # Remove object
            self.done = True  # Episode ends when object consumed
        
        self.step_count += 1
        if self.step_count >= self.max_steps:
            self.done = True
        
        return self.get_state(), reward, self.done, {'consumed_object': self.consumed_objects}
    
    def get_state(self) -> np.ndarray:
        """Get current state representation
        Returns: (size, size, 6) array with channels:
        - walls, objects (4 channels), agent position
        """
        state = np.zeros((self.size, self.size, 6))
        
        # Channel 0: walls
        state[:, :, 0] = self.walls.astype(float)
        
        # Channels 1-4: object types
        for obj_id in range(1, self.n_objects + 1):
            state[:, :, obj_id] = (self.objects == obj_id).astype(float)
        
        # Channel 5: agent position
        state[self.agent_pos[0], self.agent_pos[1], 5] = 1.0
        
        return state
    
    def get_flattened_state(self) -> np.ndarray:
        """Get flattened state for neural network input"""
        return self.get_state().flatten()
    
    def render(self) -> str:
        """Simple text rendering for debugging"""
        grid = np.full((self.size, self.size), '.', dtype=str)
        
        # Add walls
        grid[self.walls] = '#'
        
        # Add objects
        for i in range(self.size):
            for j in range(self.size):
                if self.objects[i, j] > 0:
                    grid[i, j] = str(self.objects[i, j])
        
        # Add agent
        grid[self.agent_pos] = 'A'
        
        return '\n'.join([''.join(row) for row in grid])
    
    def get_object_positions(self) -> List[Tuple[int, int]]:
        """Get positions of remaining objects"""
        positions = []
        for i in range(self.size):
            for j in range(self.size):
                if self.objects[i, j] > 0:
                    positions.append((i, j))
        return positions
    
    def copy(self):
        """Create copy of current environment state"""
        new_env = GridWorld(self.size, self.max_walls, self.max_steps)
        new_env.walls = self.walls.copy()
        new_env.objects = self.objects.copy()
        new_env.agent_pos = self.agent_pos
        new_env.step_count = self.step_count
        new_env.done = self.done
        new_env.consumed_objects = self.consumed_objects.copy()
        return new_env