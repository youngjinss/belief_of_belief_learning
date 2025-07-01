import numpy as np
from typing import Tuple, List, Dict, Optional
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

SIZE = 7
MAX_WALLS = 4
MAX_STEPS = 51

class GridWorld:
    """
    11x11 GridWorld environment for ToMnet experiments
    Features: random walls (0-4), 4 consumable objects, agent position
    """

    def __init__(self, size: int = SIZE, max_walls: int = MAX_WALLS, max_steps: int = MAX_STEPS):
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
        self.objects = np.zeros(
            (self.size, self.size), dtype=int
        )  # 0=empty, 1-4=object types

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
                if not self.walls[i, j] and self.objects[i, j] == 0:
                    # Only check agent position if it exists
                    if not hasattr(self, "agent_pos") or (i, j) != self.agent_pos:
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
        if (
            0 <= new_pos[0] < self.size
            and 0 <= new_pos[1] < self.size
            and not self.walls[new_pos]
        ):
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

        return (
            self.get_state(),
            reward,
            self.done,
            {"consumed_object": self.consumed_objects},
        )

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
        grid = np.full((self.size, self.size), ".", dtype=str)

        # Add walls
        grid[self.walls] = "#"

        # Add objects
        for i in range(self.size):
            for j in range(self.size):
                if self.objects[i, j] > 0:
                    grid[i, j] = str(self.objects[i, j])

        # Add agent
        grid[self.agent_pos] = "A"

        return "\n".join(["".join(row) for row in grid])

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
    
    def visualize(self, ax=None, title="GridWorld State"):
        """Visualize the current state of GridWorld using matplotlib
        
        Args:
            ax: matplotlib axis to draw on (creates new figure if None)
            title: title for the plot
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        ax.clear()
        
        # Set up the grid
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(-0.5, self.size - 0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()  # Invert y-axis to match array indexing
        
        # Draw grid lines
        for i in range(self.size + 1):
            ax.axhline(y=i - 0.5, color='lightgray', linewidth=0.5)
            ax.axvline(x=i - 0.5, color='lightgray', linewidth=0.5)
        
        # Draw walls
        for i in range(self.size):
            for j in range(self.size):
                if self.walls[i, j]:
                    wall = patches.Rectangle((j - 0.5, i - 0.5), 1, 1, 
                                           facecolor='black', edgecolor='gray')
                    ax.add_patch(wall)
        
        # Define colors for objects
        object_colors = ['', 'red', 'blue', 'green', 'yellow']
        object_markers = ['', '●', '■', '▲', '★']
        
        # Draw objects
        for i in range(self.size):
            for j in range(self.size):
                if self.objects[i, j] > 0:
                    obj_id = self.objects[i, j]
                    circle = patches.Circle((j, i), 0.3, 
                                          facecolor=object_colors[obj_id],
                                          edgecolor='black', linewidth=2)
                    ax.add_patch(circle)
                    ax.text(j, i, str(obj_id), ha='center', va='center', 
                           fontsize=12, fontweight='bold', color='white')
        
        # Draw agent
        agent_circle = patches.Circle((self.agent_pos[1], self.agent_pos[0]), 0.35,
                                    facecolor='purple', edgecolor='indigo', 
                                    linewidth=3)
        ax.add_patch(agent_circle)
        ax.text(self.agent_pos[1], self.agent_pos[0], 'A', 
               ha='center', va='center', fontsize=14, fontweight='bold', 
               color='white')
        
        # Add labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f"{title} (Step: {self.step_count}/{self.max_steps})")
        
        # Add legend
        legend_elements = []
        legend_elements.append(patches.Patch(facecolor='purple', label='Agent'))
        legend_elements.append(patches.Patch(facecolor='black', label='Wall'))
        for i in range(1, self.n_objects + 1):
            legend_elements.append(patches.Patch(facecolor=object_colors[i], 
                                               label=f'Object {i}'))
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))
        
        # Add grid coordinates
        ax.set_xticks(range(self.size))
        ax.set_yticks(range(self.size))
        
        plt.tight_layout()
        return ax
    
    def animate_episode(self, actions, save_path=None, interval=500):
        """Create an animation of an episode
        
        Args:
            actions: list of actions to execute
            save_path: path to save animation (if None, displays in notebook)
            interval: milliseconds between frames
        """
        # Save current state to restore later
        initial_state = self.copy()
        
        # Don't reset - use current state as starting point
        # This allows animation from the same initial state as the episode
        
        # Store states for animation
        states = [self.copy()]
        
        for action in actions:
            _, _, done, _ = self.step(action)
            states.append(self.copy())
            if done:
                break
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        def animate(frame):
            state = states[frame]
            # Copy state to current environment
            self.walls = state.walls.copy()
            self.objects = state.objects.copy()
            self.agent_pos = state.agent_pos
            self.step_count = state.step_count
            self.done = state.done
            self.consumed_objects = state.consumed_objects.copy()
            
            self.visualize(ax, title=f"Episode Progress")
            
            # Add action text
            if frame > 0 and frame - 1 < len(actions):
                action_names = ['Up', 'Down', 'Left', 'Right', 'Stay']
                ax.text(0.5, -0.05, f"Action: {action_names[actions[frame-1]]}", 
                       transform=ax.transAxes, ha='center', fontsize=12)
        
        anim = FuncAnimation(fig, animate, frames=len(states), 
                           interval=interval, repeat=True)
        
        # Restore initial state
        self.walls = initial_state.walls.copy()
        self.objects = initial_state.objects.copy()
        self.agent_pos = initial_state.agent_pos
        self.step_count = initial_state.step_count
        self.done = initial_state.done
        self.consumed_objects = initial_state.consumed_objects.copy()
        
        if save_path:
            anim.save(save_path, writer='pillow')
            print(f"Animation saved to {save_path}")
        
        return anim
