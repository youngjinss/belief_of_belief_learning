import numpy as np
import sys
import os

# Add parent directories to path for imports
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..', '..'))
lib_path = os.path.join(project_root, 'lib', 'env')
sys.path.insert(0, lib_path)
sys.path.append(os.path.join(script_dir, '..'))
sys.path.append(os.path.join(project_root, 'lib'))
sys.path.append(os.path.join(project_root))

from gym_minigrid.minigrid import Key, Door
from value_agent import BaseValueAgent


class Level0ValueBlocker(BaseValueAgent):
    """
    Level-0 Value-based Blocker Agent for partial observation with random selection
    
    Strategy for partial observation with random selection:
    - Exploration mode: Use clockwise wall-following when doors not found
    - Store discovered door positions in memory upon detection  
    - If no doors found: Continue exploration mode with clockwise pattern
    - If some doors found: Randomly select from discovered doors
    - Navigate to selected door and attempt break action
    - Mark failed doors and never retry them
    - If failed, select from other discovered doors
    - If entire map observed: Compute value iteration based on obs + memory
    
    Enhanced with memory management and robust exploration for partial observation.
    """

    def __init__(
        self,
        observability="partial",
        movement_cost=0.01,
        wall_penalty=2.0,
        conflict_penalty=2.0,
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
            gamma=gamma,
            temperature=temperature,
            q_value_clip=q_value_clip,
            role="blocker",
        )

        # Blocker-specific attributes
        self.target_inferred_color = None
        self.target_door_pos = None
        self.target_selected = False

        # Multi-attempt tracking
        self.tried_doors = set()  # Track which doors have been attempted
        self.available_doors = {"red", "green", "blue", "yellow"}
        self.just_attempted_break = False
        self.last_action = None

    def _update_agent_position(self, obs):
        """Update blocker position from observations"""
        new_pos = tuple(obs["blocker_pos"])
        if new_pos != self.agent_pos:
            self.agent_pos = new_pos

    def _update_grid_reference(self, obs):
        """Update grid reference from observations"""
        if "blocker" in obs and obs["blocker"] and "image" in obs["blocker"]:
            from gym_minigrid.minigrid import Grid
            self.grid = Grid.decode(obs["blocker"]["image"])

    def _get_opponent_position(self, obs):
        """Get achiever position for conflict penalty"""
        if obs and "achiever_pos" in obs and obs["achiever_pos"] is not None:
            return tuple(obs["achiever_pos"])
        return None

    def get_action(self, obs):
        """
        Get action for Level-0 value-based blocker agent with partial observation strategy
        """
        if obs is None:
            return 4  # Stay if no observation

        # Update internal state from observations
        self.update_observation(obs)
        self._update_memory(obs, self.agent_pos)

        # Check if we just attempted to break and game is still continuing
        if self.just_attempted_break:
            # Game didn't end, so we broke the wrong door
            # Mark current target as tried and select new target
            if self.target_inferred_color:
                self.tried_doors.add(self.target_inferred_color)
            self._reset_for_new_attempt()
            self.just_attempted_break = False

        # Select target door randomly if not already selected
        if not self.target_selected:
            self._select_random_target_door(obs)

        # If still no target selected (no doors found), continue exploration
        if not self.target_selected:
            return self._explore_with_clockwise_pattern()

        # If we're at the target door, break it
        if self._at_target_door():
            self.just_attempted_break = True
            self.last_action = 5
            return 5  # Break action

        # Navigate to target door using value iteration
        if self.target_door_pos is not None:
            action = self._navigate_with_value_iteration(self.target_door_pos, obs)
            self.last_action = action
            return action
        
        # Fallback: continue exploration
        return self._explore_with_clockwise_pattern()

    def _select_random_target_door(self, obs):
        """Select target door color randomly from discovered doors (not from tried doors)"""
        # Get doors discovered in memory
        discovered_doors = []
        for key in self.memory.keys():
            if key.startswith("door_"):
                door_color = key.split("door_")[1]
                if door_color not in self.tried_doors:
                    discovered_doors.append(door_color)
                    
        # Also check current observations for additional doors
        if obs and "door_positions" in obs:
            for color, pos in obs["door_positions"].items():
                if pos is not None and color not in self.tried_doors and color not in discovered_doors:
                    discovered_doors.append(color)

        # If no untried doors discovered, reset tried doors and use all discovered
        if not discovered_doors:
            self.tried_doors.clear()
            for key in self.memory.keys():
                if key.startswith("door_"):
                    door_color = key.split("door_")[1] 
                    discovered_doors.append(door_color)
                    
            # Add doors from observations if available
            if obs and "door_positions" in obs:
                for color, pos in obs["door_positions"].items():
                    if pos is not None and color not in discovered_doors:
                        discovered_doors.append(color)

        # If we have discovered doors, randomly select one
        if discovered_doors:
            self.target_inferred_color = np.random.choice(discovered_doors)
            
            # Get position from memory first, then observations
            self.target_door_pos = self.memory.get(f"door_{self.target_inferred_color}")
            
            if self.target_door_pos is None and obs and "door_positions" in obs:
                door_pos = obs["door_positions"].get(self.target_inferred_color)
                if door_pos is not None:
                    self.target_door_pos = tuple(door_pos)
                    
            # Mark target as selected if we found position
            if self.target_door_pos is not None:
                self.target_selected = True
                return
                
        # No doors found yet - target not selected, will continue exploration

    def _reset_for_new_attempt(self):
        """Reset targeting state for a new attempt."""
        self.target_selected = False
        self.target_inferred_color = None
        self.target_door_pos = None

    def _at_target_door(self):
        """Check if blocker is at the target door position."""
        if self.target_door_pos is None or self.agent_pos is None:
            return False
        return tuple(self.agent_pos) == tuple(self.target_door_pos)

    @property
    def target_door_color(self):
        """
        Get the current target door color for interaction checking.
        Uses target_inferred_color for Level0ValueBlocker.
        """
        return self.target_inferred_color

    def set_env(self, env):
        """Set environment reference for action decisions"""
        self.env = env

    def reset(self):
        """Reset agent state for new episode"""
        super().reset()
        self.target_inferred_color = None
        self.target_door_pos = None
        self.target_selected = False
        # Reset multi-attempt state
        self.tried_doors.clear()
        self.just_attempted_break = False
        self.last_action = None