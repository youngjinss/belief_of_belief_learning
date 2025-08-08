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


class Level0ValueAchiever(BaseValueAgent):
    """
    Level-0 Value-based Achiever Agent for partial observation environments
    
    Strategy for partial observation:
    - Exploration mode: Use clockwise wall-following when target not found
    - Store discovered key/door positions in memory upon detection
    - If target key not found: Continue exploration mode with clockwise pattern
    - If target key found but not reached: Navigate using value iteration
    - If target door not found: Continue exploration mode even after collecting key
    - If entire map observed: Compute value iteration based on obs + memory
    
    Enhanced with memory management and robust exploration for partial observation.
    """

    def __init__(
        self,
        observability="partial",
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
        self.target_door_color = None
        self.collected_keys = set()
        self.strategy_phase = "collect_key"  # "collect_key" or "open_door"

    def _update_agent_position(self, obs):
        """Update achiever position from observations"""
        if "achiever_pos" in obs:
            new_pos = tuple(obs["achiever_pos"])
        elif "agent_pos" in obs:
            new_pos = tuple(obs["agent_pos"])
        else:
            return

        if new_pos != self.agent_pos:
            self.agent_pos = new_pos

    def _update_grid_reference(self, obs):
        """Update grid reference from observations"""
        self.grid = obs.get("achiever")

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
        achiever_keys_array = obs.get("achiever_keys", [])
        color_map = ["red", "green", "blue", "yellow"]
        self.collected_keys = set()
        for i, has_key in enumerate(achiever_keys_array):
            if has_key > 0 and i < len(color_map):
                self.collected_keys.add(color_map[i])

    def get_action(self, obs):
        """
        Get the next action for the agent using value iteration with partial observation strategy
        """
        self.update_observation(obs)

        # Infer target door color from observations
        if self.target_door_color is None:
            self.target_door_color = self._infer_target_door_color(obs)
            # Set target door color in base class for consumption penalty
            self.set_target_door_color(self.target_door_color)

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

        # Fallback: use clockwise exploration pattern
        return self._explore_with_clockwise_pattern()

    def _collect_target_key(self, target_key_color, obs=None):
        """Strategy to collect the target key using value iteration or exploration"""
        # Check memory first for discovered key position
        target_key_pos = self.memory.get(f"key_{target_key_color}")
        
        # If not in memory, check current observations
        if target_key_pos is None and obs and "key_positions" in obs:
            target_key_pos = obs["key_positions"].get(target_key_color)
            if target_key_pos is not None:
                target_key_pos = tuple(target_key_pos)
                
        # If key position found, navigate to it
        if target_key_pos is not None:
            # Check if we're already at the key position
            if self.agent_pos == target_key_pos:
                return 4  # Stay - key pickup is automatic
                
            # Use value iteration to navigate to key
            return self._navigate_with_value_iteration(target_key_pos, obs)
        
        # Key not found - continue exploration with clockwise pattern
        return self._explore_with_clockwise_pattern()

    def _open_target_door(self, target_door_color, obs=None):
        """Strategy to open the target door using value iteration or exploration"""
        # Check memory first for discovered door position
        target_door_pos = self.memory.get(f"door_{target_door_color}")
        
        # If not in memory, check current observations
        if target_door_pos is None and obs and "door_positions" in obs:
            target_door_pos = obs["door_positions"].get(target_door_color)
            if target_door_pos is not None:
                target_door_pos = tuple(target_door_pos)
                
        # If door position found, navigate to it
        if target_door_pos is not None:
            # Check if we're at the door position
            if self.agent_pos == target_door_pos:
                # Check door state through grid if available
                if self.grid is not None:
                    door = self.grid.get(*target_door_pos)
                    if isinstance(door, Door) and door.is_open:
                        return 4  # Stay on opened door
                    elif (isinstance(door, Door) and door.is_locked and 
                          target_door_color in self.collected_keys):
                        return 4  # Stay - door opening is automatic
                return 4  # Stay at door position
                
            # Use value iteration to navigate to door
            return self._navigate_with_value_iteration(target_door_pos, obs)
            
        # Door not found - continue exploration with clockwise pattern
        return self._explore_with_clockwise_pattern()

    def _infer_target_door_color(self, obs=None):
        """Infer target door color from observations."""
        # Use target door color from observations if available
        if obs and "target_door_color" in obs:
            return obs["target_door_color"]

        # Fallback: use first available door color from observations
        if obs and "door_positions" in obs:
            door_colors = list(obs["door_positions"].keys())
            if door_colors:
                return door_colors[0]

        # Check memory for any discovered doors
        door_colors_in_memory = []
        for key in self.memory.keys():
            if key.startswith("door_"):
                door_colors_in_memory.append(key.split("door_")[1])
        
        if door_colors_in_memory:
            return door_colors_in_memory[0]

        # Final fallback
        return "red"

    def reset(self):
        """Reset agent state for new episode"""
        super().reset()
        self.target_door_color = None
        self.collected_keys = set()
        self.strategy_phase = "collect_key"