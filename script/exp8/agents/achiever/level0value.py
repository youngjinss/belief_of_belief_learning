import numpy as np
import sys
import os

# Add the lib directory to the path
lib_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "lib")
sys.path.insert(0, lib_path)

# Add the env directory to the path
env_path = os.path.join(lib_path, "env")
sys.path.insert(0, env_path)

from gym_minigrid.minigrid import Key, Door, Wall
from utils import set_seed

# Add current directory for config import
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from config import Config
from ..value_agent import BaseValueAgent

# Set seed using Config default value
config = Config()
set_seed(config.seed)


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