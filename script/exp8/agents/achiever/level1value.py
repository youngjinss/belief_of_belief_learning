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