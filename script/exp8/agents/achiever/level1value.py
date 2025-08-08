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


class Level1ValueAchiever(BaseValueAgent):
    """
    Level-1 Value-based Achiever Agent for partial observation with deception
    
    Strategy for partial observation with deception:
    - Exploration mode: Use clockwise wall-following when targets not found  
    - If decoy key not found: Set any discovered key as decoy and move towards it
    - Only pretend to move towards decoy key when blocker is observing
    - If target key not found: Actively explore with clockwise pattern when blocker is far
    - Move to misleading locations when blocker is nearby to cause confusion
    - If blocker never seen: Behave like Level0 (no deception needed)
    - If entire map observed: Compute value iteration based on obs + memory
    
    Enhanced with sophisticated deception mechanics and opponent tracking for partial observation.
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

        # Strategy phases: "collect_decoy_key", "collect_target_key", "open_door"
        self.strategy_phase = "collect_decoy_key"
        self.decoy_key_color = None
        self.decoy_key_collected = False

        # Blocker observation and deception attributes
        self.blocker_last_seen_pos = None
        self.blocker_observation_history = []  # Track blocker positions over time
        self.blocker_ever_seen = False
        self.deception_active = False
        
        # Distance thresholds for deception activation
        self.deception_distance_threshold = 3  # Activate deception when blocker within 3 tiles
        self.visibility_distance_threshold = 5  # Consider blocker "nearby" within 5 tiles

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
        """Get blocker position for conflict penalty and deception logic"""
        if obs and "blocker_pos" in obs and obs["blocker_pos"] is not None:
            return tuple(obs["blocker_pos"])
        return None

    def update_observation(self, obs):
        """Update agent's understanding of the environment and track blocker"""
        if obs is None:
            return

        # Call base class update
        super().update_observation(obs)

        # Update collected keys
        achiever_keys_array = obs.get("achiever_keys", [])
        color_map = ["red", "green", "blue", "yellow"]
        self.collected_keys = set()
        for i, has_key in enumerate(achiever_keys_array):
            if has_key > 0 and i < len(color_map):
                self.collected_keys.add(color_map[i])

        # Track blocker position and update deception state
        blocker_pos = self._get_opponent_position(obs)
        if blocker_pos is not None:
            self.blocker_last_seen_pos = blocker_pos
            self.blocker_observation_history.append(blocker_pos)
            self.blocker_ever_seen = True
            
            # Maintain limited history for efficiency
            if len(self.blocker_observation_history) > 10:
                self.blocker_observation_history.pop(0)
                
            # Update deception activation based on blocker proximity
            if self.agent_pos is not None:
                distance = self._calculate_distance(self.agent_pos, blocker_pos)
                self.deception_active = distance <= self.deception_distance_threshold

    def get_action(self, obs):
        """
        Get the next action with deceptive strategy for partial observation
        """
        self.update_observation(obs)

        # Infer target door color from observations
        if self.target_door_color is None:
            self.target_door_color = self._infer_target_door_color(obs)
            self.set_target_door_color(self.target_door_color)

        # If blocker never seen, behave like Level0
        if not self.blocker_ever_seen:
            return self._behave_like_level0(obs)

        # Select decoy key color if not already selected
        if self.decoy_key_color is None:
            self._select_decoy_key_color(obs)

        # Phase 1: Collect decoy key (with deception when blocker observing)
        if self.strategy_phase == "collect_decoy_key":
            if (self.decoy_key_color in self.collected_keys or 
                self._should_skip_decoy_phase(obs)):
                self.decoy_key_collected = True
                self.strategy_phase = "collect_target_key"
            else:
                return self._collect_decoy_key_with_deception(obs)

        # Phase 2: Collect target key (avoid blocker detection)
        elif self.strategy_phase == "collect_target_key":
            if self.target_door_color in self.collected_keys:
                self.strategy_phase = "open_door"
            else:
                return self._collect_target_key_stealthily(obs)

        # Phase 3: Open target door
        elif self.strategy_phase == "open_door":
            return self._open_target_door(self.target_door_color, obs)

        # Fallback: use clockwise exploration
        return self._explore_with_clockwise_pattern()

    def _behave_like_level0(self, obs):
        """Behave like Level0ValueAchiever when no blocker is observed"""
        target_key_color = self.target_door_color
        
        if target_key_color in self.collected_keys:
            # Go to door
            return self._open_target_door(target_key_color, obs)
        else:
            # Collect key
            return self._collect_target_key_directly(target_key_color, obs)

    def _should_skip_decoy_phase(self, obs):
        """Determine if we should skip decoy phase based on blocker behavior"""
        # Skip decoy if blocker has been to any door (indicates Level1+ blocker)
        if self.blocker_observation_history and obs and "door_positions" in obs:
            door_positions = list(obs["door_positions"].values())
            for blocker_pos in self.blocker_observation_history:
                if blocker_pos in [tuple(pos) for pos in door_positions if pos is not None]:
                    return True
        return False

    def _select_decoy_key_color(self, obs):
        """Select a decoy key color different from target door color"""
        all_colors = ["red", "green", "blue", "yellow"]
        available_colors = [
            color for color in all_colors if color != self.target_door_color
        ]
        
        # Prefer colors that have been discovered in memory
        discovered_decoy_colors = []
        for color in available_colors:
            if f"key_{color}" in self.memory:
                discovered_decoy_colors.append(color)
                
        if discovered_decoy_colors:
            self.decoy_key_color = np.random.choice(discovered_decoy_colors)
        elif available_colors:
            self.decoy_key_color = np.random.choice(available_colors)
        else:
            # Fallback: random color
            self.decoy_key_color = np.random.choice(all_colors)

    def _collect_decoy_key_with_deception(self, obs):
        """Collect decoy key with deceptive behavior when blocker is observing"""
        decoy_key_pos = self.memory.get(f"key_{self.decoy_key_color}")
        
        # If decoy not in memory, check observations
        if decoy_key_pos is None and obs and "key_positions" in obs:
            decoy_key_pos = obs["key_positions"].get(self.decoy_key_color)
            if decoy_key_pos is not None:
                decoy_key_pos = tuple(decoy_key_pos)
                
        if decoy_key_pos is not None:
            # If blocker is watching and we're not at decoy yet, navigate to decoy
            if self.deception_active and self.agent_pos != decoy_key_pos:
                return self._navigate_with_value_iteration(decoy_key_pos, obs)
            # If at decoy position, stay to collect
            elif self.agent_pos == decoy_key_pos:
                return 4  # Stay - automatic pickup
                
        # If decoy not found or blocker not watching, explore for any key
        if not self.deception_active:
            # Actively explore when blocker is far
            return self._explore_with_clockwise_pattern()
        else:
            # Move to misleading location when blocker is nearby
            return self._move_misleadingly(obs)

    def _collect_target_key_stealthily(self, obs):
        """Collect target key while avoiding blocker detection"""
        target_key_pos = self.memory.get(f"key_{self.target_door_color}")
        
        # If target not in memory, check observations
        if target_key_pos is None and obs and "key_positions" in obs:
            target_key_pos = obs["key_positions"].get(self.target_door_color)
            if target_key_pos is not None:
                target_key_pos = tuple(target_key_pos)
                
        if target_key_pos is not None:
            # If blocker is watching, move misleadingly
            if self.deception_active:
                return self._move_misleadingly(obs)
            # If blocker is far, navigate to target key
            else:
                if self.agent_pos == target_key_pos:
                    return 4  # Stay - automatic pickup
                return self._navigate_with_value_iteration(target_key_pos, obs)
                
        # Target key not found - explore when blocker is not watching
        if not self.deception_active:
            return self._explore_with_clockwise_pattern()
        else:
            return self._move_misleadingly(obs)

    def _collect_target_key_directly(self, target_key_color, obs):
        """Direct target key collection (Level0 behavior)"""
        target_key_pos = self.memory.get(f"key_{target_key_color}")
        
        if target_key_pos is None and obs and "key_positions" in obs:
            target_key_pos = obs["key_positions"].get(target_key_color)
            if target_key_pos is not None:
                target_key_pos = tuple(target_key_pos)
                
        if target_key_pos is not None:
            if self.agent_pos == target_key_pos:
                return 4  # Stay
            return self._navigate_with_value_iteration(target_key_pos, obs)
            
        return self._explore_with_clockwise_pattern()

    def _open_target_door(self, target_door_color, obs):
        """Navigate to and open target door"""
        target_door_pos = self.memory.get(f"door_{target_door_color}")
        
        if target_door_pos is None and obs and "door_positions" in obs:
            target_door_pos = obs["door_positions"].get(target_door_color)
            if target_door_pos is not None:
                target_door_pos = tuple(target_door_pos)
                
        if target_door_pos is not None:
            if self.agent_pos == target_door_pos:
                if self.grid is not None:
                    door = self.grid.get(*target_door_pos)
                    if isinstance(door, Door) and door.is_open:
                        return 4  # Stay on opened door
                    elif (isinstance(door, Door) and door.is_locked and
                          target_door_color in self.collected_keys):
                        return 4  # Stay - door opening is automatic
                return 4
                
            return self._navigate_with_value_iteration(target_door_pos, obs)
            
        return self._explore_with_clockwise_pattern()

    def _move_misleadingly(self, obs):
        """Move to misleading locations when blocker is watching"""
        if self.blocker_last_seen_pos is None or self.agent_pos is None:
            return self._explore_with_clockwise_pattern()
            
        # Find position that moves away from real target but looks purposeful
        misleading_positions = []
        
        # Check all discovered keys that are not our target
        for key in self.memory.keys():
            if key.startswith("key_") and not key.endswith(self.target_door_color):
                misleading_positions.append(self.memory[key])
                
        # If we have misleading positions, move toward the closest one
        if misleading_positions:
            closest_misleading = min(misleading_positions, 
                                   key=lambda pos: self._calculate_distance(self.agent_pos, pos))
            return self._navigate_with_value_iteration(closest_misleading, obs)
            
        # Otherwise, move in a direction that looks random but avoids target
        return self._explore_with_clockwise_pattern()

    def _calculate_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _infer_target_door_color(self, obs):
        """Infer target door color from observations"""
        if obs and "target_door_color" in obs:
            return obs["target_door_color"]

        if obs and "door_positions" in obs:
            door_colors = list(obs["door_positions"].keys())
            if door_colors:
                return door_colors[0]

        # Check memory
        door_colors_in_memory = []
        for key in self.memory.keys():
            if key.startswith("door_"):
                door_colors_in_memory.append(key.split("door_")[1])
        
        if door_colors_in_memory:
            return door_colors_in_memory[0]

        return "red"

    def reset(self):
        """Reset agent state for new episode"""
        super().reset()
        self.target_door_color = None
        self.collected_keys = set()
        self.strategy_phase = "collect_decoy_key"
        self.decoy_key_color = None
        self.decoy_key_collected = False
        
        # Reset blocker observation attributes
        self.blocker_last_seen_pos = None
        self.blocker_observation_history = []
        self.blocker_ever_seen = False
        self.deception_active = False