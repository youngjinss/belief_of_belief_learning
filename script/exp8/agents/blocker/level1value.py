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


class Level1ValueBlocker(BaseValueAgent):
    """
    Level-1 Value-based Blocker Agent for partial observation with inference
    
    Strategy for partial observation with inference:
    - Exploration mode: Use clockwise wall-following for systematic exploration
    - Identify door colors and positions during exploration process
    - If achiever not found: Execute exploration mode with clockwise pattern
    - If achiever found but has no key: Follow achiever
    - If achiever has key but door not found: Switch to exploration mode to identify doors
    - If door color matches inferred target, compute value iteration based on obs + memory  
    - If achiever lost: Return to exploration mode with clockwise pattern
    - If entire map observed: Compute value iteration based on obs + memory
    
    Enhanced with sophisticated opponent tracking and target inference for partial observation.
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

        # Inferred target
        self.target_color = None
        self.target_pos = None
        self.achiever_has_key = False
        self.achiever_last_seen_pos = None
        self.achiever_ever_seen = False

        # Phase tracking for partial observation
        self.phase = "exploration"  # "exploration", "follow_achiever", "go_to_door", "break_door"

        # Multi-attempt tracking
        self.observed_keys = []  # Track keys observed from achiever
        self.current_key_index = 0  # Index of current key being used for inference
        self.just_attempted_break = False
        self.last_action = None
        
        # Enhanced tracking for partial observation
        self.achiever_observation_history = []  # Track achiever positions
        self.achiever_key_pickup_locations = []  # Track where achiever picked up keys
        self.follow_distance_threshold = 3  # Stay within 3 tiles of achiever when following

    def _update_agent_position(self, obs):
        """Update blocker position from observations"""
        new_pos = tuple(obs["blocker_pos"])
        if new_pos != self.agent_pos:
            self.agent_pos = new_pos

    def _update_grid_reference(self, obs):
        """Update grid reference from observations.

        obs["blocker"] is the MiniGrid observation dict (image/direction/mission),
        not a Grid. Assigning it directly left self.grid as a dict, so
        BaseValueAgent's self.grid.width raised AttributeError and
        self.grid.get(x, y) silently resolved to dict.get, returning the y
        argument instead of a cell. Decode the image, matching level0value.py.
        """
        if "blocker" in obs and obs["blocker"] and "image" in obs["blocker"]:
            from gym_minigrid.minigrid import Grid

            self.grid = Grid.decode(obs["blocker"]["image"])

    def _get_opponent_position(self, obs):
        """Get achiever position for conflict penalty and tracking"""
        if obs and "achiever_pos" in obs and obs["achiever_pos"] is not None:
            return tuple(obs["achiever_pos"])
        return None

    def update_observation(self, obs):
        """Update agent's understanding and track achiever behavior"""
        if obs is None:
            return

        # Call base class update
        super().update_observation(obs)

        # Track achiever position
        achiever_pos = self._get_opponent_position(obs)
        if achiever_pos is not None:
            self.achiever_last_seen_pos = achiever_pos
            self.achiever_ever_seen = True
            self.achiever_observation_history.append(achiever_pos)
            
            # Maintain limited history for efficiency
            if len(self.achiever_observation_history) > 20:
                self.achiever_observation_history.pop(0)

    @property
    def target_door_color(self):
        """Get the current target door color for interaction checking."""
        return self.target_color

    @property
    def target_inferred_color(self):
        """Get the current target inferred color for Level1ValueBlocker."""
        return self.target_color

    def get_action(self, obs):
        """
        Get action for Level-1 value-based blocker agent with inference strategy
        """
        if obs is None:
            return 4  # Stay if no observation

        # Update internal state from observations
        self.update_observation(obs)

        # Check if we just attempted to break and game is still continuing
        if self.just_attempted_break:
            # Game didn't end, so we broke the wrong door
            # Move to next observed key if available
            self.current_key_index += 1
            if self.current_key_index >= len(self.observed_keys):
                # No more observed keys, return to exploration
                self.phase = "exploration" 
                self.target_color = None
                self.target_pos = None
            else:
                # Use next observed key, go back to inference
                self.phase = "go_to_door"
                self.target_color = None
                self.target_pos = None
            self.just_attempted_break = False

        # Check and store achiever keys
        self._check_and_store_achiever_keys(obs)
        
        # Update phase based on current situation
        self._update_phase_based_on_situation(obs)

        # Execute action based on current phase
        if self.phase == "exploration":
            return self._handle_exploration_phase()
            
        elif self.phase == "follow_achiever":
            return self._handle_follow_achiever_phase(obs)
            
        elif self.phase == "go_to_door":
            return self._handle_go_to_door_phase(obs)
            
        elif self.phase == "break_door":
            self.just_attempted_break = True
            self.last_action = 5
            return 5  # Break action

        # Default: exploration
        return self._explore_with_clockwise_pattern()

    def _update_phase_based_on_situation(self, obs):
        """Update phase based on current observations and agent state"""
        # If achiever never seen, stay in exploration
        if not self.achiever_ever_seen:
            self.phase = "exploration"
            return
            
        # If achiever was seen but now lost, return to exploration
        achiever_pos = self._get_opponent_position(obs)
        if achiever_pos is None:
            self.phase = "exploration"
            return
            
        # If achiever found but has no keys, follow achiever
        if not self.achiever_has_key:
            self.phase = "follow_achiever"
            return
            
        # If achiever has key(s), try to infer target and go to door
        if self.achiever_has_key:
            if self.target_color is None:
                self._infer_target_from_observed_keys(obs)
                
            if self.target_color is not None and self.target_pos is not None:
                # If we're at the target door, break it
                if self._at_target_door():
                    self.phase = "break_door"
                else:
                    self.phase = "go_to_door"
            else:
                # Target door not found, explore to find doors
                self.phase = "exploration"

    def _handle_exploration_phase(self):
        """Handle exploration phase using clockwise wall-following"""
        return self._explore_with_clockwise_pattern()

    def _handle_follow_achiever_phase(self, obs):
        """Handle following achiever phase"""
        achiever_pos = self._get_opponent_position(obs)
        if achiever_pos is None or self.agent_pos is None:
            return 4  # Stay if positions unknown
            
        # Calculate distance to achiever
        distance = self._calculate_distance(self.agent_pos, achiever_pos)
        
        # If too far, move closer; if too close, maintain distance
        if distance > self.follow_distance_threshold:
            return self._navigate_with_value_iteration(achiever_pos, obs)
        else:
            # Stay nearby but don't get too close
            return 4  # Stay

    def _handle_go_to_door_phase(self, obs):
        """Handle navigating to inferred target door"""
        if self.target_pos is None:
            # Target door position unknown, return to exploration
            self.phase = "exploration"
            return self._explore_with_clockwise_pattern()
            
        # Navigate to target door using value iteration
        return self._navigate_with_value_iteration(self.target_pos, obs)

    def _at_target_door(self):
        """Check if blocker is at the target door position."""
        if self.target_pos is None or self.agent_pos is None:
            return False
        return tuple(self.agent_pos) == tuple(self.target_pos)

    def _calculate_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _check_and_store_achiever_keys(self, obs):
        """Check for new achiever key pickups and store them in observed_keys."""
        if obs and "achiever_keys" in obs:
            achiever_keys = obs["achiever_keys"]
            color_map = ["red", "green", "blue", "yellow"]

            # Find any keys the achiever has and store color names
            for i, has_key in enumerate(achiever_keys):
                if has_key > 0 and i < len(color_map):
                    key_color = color_map[i]
                    if key_color not in self.observed_keys:
                        self.observed_keys.append(key_color)
                        
                        # Store pickup location if we know achiever position
                        if self.achiever_last_seen_pos:
                            self.achiever_key_pickup_locations.append({
                                'color': key_color,
                                'location': self.achiever_last_seen_pos
                            })

            # Update achiever_has_key flag
            if (
                len(achiever_keys) > 0
                and achiever_keys.sum() > 0
                and not self.achiever_has_key
            ):
                self.achiever_has_key = True

    def _infer_target_from_observed_keys(self, obs):
        """Infer target door color from current observed key."""
        if self.current_key_index < len(self.observed_keys):
            # Use the key at current_key_index for inference
            key_color = self.observed_keys[self.current_key_index]
            self.target_color = key_color

            # Find position of inferred door from memory first
            self.target_pos = self.memory.get(f"door_{self.target_color}")
            
            # If not in memory, check current observations
            if self.target_pos is None and obs and "door_positions" in obs:
                door_pos = obs["door_positions"].get(self.target_color)
                if door_pos is not None:
                    self.target_pos = tuple(door_pos)

    def set_env(self, env):
        """Set environment reference for action decisions"""
        self.env = env

    def reset(self):
        """Reset agent state for new episode"""
        super().reset()
        # Inferred target
        self.target_color = None
        self.target_pos = None
        self.achiever_has_key = False
        self.achiever_last_seen_pos = None
        self.achiever_ever_seen = False

        # Phase tracking
        self.phase = "exploration"

        # Reset multi-attempt state
        self.observed_keys.clear()
        self.current_key_index = 0
        self.just_attempted_break = False
        self.last_action = None
        
        # Reset tracking
        self.achiever_observation_history.clear()
        self.achiever_key_pickup_locations.clear()