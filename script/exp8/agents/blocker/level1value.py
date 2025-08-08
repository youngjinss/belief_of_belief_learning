import numpy as np
import sys
import os

# Add the lib directory to the path
lib_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "lib")
sys.path.insert(0, lib_path)

# Add the env directory to the path
env_path = os.path.join(lib_path, "env")
sys.path.insert(0, env_path)

from utils import set_seed

# Add current directory for config import
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from config import Config
from ..value_agent import BaseValueAgent

# Set seed using Config default value
config = Config()
set_seed(config.seed)


class Level1ValueBlocker(BaseValueAgent):
    """
    Level-1 Value-based Blocker Agent for AchieverBlocker environment.

    Simple "Stay, Watch, and Go" Strategy:

    Phase 1: Observation
    - Stay in place and wait until achiever picks up the first key
    - Store the observed key color

    Phase 2: Target inference and navigation
    - Infer that the first picked key is the target
    - Navigate to the corresponding door using value iteration

    Phase 3: Door breaking
    - Attempt to break the target door
    - If game continues (wrong door), wait for next key pickup
    - Use the second observed key as the new target and repeat

    The agent uses a simple reactive strategy without bluffing or prediction,
    relying solely on observed key pickups to infer the achiever's target.
    """

    def __init__(
        self,
        observability="full",
        movement_cost=0.01,
        wall_penalty=2.0,
        conflict_penalty=2.0,
        gamma=0.99,
        temperature=0.1,
        q_value_clip=100,
    ):
        """Initialize Level-1 value-based blocker agent."""
        # Initialize base class with blocker role
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

        # Navigation state
        self.path_to_door = []
        self.current_path_index = 0
        self.blocker_pos = None
        self.achiever_pos = None

        # Phase tracking
        self.phase = 1  # 1: wait and observe, 2: go to inferred door, 3: break door

        # Multi-attempt tracking
        self.observed_keys = []  # Track keys observed from achiever
        self.current_key_index = 0  # Index of current key being used for inference
        self.just_attempted_break = False
        self.last_action = None

    def _update_agent_position(self, obs):
        """Update blocker position from observations"""
        new_pos = tuple(obs["blocker_pos"])
        if new_pos != self.agent_pos:
            self.agent_pos = new_pos

    def _update_grid_reference(self, obs):
        """Update grid reference from observations"""
        self.grid = obs["blocker"]

    def _get_opponent_position(self, obs):
        """Get achiever position for conflict penalty"""
        if obs and "achiever_pos" in obs:
            return tuple(obs["achiever_pos"])
        return None

    def update_observation(self, obs):
        """Update agent's understanding of the environment"""
        if obs is None:
            return

        # Call base class update
        super().update_observation(obs)

        # Update blocker-specific state
        self.blocker_pos = self.agent_pos
        self.achiever_pos = self._get_opponent_position(obs)
        # Set opponent_pos for base class target finding
        self.opponent_pos = self.achiever_pos

    @property
    def target_door_color(self):
        """
        Get the current target door color for interaction checking.
        For Level1ValueBlocker, this returns the inferred target color.

        Returns:
            str: Current target door color
        """
        return self.target_color

    @property
    def target_inferred_color(self):
        """
        Get the current target inferred color for Level1ValueBlocker.
        This matches the interface expected by other parts of the system.

        Returns:
            str: Current target inferred color
        """
        return self.target_door_color

    def get_action(self, obs):
        """
        Level1ValueBlocker with enhanced inference strategy for partial observation:
        
        - Exploration mode: Move in clockwise pattern from slightly off-center position, identify door colors/positions
        - If achiever not found: Execute exploration mode  
        - If achiever found but has no key: Follow achiever
        - If achiever has key but door not found: Switch to exploration mode to identify door positions
        - If door color matches inferred target: Compute value iteration based on obs + memory
        - If achiever lost: Return to exploration mode
        - If entire map observed: Compute value iteration based on obs + memory as in full observation
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
                # No more observed keys, reset to exploration
                self.phase = 1
                self.target_color = None
                self.target_pos = None
            else:
                # Use next observed key, go back to Phase 2
                self.phase = 2
                self.target_color = None
                self.target_pos = None
            self.just_attempted_break = False

        # Check if achiever has picked up a key and store it
        self._check_and_store_achiever_keys(obs)

        # Enhanced strategy based on achiever and door discovery state
        achiever_visible = self._is_achiever_visible(obs)
        
        # Phase 1: Exploration or following achiever
        if self.phase == 1:
            if not achiever_visible:
                # Achiever not found, execute exploration mode
                return self._clockwise_exploration()
            elif not self.achiever_has_key:
                # Achiever found but has no key, follow achiever
                return self._follow_achiever(obs)
            else:
                # Achiever has key, move to inference phase
                self.phase = 2
                return self._handle_phase_2_enhanced(obs)

        # Phase 2: Enhanced navigation with exploration fallback
        elif self.phase == 2:
            return self._handle_phase_2_enhanced(obs)

        # Phase 3: Break the door
        elif self.phase == 3:
            self.just_attempted_break = True
            self.last_action = 5
            return 5  # Break action

        return 4  # Default: stay
        
    def _is_achiever_visible(self, obs):
        """Check if achiever is visible in current observation"""
        return obs and "achiever_pos" in obs and obs["achiever_pos"] is not None
        
    def _clockwise_exploration(self):
        """Execute clockwise exploration pattern with proper wall following"""
        # Use the base class clockwise wall-following logic
        return self._explore_action()
                
    def _follow_achiever(self, obs):
        """Follow achiever when they have no key"""
        if not self._is_achiever_visible(obs) or self.agent_pos is None:
            return 4  # Stay if can't see achiever
            
        achiever_pos = obs["achiever_pos"]
        
        # Move towards achiever using simple navigation
        dx = achiever_pos[0] - self.agent_pos[0] 
        dy = achiever_pos[1] - self.agent_pos[1]
        
        # Prioritize larger difference
        if abs(dx) > abs(dy):
            if dx > 0:
                return 1  # right
            else:
                return 3  # left
        else:
            if dy > 0:
                return 2  # down
            else:
                return 0  # up
                
    def _handle_phase_2_enhanced(self, obs):
        """Enhanced phase 2: infer target with exploration fallback"""
        # Check if achiever is still visible
        if not self._is_achiever_visible(obs):
            # Achiever lost, return to exploration mode
            self.phase = 1
            return self._clockwise_exploration()
            
        # If we don't have a target yet, infer it from current observed key
        if self.target_color is None:
            self._infer_target_from_observed_keys(obs)
            
        # Check if we have the door position for inferred target
        if self.target_color and self.target_color not in self.memory['door_positions']:
            # Door not found in memory, but check if we're standing on a door first
            current_pos = tuple(self.blocker_pos)
            
            # Check if we're standing on a door directly by examining door positions from observations  
            # Handle both full observation (door_positions) and partial observation (blocker_visible_doors)
            env_door_positions = None
            if obs and "door_positions" in obs:
                # Full observation mode
                env_door_positions = obs["door_positions"]
            elif obs and "blocker_visible_doors" in obs:
                # Partial observation mode - use visible doors
                env_door_positions = obs["blocker_visible_doors"]
            
            if env_door_positions:
                # Check if current position matches any door position
                for color, door_pos in env_door_positions.items():
                    if current_pos == tuple(door_pos):
                        # We're standing on a door! Break it when we have an inferred target
                        self.phase = 3
                        return 5  # Break action
            
            # Door not found in memory and not standing on one, switch to exploration mode
            return self._clockwise_exploration()
            
        # Ensure we have a valid target before proceeding
        if self.target_pos is None and self.target_color:
            # Get position from memory 
            if self.target_color in self.memory['door_positions']:
                self.target_pos = self.memory['door_positions'][self.target_color]
            else:
                # Door position not in memory, but check if we're standing on a door
                current_pos = tuple(self.blocker_pos)
                
                # Check if we're standing on a door directly by examining door positions from observations  
                # Handle both full observation (door_positions) and partial observation (blocker_visible_doors)
                env_door_positions = None
                if obs and "door_positions" in obs:
                    # Full observation mode
                    env_door_positions = obs["door_positions"]
                elif obs and "blocker_visible_doors" in obs:
                    # Partial observation mode - use visible doors
                    env_door_positions = obs["blocker_visible_doors"]
                
                if env_door_positions:
                    # Check if current position matches any door position
                    for color, door_pos in env_door_positions.items():
                        if current_pos == tuple(door_pos):
                            # We're standing on a door! Break it when we have an inferred target
                            self.phase = 3
                            return 5  # Break action
                
                # Door position not in memory and not standing on one, explore more
                return self._clockwise_exploration()
                
        if self.target_pos is None:
            return 4  # Stay if we couldn't infer target
            
        # If we're at the target door, break it
        if self._at_target_door():
            self.phase = 3
            return 5  # Break action
            
        # Navigate to target door using value iteration
        return self._navigate_to_door_with_value_iteration(obs)

    def _handle_phase_2(self, obs):
        """Handle phase 2: infer target and navigate to it using value iteration."""
        # If we don't have a target yet, infer it from current observed key
        if self.target_color is None:
            self._infer_target_from_observed_keys(obs)

        # Ensure we have a valid target before proceeding
        if self.target_pos is None:
            return 4  # Stay if we couldn't infer target

        # If we're at the target door, break it
        if self._at_target_door():
            self.phase = 3
            return 5  # Break action

        # Navigate to target door using value iteration
        return self._navigate_to_door_with_value_iteration(obs)

    def _at_target_door(self):
        """Check if blocker is at the target door position."""
        if self.target_pos is None:
            return False
        # Ensure both positions are tuples for comparison
        return tuple(self.blocker_pos) == tuple(self.target_pos)

    def _navigate_to_door_with_value_iteration(self, obs=None):
        """Navigate to target door using value iteration."""
        return self._navigate_with_value_iteration(self.target_pos, obs)

    def _find_door_position_from_obs(self, color, obs):
        """Find position of door with given color from observations."""
        # Use door positions from observations
        if obs and "door_positions" in obs:
            return obs["door_positions"].get(color, None)
        return None

    def set_env(self, env):
        """Set environment reference for action decisions (legacy method)."""
        # This method is now deprecated since we use observations
        pass

    def reset(self):
        """Reset agent state for new episode."""
        super().reset()
        # Inferred target
        self.target_color = None
        self.target_pos = None
        self.achiever_has_key = False

        # Navigation state
        self.path_to_door = []
        self.current_path_index = 0
        self.blocker_pos = None
        self.achiever_pos = None

        # Phase tracking
        self.phase = 1

        # Reset multi-attempt state
        self.observed_keys.clear()
        self.current_key_index = 0
        self.just_attempted_break = False
        self.last_action = None

        # Keep width/height since they don't change between episodes

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

            # Update achiever_has_key flag for backward compatibility
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

            # Find position of inferred door
            self.target_pos = self._find_door_position_from_obs(self.target_color, obs)