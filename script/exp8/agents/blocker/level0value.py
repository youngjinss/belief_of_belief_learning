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


class Level0ValueBlocker(BaseValueAgent):
    """
    Level-0 Value-based Blocker Agent for AchieverBlocker environment.

    Strategy (Random Selection with Inference Correction):
    1. Randomly select target door from discovered doors in memory
    2. Navigate to target door using value iteration planning  
    3. Use break action (5) to attempt breaking when at door position
    4. If game continues (wrong door), try to infer correct target from achiever's keys
    5. If inference fails, select randomly from remaining untried doors and repeat

    Uses BaseValueAgent planning with random door selection and inference-based correction after failures.
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
        """
        Initialize Level-0 value-based blocker agent.
        """
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

        # Blocker-specific attributes
        self.target_inferred_color = None
        self.target_door_pos = None
        self.path_to_door = []
        self.current_path_index = 0
        self.blocker_pos = None
        self.achiever_pos = None
        self.target_selected = False

        # Remove stay probability - use direct value iteration decisions

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

    def get_action(self, obs):
        """
        Level0ValueBlocker with random selection strategy:
        
        - Randomly select doors from discovered doors
        - Navigate to selected door and attempt break action  
        - After failed attempt, infer correct target from achiever's keys
        - Mark failed doors and never retry them
        """
        if obs is None:
            return 4  # Stay if no observation

        # Update internal state from observations  
        self.update_observation(obs)

        # Check if we just attempted to break and game is still continuing
        if self.just_attempted_break:
            # Game didn't end, so we broke the wrong door
            # Mark current target as tried
            if self.target_inferred_color:
                self.tried_doors.add(self.target_inferred_color)
            
            # After failed attempt, try to infer correct target from achiever's keys
            inferred_target = self._infer_achiever_target(obs)
            if inferred_target and inferred_target not in self.tried_doors:
                # Use inference to correct our target selection
                self.target_inferred_color = inferred_target
                if inferred_target in self.memory['door_positions']:
                    self.target_door_pos = self.memory['door_positions'][inferred_target]
                    self.target_selected = True
                    if os.getenv('DEBUG_MODE'):
                        print(f"DEBUG: After failed attempt, inferred correct target: {inferred_target}")
                else:
                    self._reset_for_new_attempt()
            else:
                # No valid inference available, select randomly from remaining doors
                self._reset_for_new_attempt()
            
            self.just_attempted_break = False

        # DEBUG: Print relevant info
        if os.getenv('DEBUG_MODE'):
            obs_blocker_pos = obs.get('blocker_pos', None) if obs else None
            print(f"DEBUG: Level0ValueBlocker self.blocker_pos={tuple(self.blocker_pos) if self.blocker_pos else None}, obs_blocker_pos={obs_blocker_pos}")
            if obs and "achiever_keys" in obs:
                print(f"DEBUG: achiever_keys = {obs['achiever_keys']}")

        # Check if we're at a door position and should break it
        should_check_door_breaking = True  # Level0 always checks door breaking when at door position
        
        if should_check_door_breaking:
            # Use position from obs to be accurate
            if obs and "blocker_pos" in obs:
                current_pos = tuple(obs["blocker_pos"])
            else:
                current_pos = tuple(self.blocker_pos) if self.blocker_pos else None
            
            if current_pos and os.getenv('DEBUG_MODE'):
                print(f"DEBUG: Checking door at position {current_pos}")
            
            # Check if we're standing on a door directly by examining door positions from observations
            env_door_positions = None
            if obs and "door_positions" in obs:
                # Full observation mode
                env_door_positions = obs["door_positions"]
                if os.getenv('DEBUG_MODE'):
                    print(f"DEBUG: Using full obs door_positions = {env_door_positions}")
            elif obs and "blocker_visible_doors" in obs:
                # Partial observation mode - use visible doors
                env_door_positions = obs["blocker_visible_doors"]
                if os.getenv('DEBUG_MODE'):
                    print(f"DEBUG: Using partial obs blocker_visible_doors = {env_door_positions}")
            
            if env_door_positions and current_pos:
                # Check if current position matches any door position
                for color, door_pos in env_door_positions.items():
                    if os.getenv('DEBUG_MODE'):
                        print(f"DEBUG: Checking if {current_pos} == {tuple(door_pos)} (color {color})")
                    if current_pos == tuple(door_pos):
                        # We're standing on a door! Break it (random selection strategy)
                        if os.getenv('DEBUG_MODE'):
                            print(f"DEBUG: *** BREAKING DOOR *** at {current_pos} color {color} (random selection)")
                        self.just_attempted_break = True
                        self.last_action = 5
                        return 5  # Break action
            else:
                if os.getenv('DEBUG_MODE'):
                    print(f"DEBUG: No door positions available in observations")
            
            # Fallback: check memory-based door positions
            door_positions = self.memory.get('door_positions', {})
            if os.getenv('DEBUG_MODE'):
                print(f"DEBUG: Fallback memory door_positions = {door_positions}")
            if current_pos:
                for color, door_pos in door_positions.items():
                    if current_pos == tuple(door_pos):
                        # We're at a door position in memory! Break it  
                        if os.getenv('DEBUG_MODE'):
                            print(f"DEBUG: *** BREAKING DOOR (fallback) *** at {current_pos} color {color} (random selection)")
                        self.just_attempted_break = True
                        self.last_action = 5
                        return 5  # Break action

        # Select target door (random selection with inference correction after failure) AFTER door breaking check
        if not self.target_selected:
            self._select_target_door_randomly(obs)

        # Set preferred door color for base class target finding
        if self.target_inferred_color:
            self._preferred_door_color = self.target_inferred_color

        # Use base class act method for strategy coordination and clockwise exploration
        action = self.act(obs)
        self.last_action = action
        return action
        
    def _infer_achiever_target(self, obs):
        """
        Infer achiever's target door color from observations.
        Uses heuristics based on achiever's collected keys.
        If multiple keys collected, infers based on most recent key pickup.
        """
        if obs is None:
            return None
            
        # Check achiever's keys to infer target
        if "achiever_keys" in obs:
            achiever_keys = obs["achiever_keys"]
            color_map = ["red", "green", "blue", "yellow"]
            
            # Collect all keys the achiever has
            collected_keys = []
            for i, has_key in enumerate(achiever_keys):
                if has_key > 0 and i < len(color_map):
                    collected_keys.append(color_map[i])
            
            if collected_keys:
                # Simple heuristic: if only one key, that's the target
                if len(collected_keys) == 1:
                    return collected_keys[0]
                
                # If multiple keys, use last one (most recent pickup) as inference
                # This is a reasonable heuristic for partial observation
                return collected_keys[-1]  # Return last collected key
        
        return None
    

    def _select_target_door_randomly(self, obs):
        """Select target door color randomly from discovered doors in memory."""
        # Get doors discovered in memory
        discovered_doors = set(self.memory['door_positions'].keys())
        
        # Get remaining doors that haven't been tried
        remaining_doors = list(discovered_doors - self.tried_doors)
        
        if not remaining_doors:
            # All discovered doors have been tried
            if len(self.tried_doors) > 0:
                # Reset tried doors and start over with discovered doors
                self.tried_doors.clear()
                remaining_doors = list(discovered_doors)
            else:
                # No doors discovered yet, cannot select target
                self.target_selected = False
                return

        if remaining_doors:
            # Randomly select a door color from remaining doors
            self.target_inferred_color = np.random.choice(remaining_doors)
            self.target_door_pos = self.memory['door_positions'][self.target_inferred_color]
            self.target_selected = True
            
            if os.getenv('DEBUG_MODE'):
                print(f"DEBUG: Randomly selected target door: {self.target_inferred_color} at {self.target_door_pos}")
        else:
            self.target_selected = False

    def _select_random_target_door(self, obs):
        """Legacy method - redirect to new random selection method."""
        return self._select_target_door_randomly(obs)

    def _reset_for_new_attempt(self):
        """Reset targeting state for a new attempt."""
        self.target_selected = False
        self.target_inferred_color = None
        self.target_door_pos = None
        self.path_to_door = []
        self.current_path_index = 0

    def _find_door_position_from_obs(self, color, obs):
        """Find position of door with given color from observations."""
        # Use door positions from observations
        if obs and "door_positions" in obs:
            return obs["door_positions"].get(color, None)
        return None

    def _at_target_door(self):
        """Check if blocker is at the target door position."""
        if self.target_door_pos is None:
            return False
        return self.blocker_pos == self.target_door_pos

    @property
    def target_door_color(self):
        """
        Get the current target door color for interaction checking.
        Uses target_inferred_color for Level0ValueBlocker.

        Returns:
            str: Current target door color
        """
        return self.target_inferred_color

    def set_env(self, env):
        """Set environment reference for action decisions (legacy method)."""
        # This method is now deprecated since we use observations
        pass

    def reset(self):
        """Reset agent state for new episode."""
        super().reset()
        self.target_inferred_color = None
        self.target_door_pos = None
        self.path_to_door = []
        self.current_path_index = 0
        self.blocker_pos = None
        self.achiever_pos = None
        self.target_selected = False
        # Reset multi-attempt state
        self.tried_doors.clear()
        self.just_attempted_break = False
        self.last_action = None
        # Keep width/height since they don't change between episodes