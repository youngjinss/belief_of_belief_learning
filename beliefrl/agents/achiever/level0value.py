import os

from beliefrl.agents.value_agent import BaseValueAgent
from beliefrl.env.minigrid import Door, Key
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
        goal_rewards=None,
        grid_width=None,
        grid_height=None,
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
            grid_width=grid_width,
            grid_height=grid_height,
        )

        # Achiever-specific attributes
        # Determine target door color from goal_rewards (highest reward)
        if goal_rewards is None:
            raise ValueError("goal_rewards is required to determine target door color")
        # Get the color with highest reward - this should be lowercase (door color)
        self.target_door_color = max(goal_rewards, key=goal_rewards.get)
        
        # Debug logging to verify target color
        if os.getenv("DEBUG_MODE"):
            print(f"DEBUG: Initialized target_door_color = '{self.target_door_color}' from goal_rewards = {goal_rewards}")
        
        self.collected_keys = set()
        self.strategy_phase = "collect_key"  # "collect_key" or "open_door"

    def _update_agent_position(self, obs):
        """Update achiever position from observations"""
        if os.getenv("DEBUG_MODE") and obs:
            print(f"DEBUG: _update_agent_position called with obs keys: {list(obs.keys())}")
            
        if "achiever_pos" in obs:
            new_pos = tuple(obs["achiever_pos"])
            if os.getenv("DEBUG_MODE"):
                print(f"DEBUG: Found achiever_pos: {new_pos}")
        elif "agent_pos" in obs:
            new_pos = tuple(obs["agent_pos"])
            if os.getenv("DEBUG_MODE"):
                print(f"DEBUG: Found agent_pos: {new_pos}")
        else:
            if os.getenv("DEBUG_MODE"):
                print(f"DEBUG: No position found in obs")
            return

        if new_pos != self.agent_pos:
            if os.getenv("DEBUG_MODE"):
                print(f"DEBUG: Updating agent_pos from {self.agent_pos} to {new_pos}")
            self.agent_pos = new_pos

    def _update_grid_reference(self, obs):
        """Update grid reference from observations"""
        if "achiever" in obs and obs["achiever"] and "image" in obs["achiever"]:
            from beliefrl.env.minigrid import Grid
            self.grid = Grid.decode(obs["achiever"]["image"])

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

        # Check if we have observed the target key
        # Set exploration_mode to False when we know where the target key is
        target_key_in_memory = f"key_{self.target_door_color}" in self.memory
        
        # Check for target key in observations - handle both full and partial observability
        target_key_in_obs = False
        if obs and self.target_door_color:
            # Full observability: check key_positions
            if "key_positions" in obs and self.target_door_color in obs.get("key_positions", {}):
                target_key_in_obs = True
            # Partial observability: check achiever_visible_keys
            elif "achiever_visible_keys" in obs and self.target_door_color in obs.get("achiever_visible_keys", {}):
                target_key_in_obs = True
        
        # DEBUG: Add detailed debugging
        if os.getenv("DEBUG_MODE"):
            print(f"DEBUG: update_observation at pos {self.agent_pos}")
            print(f"DEBUG: target_door_color = {self.target_door_color}")
            print(f"DEBUG: memory keys = {list(self.memory.keys())}")
            print(f"DEBUG: collected_keys = {self.collected_keys}")
            print(f"DEBUG: target_key_in_memory = {target_key_in_memory}")
            if obs:
                print(f"DEBUG: obs has key_positions = {'key_positions' in obs}")
                print(f"DEBUG: obs has achiever_visible_keys = {'achiever_visible_keys' in obs}")
                print(f"DEBUG: obs has door_positions = {'door_positions' in obs}")
                print(f"DEBUG: obs has achiever_visible_doors = {'achiever_visible_doors' in obs}")
                if "key_positions" in obs:
                    print(f"DEBUG: key_positions = {obs['key_positions']}")
                if "achiever_visible_keys" in obs:
                    print(f"DEBUG: achiever_visible_keys = {obs['achiever_visible_keys']}")
                if "door_positions" in obs:
                    print(f"DEBUG: door_positions = {obs['door_positions']}")
                if "achiever_visible_doors" in obs:
                    print(f"DEBUG: achiever_visible_doors = {obs['achiever_visible_doors']}")
            print(f"DEBUG: target_key_in_obs = {target_key_in_obs}")
            print(f"DEBUG: exploration_mode before = {self.exploration_mode}")
        
        # Logic for exploration_mode switching
        if target_key_in_memory or target_key_in_obs:
            self.exploration_mode = False
            if os.getenv("DEBUG_MODE"):
                print(f"DEBUG: *** SWITCHING exploration_mode to False (key discovered) ***")
        else:
            # After collecting key, check for target door
            if self.target_door_color and self.target_door_color in self.collected_keys:
                # Key collected, now look for door
                target_door_in_memory = f"door_{self.target_door_color}" in self.memory
                target_door_in_obs = False
                
                if obs and self.target_door_color:
                    # Full observability: check door_positions
                    if "door_positions" in obs and self.target_door_color in obs.get("door_positions", {}):
                        target_door_in_obs = True
                    # Partial observability: check achiever_visible_doors
                    elif "achiever_visible_doors" in obs and self.target_door_color in obs.get("achiever_visible_doors", {}):
                        target_door_in_obs = True
                
                if os.getenv("DEBUG_MODE"):
                    print(f"DEBUG: target_door_in_memory = {target_door_in_memory}")
                    print(f"DEBUG: target_door_in_obs = {target_door_in_obs}")
                
                if target_door_in_memory or target_door_in_obs:
                    self.exploration_mode = False
                    if os.getenv("DEBUG_MODE"):
                        print(f"DEBUG: *** SWITCHING exploration_mode to False (door discovered) ***")
                else:
                    self.exploration_mode = True
                    if os.getenv("DEBUG_MODE"):
                        print(f"DEBUG: *** SWITCHING exploration_mode to True (need to find door) ***")

    def get_action(self, obs):
        """
        Get the next action for the agent using value iteration with partial observation strategy
        """
        # Update observations and memory first
        self.update_observation(obs)
        self._update_memory(obs, self.agent_pos)

        # Set target door color in base class for consumption penalty
        self.set_target_door_color(self.target_door_color)

        target_key_color = self.target_door_color
        
        # Check if we have the target key from inventory
        achiever_keys_array = obs.get("achiever_keys", [])
        color_map = ["red", "green", "blue", "yellow"]
        self.collected_keys = set()
        for i, has_key in enumerate(achiever_keys_array):
            if has_key > 0 and i < len(color_map):
                self.collected_keys.add(color_map[i])

        if not self.exploration_mode:
            if self.strategy_phase == "collect_key":
                # Check if we already have the target key
                if target_key_color in self.collected_keys:
                    if os.getenv("DEBUG_MODE"):
                        print(f"DEBUG: *** KEY COLLECTED! Switching strategy_phase from 'collect_key' to 'open_door' ***")
                        print(f"DEBUG: target_key_color='{target_key_color}', collected_keys={self.collected_keys}")
                    self.strategy_phase = "open_door"
                    # After changing phase, go to door opening logic
                    return self._open_target_door(target_key_color, obs)
                else:
                    if os.getenv("DEBUG_MODE"):
                        print(f"DEBUG: Still need to collect key. target_key_color='{target_key_color}', collected_keys={self.collected_keys}")
                    # Find and collect the target key
                    return self._collect_target_key(target_key_color, obs)

            elif self.strategy_phase == "open_door":
                # Go to target door and open it
                return self._open_target_door(target_key_color, obs)
            
            # Fallback for unknown strategy_phase when not exploring
            else:
                if os.getenv("DEBUG_MODE"):
                    print(f"DEBUG: Unknown strategy_phase '{self.strategy_phase}', falling back to exploration")
                return self._explore_with_clockwise_pattern()
        else:
            # Fallback: use clockwise exploration pattern
            return self._explore_with_clockwise_pattern()
        
        # Final fallback - should never reach here, but just in case
        if os.getenv("DEBUG_MODE"):
            print(f"DEBUG: WARNING - get_action reached end without returning, using exploration")
        return self._explore_with_clockwise_pattern()

    def _collect_target_key(self, target_key_color, obs=None):
        """Strategy to collect the target key using value iteration or exploration"""
        # Check memory first for discovered key position
        target_key_pos = self.memory.get(f"key_{target_key_color}")
        
        # If not in memory, check current observations (handle both full and partial observability)
        if target_key_pos is None and obs:
            # Full observability: check key_positions
            if "key_positions" in obs:
                target_key_pos = obs["key_positions"].get(target_key_color)
            # Partial observability: check achiever_visible_keys
            elif "achiever_visible_keys" in obs:
                target_key_pos = obs["achiever_visible_keys"].get(target_key_color)
            
            if target_key_pos is not None:
                target_key_pos = tuple(target_key_pos)
                
        # If key position found, switch to value iteration mode
        if target_key_pos is not None:
            self.exploration_mode = False
            # Check if we're already at the key position
            if self.agent_pos == target_key_pos:
                return 4  # Stay - key pickup is automatic
                
            # Use value iteration to navigate to key
            return self._navigate_with_value_iteration(target_key_pos, obs)
        
        # Key not found - use exploration or value iteration based on mode
        if self.exploration_mode:
            return self._explore_with_clockwise_pattern()
        else:
            # If not in exploration mode but key not found, try value iteration with memory
            if self.memory:
                # Try to navigate to any discovered key position
                for color in ["red", "green", "blue", "yellow"]:
                    key_pos = self.memory.get(f"key_{color}")
                    if key_pos and color == target_key_color:
                        return self._navigate_with_value_iteration(key_pos, obs)
            # Fallback to exploration if nothing found
            return self._explore_with_clockwise_pattern()

    def _open_target_door(self, target_door_color, obs=None):
        """Strategy to open the target door using value iteration or exploration"""
        if os.getenv("DEBUG_MODE"):
            print(f"DEBUG: _open_target_door called for color '{target_door_color}'")
            print(f"DEBUG: Current memory keys: {list(self.memory.keys())}")
            
        # Check memory first for discovered door position
        target_door_pos = self.memory.get(f"door_{target_door_color}")
        
        if os.getenv("DEBUG_MODE"):
            print(f"DEBUG: door_{target_door_color} in memory: {target_door_pos}")
        
        # If not in memory, check current observations (handle both full and partial observability)
        if target_door_pos is None and obs:
            # Full observability: check door_positions
            if "door_positions" in obs:
                target_door_pos = obs["door_positions"].get(target_door_color)
            # Partial observability: check achiever_visible_doors
            elif "achiever_visible_doors" in obs:
                target_door_pos = obs["achiever_visible_doors"].get(target_door_color)
            
            if target_door_pos is not None:
                target_door_pos = tuple(target_door_pos)
                if os.getenv("DEBUG_MODE"):
                    print(f"DEBUG: Found door_{target_door_color} in obs at {target_door_pos}")
                
        # If door position found, navigate to it
        if target_door_pos is not None:
            if os.getenv("DEBUG_MODE"):
                print(f"DEBUG: Navigating to door_{target_door_color} at {target_door_pos} using value iteration")
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
        if os.getenv("DEBUG_MODE"):
            print(f"DEBUG: door_{target_door_color} NOT FOUND in memory or obs - falling back to exploration")
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
        # Don't reset target_door_color - it should persist for the entire game
        # self.target_door_color = None  # REMOVED: This was causing target_door_color to be None
        self.collected_keys = set()
        self.strategy_phase = "collect_key"
