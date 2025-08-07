"""
Base class for V2 environments with shared partial observation functionality.
Both AchieverBlockerEnvV2 and KeyDoorEnvV2 inherit from this base class.
"""
from ..minigrid import *
import numpy as np


class BaseEnvV2(MiniGridEnv):
    """
    Base class for V2 environments with partial observation support.
    
    Provides shared functionality:
    - Observability mode handling (full/partial)  
    - Shared partial observation logic
    - Optimized visibility computation
    """
    
    def __init__(self, observability="full", partial_view_size=7, **kwargs):
        """
        Initialize base V2 environment.
        
        Args:
            observability: "full" or "partial" observation mode
            partial_view_size: Size of partial view window
            **kwargs: Additional arguments passed to MiniGridEnv
        """
        self.observability = observability
        self.partial_view_size = partial_view_size
        
        # Set observation parameters based on observability mode
        if observability == "partial":
            see_through_walls = False
            agent_view_size = partial_view_size
        else:  # full
            see_through_walls = True
            agent_view_size = kwargs.get('grid_size', 9)  # Use grid size for full view
            
        # Update kwargs with observation settings
        kwargs['see_through_walls'] = see_through_walls
        kwargs['agent_view_size'] = agent_view_size
        
        super().__init__(**kwargs)
    
    def _get_observations(self):
        """Generate observations based on observability mode"""
        if self.observability == "partial":
            return self._get_partial_observations()
        else:
            # Skip visibility computation in full observation mode
            return self._get_full_observations()
    
    def _filter_visible_objects(self, positions_dict, vis_mask):
        """
        Filter object positions based on visibility mask.
        
        Args:
            positions_dict: Dictionary of {color: (x, y)} positions
            vis_mask: Visibility mask from gen_obs_grid()
            
        Returns:
            Dictionary of visible objects {color: (x, y)}
        """
        visible_objects = {}
        
        for color, pos in positions_dict.items():
            if pos is not None:
                rel_coords = self.relative_coords(*pos)
                if rel_coords is not None:
                    vx, vy = rel_coords
                    if vis_mask[vx, vy]:
                        visible_objects[color] = pos
                        
        return visible_objects
    
    def _check_agent_visibility(self, target_pos, vis_mask):
        """
        Check if target agent position is visible.
        
        Args:
            target_pos: (x, y) position of target agent
            vis_mask: Visibility mask from gen_obs_grid()
            
        Returns:
            Tuple of (is_visible: bool, visible_pos: np.array or None)
        """
        rel_coords = self.relative_coords(*target_pos)
        if rel_coords is not None:
            vx, vy = rel_coords
            if vis_mask[vx, vy]:
                return True, target_pos.astype(np.int32)
        
        return False, None
    
    # Abstract methods that must be implemented by subclasses
    def _get_full_observations(self):
        """Generate full observations (must be implemented by subclass)"""
        raise NotImplementedError("Subclass must implement _get_full_observations")
    
    def _get_partial_observations(self):
        """Generate partial observations (must be implemented by subclass)"""  
        raise NotImplementedError("Subclass must implement _get_partial_observations")