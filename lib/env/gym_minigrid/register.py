try:
    from gymnasium.envs.registration import register as gym_register
except ImportError:
    from gym.envs.registration import register as gym_register

env_list = []


def register(id, entry_point, reward_threshold=0.95):
    assert id.startswith("MiniGrid-")
    
    # Skip if already registered in our list
    if id in env_list:
        return
    
    # Check if already registered in gymnasium registry
    try:
        from gymnasium.envs import registry as gym_registry
        if hasattr(gym_registry, 'all') and id in gym_registry.all():
            env_list.append(id)
            return
    except (ImportError, AttributeError):
        # Try alternative method
        try:
            import gymnasium
            # Use a safer check that won't cause circular imports
            if hasattr(gymnasium, 'envs') and hasattr(gymnasium.envs, 'registry'):
                registry_dict = getattr(gymnasium.envs.registry, 'registry', {})
                if id in registry_dict:
                    env_list.append(id)
                    return
        except (ImportError, AttributeError):
            pass

    # Try to register, catching the warning if it's already registered
    try:
        gym_register(id=id, entry_point=entry_point, reward_threshold=reward_threshold)
    except Exception:
        # If registration fails (likely because it's already registered), just continue
        pass

    # Add the environment to the set
    env_list.append(id)
