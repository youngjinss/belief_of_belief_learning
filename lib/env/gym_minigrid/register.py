try:
    from gymnasium.envs.registration import register as gym_register
except ImportError:
    from gym.envs.registration import register as gym_register

env_list = []


def register(id, entry_point, reward_threshold=0.95):
    assert id.startswith("MiniGrid-")
    assert id not in env_list

    # Register the environment with both gym and gymnasium if available
    gym_register(id=id, entry_point=entry_point, reward_threshold=reward_threshold)

    # Try to also register with gymnasium if it's available and different from gym
    try:
        import gymnasium

        if (
            hasattr(gymnasium.envs.registration, "register")
            and gymnasium.envs.registration.register != gym_register
        ):
            gymnasium.envs.registration.register(
                id=id, entry_point=entry_point, reward_threshold=reward_threshold
            )
    except ImportError:
        pass

    # Add the environment to the set
    env_list.append(id)
