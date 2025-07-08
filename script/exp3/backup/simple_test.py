import sys
import os

# Add gym_minigrid to Python path at the beginning
gym_minigrid_path = os.path.join(os.path.dirname(__file__), "../../lib/env")
sys.path.insert(0, gym_minigrid_path)

# Import and test
try:
    import gym_minigrid

    print("✓ gym_minigrid imported successfully")
    print(f"gym_minigrid path: {gym_minigrid.__file__}")

    # Check if our environments are registered
    try:
        from gym_minigrid.register import env_list

        print(f"Registered environments: {env_list}")
    except ImportError:
        print("Could not import env_list from gym_minigrid.register")

    # Try to create environment directly
    from gym_minigrid.envs.keydoor import KeyDoor5x5Env

    env = KeyDoor5x5Env()
    print("✓ KeyDoor5x5Env created successfully")

    # Test reset
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        obs, info = reset_result
    else:
        obs = reset_result
        info = {}
    print(f"✓ Environment reset successful. Mission: {env.mission}")

    # Test step
    obs, reward, terminated, truncated, info = env.step(0)  # up action
    print(f"✓ Step successful. Reward: {reward}")

    env.close()
    print("✓ All tests passed!")

except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback

    traceback.print_exc()
