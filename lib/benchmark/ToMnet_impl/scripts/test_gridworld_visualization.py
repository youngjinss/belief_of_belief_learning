#!/usr/bin/env python3
"""Test script for GridWorld visualization"""

import numpy as np
import matplotlib.pyplot as plt
from environment import GridWorld, SIZE, MAX_WALLS, MAX_STEPS
import random


def test_static_visualization():
    """Test static visualization of GridWorld"""
    print("Testing GridWorld Static Visualization")
    print("=" * 50)

    # Create environment
    env = GridWorld(size=SIZE, max_walls=MAX_WALLS, max_steps=MAX_STEPS)

    # Create multiple random environments
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    for i in range(4):
        env.reset()
        env.visualize(ax=axes[i], title=f"Random Environment {i+1}")

        # Print environment info
        print(f"\nEnvironment {i+1}:")
        print(f"- Agent position: {env.agent_pos}")
        print(f"- Number of walls: {np.sum(env.walls)}")
        print(f"- Object positions: {env.get_object_positions()}")

    plt.tight_layout()
    plt.savefig("gridworld_examples.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved visualization to: gridworld_examples.png")
    plt.show()


def test_episode_visualization():
    """Test episode visualization with random actions"""
    print("\n\nTesting GridWorld Episode Visualization")
    print("=" * 50)

    # Create environment
    env = GridWorld(size=SIZE, max_walls=MAX_WALLS, max_steps=MAX_STEPS)
    env.reset()
    
    # Save initial state for animation
    initial_state = env.copy()
    
    # Generate random actions for an episode
    actions = []
    states_info = []

    print("Initial state:")
    print(env.render())
    states_info.append(
        {
            "agent_pos": env.agent_pos,
            "step": env.step_count,
            "consumed": list(env.consumed_objects),
        }
    )

    # Run episode
    done = False
    while not done:
        # Random action with bias towards moving (less staying)
        action = random.choices([0, 1, 2, 3, 4], weights=[2, 2, 2, 2, 1])[0]
        actions.append(action)

        state, reward, done, info = env.step(action)

        action_names = ["Up", "Down", "Left", "Right", "Stay"]
        states_info.append(
            {
                "action": action_names[action],
                "agent_pos": env.agent_pos,
                "step": env.step_count,
                "reward": reward,
                "consumed": list(env.consumed_objects),
                "done": done,
            }
        )

    # Print episode summary
    print(f"\nEpisode Summary:")
    print(f"- Total steps: {len(actions)}")
    print(
        f"- Actions taken: {[['Up', 'Down', 'Left', 'Right', 'Stay'][a] for a in actions]}"
    )
    print(f"- Objects consumed: {env.consumed_objects}")
    print(f"- Episode done: {done}")

    # Visualize final state
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    env.visualize(ax=ax, title="Final State")
    plt.savefig("result/gridworld_final_state.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved final state to: gridworld_final_state.png")
    plt.show()

    # Create animation
    print("\nCreating episode animation...")
    # Restore initial state for animation
    env_for_animation = initial_state.copy()
    animation = env_for_animation.animate_episode(actions, save_path="result/gridworld_episode.gif")
    print("Animation saved to: gridworld_episode.gif")

    return actions, states_info


def test_multiple_episodes():
    """Test multiple episodes and collect statistics"""
    print("\n\nTesting Multiple Episodes")
    print("=" * 50)

    env = GridWorld(size=SIZE, max_walls=MAX_WALLS, max_steps=MAX_STEPS)

    # Run multiple episodes
    n_episodes = 10
    episode_stats = []

    for ep in range(n_episodes):
        env.reset()
        done = False
        steps = 0
        consumed = []

        while not done and steps < MAX_STEPS:
            action = random.choice([0, 1, 2, 3, 4])
            _, _, done, info = env.step(action)
            steps += 1

            if info["consumed_object"]:
                consumed = info["consumed_object"]

        episode_stats.append(
            {
                "episode": ep + 1,
                "steps": steps,
                "consumed": consumed,
                "success": len(consumed) > 0,
            }
        )

    # Print statistics
    print("\nEpisode Statistics:")
    success_rate = sum(1 for stat in episode_stats if stat["success"]) / n_episodes
    avg_steps = np.mean([stat["steps"] for stat in episode_stats])

    print(f"- Success rate: {success_rate:.1%}")
    print(f"- Average steps: {avg_steps:.1f}")
    print(f"- Successful episodes:")
    for stat in episode_stats:
        if stat["success"]:
            print(
                f"  Episode {stat['episode']}: consumed object {stat['consumed']} in {stat['steps']} steps"
            )


def test_trajectory_visualization():
    """Visualize agent trajectory over an episode"""
    print("\n\nTesting Trajectory Visualization")
    print("=" * 50)

    env = GridWorld(size=SIZE, max_walls=MAX_WALLS, max_steps=MAX_STEPS)
    initial_state = env.reset()

    # Store trajectory
    trajectory = [env.agent_pos]
    actions_taken = []

    # Run episode with semi-random policy (prefer moving towards objects)
    done = False
    while not done and len(actions_taken) < MAX_STEPS:
        # Get object positions
        obj_positions = env.get_object_positions()

        if obj_positions and random.random() < 0.7:  # 70% chance to move towards object
            # Move towards nearest object
            target = min(
                obj_positions,
                key=lambda p: abs(p[0] - env.agent_pos[0])
                + abs(p[1] - env.agent_pos[1]),
            )

            # Choose action towards target
            dy = target[0] - env.agent_pos[0]
            dx = target[1] - env.agent_pos[1]

            if abs(dy) > abs(dx):
                action = 0 if dy < 0 else 1  # Up or Down
            else:
                action = 2 if dx < 0 else 3  # Left or Right
        else:
            action = random.choice([0, 1, 2, 3, 4])

        actions_taken.append(action)
        _, _, done, _ = env.step(action)
        trajectory.append(env.agent_pos)

    # Visualize trajectory
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Draw final state
    env.visualize(ax=ax, title="Agent Trajectory")

    # Overlay trajectory
    trajectory = np.array(trajectory)
    ax.plot(
        trajectory[:, 1], trajectory[:, 0], "w-", linewidth=2, alpha=0.7, label="Path"
    )
    ax.plot(trajectory[:, 1], trajectory[:, 0], "yo", markersize=4, alpha=0.5)

    # Mark start and end
    ax.plot(trajectory[0, 1], trajectory[0, 0], "go", markersize=10, label="Start")
    ax.plot(trajectory[-1, 1], trajectory[-1, 0], "ro", markersize=10, label="End")

    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1))

    plt.tight_layout()
    plt.savefig("gridworld_trajectory.png", dpi=150, bbox_inches="tight")
    print(f"Saved trajectory visualization to: gridworld_trajectory.png")
    print(f"- Path length: {len(trajectory)} steps")
    print(f"- Objects consumed: {env.consumed_objects}")
    plt.show()


if __name__ == "__main__":
    # Test all visualization functions
    # test_static_visualization()
    test_episode_visualization()
    # test_multiple_episodes()
    # test_trajectory_visualization()

    print("\n" + "=" * 50)
    print("All visualization tests completed!")
    print("Generated files:")
    print("- gridworld_examples.png: Multiple random environment layouts")
    print("- gridworld_final_state.png: Final state after an episode")
    print("- gridworld_episode.gif: Animated episode")
    print("- gridworld_trajectory.png: Agent trajectory visualization")
