#!/usr/bin/env python3
"""Test script for GridWorld visualization with Goal-Directed Agent"""

import numpy as np
import matplotlib.pyplot as plt
from environment import GridWorld, SIZE, MAX_WALLS, MAX_STEPS
from agents import GoalDirectedAgent
import random


def test_episode_visualization():
    """Test episode visualization with goal-directed agent"""
    print("\n\nTesting GridWorld Episode Visualization with Goal-Directed Agent")
    print("=" * 50)

    # Create environment
    env = GridWorld(size=SIZE, max_walls=MAX_WALLS, max_steps=MAX_STEPS)
    env.reset()

    # Create goal-directed agent with specific reward preferences
    # Higher reward for object 1, decreasing for objects 2, 3, 4
    rewards = np.array([1.0, 0.5, 0.2, 0.1])
    agent = GoalDirectedAgent(
        rewards=rewards,
        movement_cost=0.01,
        wall_penalty=0.05,
        gamma=0.99
    )

    # Plan optimal policy for the current environment
    print("Planning optimal policy...")
    agent.plan(env, max_iterations=1000)
    print(f"Value iteration converged: {agent.converged}")
    print(f"Agent prefers object type: {agent.preferred_object}")

    # Save initial state for animation
    initial_state = env.copy()

    # Generate goal-directed actions for an episode
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

    # Run episode with goal-directed agent
    done = False
    while not done and len(actions) < MAX_STEPS:
        # Get current state
        state = env.get_state()
        
        # Get action from goal-directed agent
        action = agent.act(state, env)
        actions.append(action)

        # Get action probabilities for analysis
        action_probs = agent.get_action_probabilities(state, env)

        # Step environment
        next_state, reward, done, info = env.step(action)

        action_names = ["Up", "Down", "Left", "Right", "Stay"]
        states_info.append(
            {
                "action": action_names[action],
                "action_probs": action_probs,
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
    print(f"- Agent reward preferences: {rewards}")
    print(f"- Preferred object type: {agent.preferred_object}")
    print(
        f"- Actions taken: {[['Up', 'Down', 'Left', 'Right', 'Stay'][a] for a in actions]}"
    )
    print(f"- Objects consumed: {env.consumed_objects}")
    print(f"- Episode done: {done}")
    if env.consumed_objects:
        consumed_type = env.consumed_objects[0]
        print(f"- Consumed object reward: {rewards[consumed_type-1]:.1f}")

    # Visualize final state
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    env.visualize(ax=ax, title=f"Goal-Directed Agent (prefers obj {agent.preferred_object})")
    plt.savefig("result/gridworld_goal_directed_final.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved final state to: result/gridworld_goal_directed_final.png")
    plt.show()

    # Create animation
    print("\nCreating episode animation...")
    # Restore initial state for animation
    env_for_animation = initial_state.copy()
    animation = env_for_animation.animate_episode(
        actions, save_path="result/gridworld_goal_directed_episode.gif"
    )
    print("Animation saved to: result/gridworld_goal_directed_episode.gif")

    return actions, states_info


def test_static_visualization_with_policy():
    """Test static visualization showing agent's policy and value function"""
    print("\n\nTesting GridWorld Static Visualization with Policy")
    print("=" * 50)

    # Create environment
    env = GridWorld(size=SIZE, max_walls=MAX_WALLS, max_steps=MAX_STEPS)
    env.reset()

    # Create goal-directed agent with different reward preferences
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.flatten()

    reward_configs = [
        [1.0, 0.1, 0.1, 0.1],  # Strongly prefers object 1
        [0.1, 1.0, 0.1, 0.1],  # Strongly prefers object 2
        [0.1, 0.1, 1.0, 0.1],  # Strongly prefers object 3
        [0.1, 0.1, 0.1, 1.0],  # Strongly prefers object 4
    ]

    for i, rewards in enumerate(reward_configs):
        # Create agent with specific preferences
        agent = GoalDirectedAgent(
            rewards=np.array(rewards),
            movement_cost=0.01,
            wall_penalty=0.05,
            gamma=0.99
        )

        # Plan policy
        agent.plan(env)

        # Visualize environment
        env.visualize(ax=axes[i], title=f"Prefers Object {agent.preferred_object} (reward={rewards[agent.preferred_object-1]:.1f})")

        # Overlay policy arrows (simplified)
        for row in range(env.size):
            for col in range(env.size):
                if not env.walls[row, col] and env.objects[row, col] == 0 and (row, col) != env.agent_pos:
                    # Get action probabilities for this position
                    mock_state = env.get_state()
                    mock_state[row, col, 5] = 1.0  # Place agent at this position
                    mock_state[env.agent_pos[0], env.agent_pos[1], 5] = 0.0  # Remove from original position
                    
                    action_probs = agent.get_action_probabilities(mock_state, env)
                    best_action = np.argmax(action_probs)
                    
                    # Draw arrow for best action
                    if best_action < 4:  # Not "stay"
                        delta = env.actions[best_action]
                        if action_probs[best_action] > 0.3:  # Only show confident actions
                            axes[i].arrow(col, row, delta[1]*0.3, -delta[0]*0.3, 
                                        head_width=0.1, head_length=0.1, 
                                        fc='white', ec='white', alpha=0.7)

    plt.tight_layout()
    plt.savefig("result/gridworld_policy_comparison.png", dpi=150, bbox_inches="tight")
    print(f"Saved policy comparison to: result/gridworld_policy_comparison.png")
    plt.show()


def test_trajectory_visualization():
    """Visualize agent trajectory with value function overlay"""
    print("\n\nTesting Trajectory Visualization with Value Function")
    print("=" * 50)

    env = GridWorld(size=SIZE, max_walls=MAX_WALLS, max_steps=MAX_STEPS)
    env.reset()

    # Create goal-directed agent
    rewards = np.array([1.0, 0.5, 0.2, 0.1])
    agent = GoalDirectedAgent(rewards=rewards, movement_cost=0.01)
    agent.plan(env)

    # Store trajectory
    trajectory = [env.agent_pos]
    actions_taken = []

    # Run episode
    done = False
    while not done and len(actions_taken) < MAX_STEPS:
        state = env.get_state()
        action = agent.act(state, env)
        actions_taken.append(action)
        
        _, _, done, _ = env.step(action)
        trajectory.append(env.agent_pos)

    # Visualize trajectory with value function
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Left plot: Trajectory
    env.visualize(ax=ax1, title="Goal-Directed Agent Trajectory")
    
    # Overlay trajectory
    trajectory = np.array(trajectory)
    ax1.plot(trajectory[:, 1], trajectory[:, 0], 'w-', linewidth=3, alpha=0.8, label="Path")
    ax1.plot(trajectory[:, 1], trajectory[:, 0], 'yo', markersize=6, alpha=0.7)
    
    # Mark start and end
    ax1.plot(trajectory[0, 1], trajectory[0, 0], 'go', markersize=12, label="Start")
    ax1.plot(trajectory[-1, 1], trajectory[-1, 0], 'ro', markersize=12, label="End")
    ax1.legend(loc="upper left", bbox_to_anchor=(1.05, 1))

    # Right plot: Value function heatmap
    if hasattr(agent, 'value_function'):
        im = ax2.imshow(agent.value_function, cmap='viridis', origin='upper')
        ax2.set_title("Value Function")
        ax2.set_xlabel("Column")
        ax2.set_ylabel("Row")
        
        # Overlay walls and objects
        for i in range(env.size):
            for j in range(env.size):
                if env.walls[i, j]:
                    ax2.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=True, color='red', alpha=0.7))
                elif env.objects[i, j] > 0:
                    ax2.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=True, color='yellow', alpha=0.5))
        
        # Mark agent path
        ax2.plot(trajectory[:, 1], trajectory[:, 0], 'w-', linewidth=2, alpha=0.8)
        ax2.plot(trajectory[0, 1], trajectory[0, 0], 'go', markersize=8)
        ax2.plot(trajectory[-1, 1], trajectory[-1, 0], 'ro', markersize=8)
        
        plt.colorbar(im, ax=ax2, label="Value")

    plt.tight_layout()
    plt.savefig("result/gridworld_goal_trajectory.png", dpi=150, bbox_inches="tight")
    print(f"Saved trajectory visualization to: result/gridworld_goal_trajectory.png")
    print(f"- Path length: {len(trajectory)} steps")
    print(f"- Objects consumed: {env.consumed_objects}")
    print(f"- Agent preferred object: {agent.preferred_object}")
    plt.show()


def test_multiple_episodes():
    """Test multiple episodes with goal-directed agents"""
    print("\n\nTesting Multiple Episodes with Goal-Directed Agents")
    print("=" * 50)

    # Run episodes with different reward preferences
    reward_configs = [
        [1.0, 0.1, 0.1, 0.1],  # Prefers object 1
        [0.1, 1.0, 0.1, 0.1],  # Prefers object 2
        [0.1, 0.1, 1.0, 0.1],  # Prefers object 3
        [0.1, 0.1, 0.1, 1.0],  # Prefers object 4
    ]

    all_stats = []
    n_episodes_per_config = 5

    for config_idx, rewards in enumerate(reward_configs):
        print(f"\nTesting agent that prefers object {np.argmax(rewards) + 1}:")
        
        episode_stats = []
        for ep in range(n_episodes_per_config):
            env = GridWorld(size=SIZE, max_walls=MAX_WALLS, max_steps=MAX_STEPS)
            env.reset()
            
            agent = GoalDirectedAgent(rewards=np.array(rewards))
            agent.plan(env)
            
            done = False
            steps = 0
            consumed = []

            while not done and steps < MAX_STEPS:
                state = env.get_state()
                action = agent.act(state, env)
                _, _, done, info = env.step(action)
                steps += 1

                if info["consumed_object"]:
                    consumed = info["consumed_object"]

            episode_stats.append({
                "config": config_idx,
                "preferred_obj": agent.preferred_object,
                "episode": ep + 1,
                "steps": steps,
                "consumed": consumed,
                "success": len(consumed) > 0,
                "got_preferred": len(consumed) > 0 and consumed[0] == agent.preferred_object
            })

        all_stats.extend(episode_stats)

        # Print config-specific statistics
        success_rate = sum(1 for stat in episode_stats if stat["success"]) / n_episodes_per_config
        preferred_rate = sum(1 for stat in episode_stats if stat["got_preferred"]) / n_episodes_per_config
        avg_steps = np.mean([stat["steps"] for stat in episode_stats])

        print(f"- Success rate: {success_rate:.1%}")
        print(f"- Got preferred object: {preferred_rate:.1%}")
        print(f"- Average steps: {avg_steps:.1f}")

    return all_stats


if __name__ == "__main__":
    # Test all visualization functions
    test_episode_visualization()
    # test_static_visualization_with_policy()
    # test_trajectory_visualization()
    # test_multiple_episodes()

    print("\n" + "=" * 50)
    print("All goal-directed agent visualization tests completed!")
    print("Generated files:")
    print("- result/gridworld_goal_directed_final.png: Final state after goal-directed episode")
    print("- result/gridworld_goal_directed_episode.gif: Animated goal-directed episode")
    print("- result/gridworld_policy_comparison.png: Policy comparison for different preferences")
    print("- result/gridworld_goal_trajectory.png: Trajectory with value function")