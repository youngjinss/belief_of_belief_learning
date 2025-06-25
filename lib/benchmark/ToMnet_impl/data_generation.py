import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Union
import pickle
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

from environment import GridWorld
from agents import (
    RandomAgent,
    GoalDirectedAgent,
    create_random_agents,
    create_goal_directed_agents,
)


class TrajectoryData:
    """Container for trajectory data"""

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.agent_id = None
        self.episode_id = None

    def add_step(self, state: np.ndarray, action: int, reward: float):
        self.states.append(state.copy())
        self.actions.append(action)
        self.rewards.append(reward)

    def to_tensor(self, device: str = "cpu") -> Dict[str, torch.Tensor]:
        """Convert to PyTorch tensors"""
        return {
            "states": torch.tensor(
                np.array(self.states), dtype=torch.float32, device=device
            ),
            "actions": torch.tensor(self.actions, dtype=torch.long, device=device),
            "rewards": torch.tensor(self.rewards, dtype=torch.float32, device=device),
        }


def generate_agent_episodes(args):
    """Helper function to generate episodes for a single agent (for parallel processing)"""
    (
        agent_id,
        agent,
        n_episodes_per_agent,
        grid_size,
        max_walls,
        max_steps,
        experiment_type,
    ) = args

    agent_trajectories = []

    for episode_id in range(n_episodes_per_agent):
        env = GridWorld(grid_size, max_walls, max_steps)
        state = env.reset()

        if experiment_type == "goal_directed":
            agent.plan(env)

        trajectory = TrajectoryData()
        trajectory.agent_id = agent_id
        trajectory.episode_id = episode_id

        if experiment_type == "goal_directed":
            trajectory.env_state = {
                "walls": env.walls.copy(),
                "objects": env.objects.copy(),
                "initial_agent_pos": env.agent_pos,
            }

        # Run episode
        step_count = 0
        while not env.done and step_count < max_steps:
            if experiment_type == "goal_directed":
                action = agent.act(state, env)
            else:
                action = agent.act(state)
            next_state, reward, done, info = env.step(action)

            trajectory.add_step(state.flatten(), action, reward)
            state = next_state
            step_count += 1

            if experiment_type == "goal_directed" and done:
                break

        agent_trajectories.append(trajectory)

    return agent_id, agent_trajectories


class DataGenerator:
    """Generate training and evaluation data for ToMnet experiments"""

    def __init__(self, grid_size: int = 11, max_walls: int = 4, max_steps: int = 31):
        self.grid_size = grid_size
        self.max_walls = max_walls
        self.max_steps = max_steps
        self.state_dim = (
            grid_size * grid_size * 6
        )  # 6 channels: walls, 4 objects, agent

    def generate_random_agent_data(
        self,
        n_agents: int,
        n_episodes_per_agent: int,
        alpha: float,
        min_past: int = 0,
        max_past: int = 10,
        save_path: Optional[str] = None,
        n_workers: Optional[int] = None,
    ) -> Dict:
        """
        Generate data for Figure 3 experiments with random agents

        Args:
            n_agents: Number of agents to create
            n_episodes_per_agent: Episodes per agent
            alpha: Dirichlet concentration parameter
            min_past/max_past: Range for number of past episodes
            save_path: Optional path to save data
            n_workers: Number of parallel workers (default: CPU count)
        """
        print(f"Generating random agent data: {n_agents} agents, alpha={alpha}")

        # Create agents
        agents = create_random_agents(n_agents, alpha)

        # Determine number of workers
        if n_workers is None:
            n_workers = min(cpu_count(), n_agents)
        print(f"Using {n_workers} parallel workers")

        # Prepare arguments for parallel processing
        agent_args = [
            (
                agent_id,
                agent,
                n_episodes_per_agent,
                self.grid_size,
                self.max_walls,
                self.max_steps,
                "random",
            )
            for agent_id, agent in enumerate(agents)
        ]

        # Generate episodes in parallel
        all_agent_trajectories = {}
        with Pool(n_workers) as pool:
            results = list(
                tqdm(
                    pool.imap(generate_agent_episodes, agent_args),
                    total=n_agents,
                    desc="Generating agent data (parallel)",
                )
            )

            for agent_id, trajectories in results:
                all_agent_trajectories[agent_id] = trajectories

        # Create training samples
        all_data = []

        for agent_id in range(n_agents):
            agent = agents[agent_id]
            agent_trajectories = all_agent_trajectories[agent_id]

            # Create training samples with variable past episodes
            for query_episode_id in range(n_episodes_per_agent):
                # Sample number of past episodes
                n_past = np.random.randint(min_past, max_past + 1)

                # Select past episodes (excluding query episode)
                available_episodes = [
                    i for i in range(n_episodes_per_agent) if i != query_episode_id
                ]
                if len(available_episodes) >= n_past:
                    past_episode_ids = np.random.choice(
                        available_episodes, n_past, replace=False
                    )
                else:
                    past_episode_ids = available_episodes

                past_trajectories = [agent_trajectories[i] for i in past_episode_ids]
                query_trajectory = agent_trajectories[query_episode_id]

                # For Figure 3: predict initial action from single state-action pair
                if len(query_trajectory.actions) > 0:
                    sample = {
                        "agent_id": agent_id,
                        "alpha": alpha,
                        "past_trajectories": past_trajectories,
                        "query_state": query_trajectory.states[0],
                        "query_action": query_trajectory.actions[0],
                        "true_policy": agent.get_action_probabilities(None),
                        "n_past": len(past_trajectories),
                    }
                    all_data.append(sample)

        dataset = {
            "data": all_data,
            "meta": {
                "n_agents": n_agents,
                "n_episodes_per_agent": n_episodes_per_agent,
                "alpha": alpha,
                "state_dim": self.state_dim,
                "grid_size": self.grid_size,
            },
        }

        if save_path:
            with open(save_path, "wb") as f:
                pickle.dump(dataset, f)
            print(f"Saved {len(all_data)} samples to {save_path}")

        return dataset

    def generate_goal_directed_agent_data(
        self,
        n_agents: int,
        n_episodes_per_agent: int,
        alpha_reward: float = 0.01,
        high_cost_ratio: float = 0.2,
        min_past: int = 0,
        max_past: int = 10,
        save_path: Optional[str] = None,
        n_workers: Optional[int] = None,
    ) -> Dict:
        """
        Generate data for Figure 5 experiments with goal-directed agents

        Args:
            n_agents: Number of agents to create
            n_episodes_per_agent: Episodes per agent
            alpha_reward: Dirichlet concentration parameter for rewards
            high_cost_ratio: Ratio of high-cost agents
            min_past/max_past: Range for number of past episodes
            save_path: Optional path to save data
            n_workers: Number of parallel workers (default: CPU count)
        """
        print(f"Generating goal-directed agent data: {n_agents} agents")

        # Create agents
        agents = create_goal_directed_agents(n_agents, alpha_reward, high_cost_ratio)

        # Determine number of workers
        if n_workers is None:
            n_workers = min(cpu_count(), n_agents)
        print(f"Using {n_workers} parallel workers")

        # Prepare arguments for parallel processing
        agent_args = [
            (
                agent_id,
                agent,
                n_episodes_per_agent,
                self.grid_size,
                self.max_walls,
                self.max_steps,
                "goal_directed",
            )
            for agent_id, agent in enumerate(agents)
        ]

        # Generate episodes in parallel
        all_agent_trajectories = {}
        with Pool(n_workers) as pool:
            results = list(
                tqdm(
                    pool.imap(generate_agent_episodes, agent_args),
                    total=n_agents,
                    desc="Generating agent data (parallel)",
                )
            )

            for agent_id, trajectories in results:
                all_agent_trajectories[agent_id] = trajectories

        # Create training samples
        all_data = []

        for agent_id in range(n_agents):
            agent = agents[agent_id]
            agent_trajectories = all_agent_trajectories[agent_id]

            # Create training samples
            for query_episode_id in range(n_episodes_per_agent):
                # Sample number of past episodes
                n_past = np.random.randint(min_past, max_past + 1)

                # Select past episodes
                available_episodes = [
                    i for i in range(n_episodes_per_agent) if i != query_episode_id
                ]
                if len(available_episodes) >= n_past:
                    past_episode_ids = np.random.choice(
                        available_episodes, n_past, replace=False
                    )
                else:
                    past_episode_ids = available_episodes

                past_trajectories = [agent_trajectories[i] for i in past_episode_ids]
                query_trajectory = agent_trajectories[query_episode_id]

                # Create samples for each step in query trajectory
                for step_idx in range(len(query_trajectory.actions)):
                    # Reconstruct environment for this episode
                    env_copy = GridWorld(self.grid_size, self.max_walls, self.max_steps)
                    env_copy.walls = query_trajectory.env_state["walls"]
                    env_copy.objects = query_trajectory.env_state["objects"]
                    env_copy.agent_pos = query_trajectory.env_state["initial_agent_pos"]

                    # Get current trajectory up to step_idx
                    current_trajectory = TrajectoryData()
                    for i in range(step_idx):
                        current_trajectory.add_step(
                            query_trajectory.states[i],
                            query_trajectory.actions[i],
                            query_trajectory.rewards[i],
                        )

                    # Get true labels
                    current_state = query_trajectory.states[step_idx]
                    true_action = query_trajectory.actions[step_idx]

                    # Object consumption (which object will be consumed eventually)
                    consumed_objects = np.zeros(4)  # One-hot encoding
                    if "consumed_object" in query_trajectory.__dict__:
                        for obj in query_trajectory.consumed_object:
                            consumed_objects[obj - 1] = 1.0

                    # Successor representation
                    sr = agent.get_successor_representation(env_copy)

                    sample = {
                        "agent_id": agent_id,
                        "rewards": agent.rewards,
                        "movement_cost": agent.movement_cost,
                        "past_trajectories": past_trajectories,
                        "current_trajectory": current_trajectory,
                        "query_state": current_state,
                        "true_action": true_action,
                        "true_consumption": consumed_objects,
                        "true_sr": sr.flatten(),
                        "n_past": len(past_trajectories),
                        "step_idx": step_idx,
                    }
                    all_data.append(sample)

        dataset = {
            "data": all_data,
            "meta": {
                "n_agents": n_agents,
                "n_episodes_per_agent": n_episodes_per_agent,
                "alpha_reward": alpha_reward,
                "high_cost_ratio": high_cost_ratio,
                "state_dim": self.state_dim,
                "grid_size": self.grid_size,
            },
        }

        if save_path:
            with open(save_path, "wb") as f:
                pickle.dump(dataset, f)
            print(f"Saved {len(all_data)} samples to {save_path}")

        return dataset


class ToMnetDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for ToMnet training"""

    def __init__(self, dataset: Dict, experiment_type: str = "figure3"):
        self.data = dataset["data"]
        self.meta = dataset["meta"]
        self.experiment_type = experiment_type
        self.state_dim = self.meta["state_dim"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Process past trajectories
        past_trajectories = self._process_past_trajectories(sample["past_trajectories"])

        if self.experiment_type == "figure3":
            # For Figure 3: simple action prediction
            return {
                "past_trajectories": past_trajectories,
                "current_trajectory": torch.zeros(1, 1, self.state_dim + 5),  # Dummy
                "current_state": torch.tensor(
                    sample["query_state"], dtype=torch.float32
                ),
                "true_actions": torch.tensor(sample["query_action"], dtype=torch.long),
                "agent_id": sample["agent_id"],
                "n_past": sample["n_past"],
            }

        elif self.experiment_type == "figure5":
            # For Figure 5: full predictions
            current_trajectory = self._process_current_trajectory(
                sample["current_trajectory"]
            )

            return {
                "past_trajectories": past_trajectories,
                "current_trajectory": current_trajectory,
                "current_state": torch.tensor(
                    sample["query_state"], dtype=torch.float32
                ),
                "true_actions": torch.tensor(sample["true_action"], dtype=torch.long),
                "true_consumption": torch.tensor(
                    sample["true_consumption"], dtype=torch.float32
                ),
                "true_sr": torch.tensor(sample["true_sr"], dtype=torch.float32),
                "agent_id": sample["agent_id"],
                "n_past": sample["n_past"],
            }

    def _process_past_trajectories(
        self, trajectories: List[TrajectoryData]
    ) -> torch.Tensor:
        """Convert past trajectories to tensor format"""
        if not trajectories:
            # Return dummy tensor if no past trajectories
            return torch.zeros(1, 1, self.state_dim + 5)

        # For simplicity, use first state-action pair from each trajectory
        processed = []
        for traj in trajectories:
            if len(traj.states) > 0 and len(traj.actions) > 0:
                state = traj.states[0]
                action_onehot = np.zeros(5)
                action_onehot[traj.actions[0]] = 1.0
                combined = np.concatenate([state, action_onehot])
                processed.append(combined)

        if not processed:
            return torch.zeros(1, 1, self.state_dim + 5)

        # Shape: (n_past, 1, state_dim + action_dim)
        tensor = torch.tensor(np.array(processed), dtype=torch.float32)
        return tensor.unsqueeze(1)

    def _process_current_trajectory(self, trajectory: TrajectoryData) -> torch.Tensor:
        """Convert current trajectory to tensor format"""
        if len(trajectory.states) == 0:
            return torch.zeros(1, self.state_dim + 5)

        processed = []
        for i in range(len(trajectory.states)):
            state = trajectory.states[i]
            action_onehot = np.zeros(5)
            if i < len(trajectory.actions):
                action_onehot[trajectory.actions[i]] = 1.0
            combined = np.concatenate([state, action_onehot])
            processed.append(combined)

        return torch.tensor(np.array(processed), dtype=torch.float32)


def collate_fn(batch):
    """Custom collate function for variable-length sequences"""
    # Find maximum sequence lengths
    max_n_past = max(item["past_trajectories"].size(0) for item in batch)
    max_current_len = max(item["current_trajectory"].size(0) for item in batch)

    # Pad sequences
    batch_size = len(batch)
    state_action_dim = batch[0]["past_trajectories"].size(-1)
    state_dim = batch[0]["current_state"].size(0)

    # Initialize padded tensors
    past_traj_padded = torch.zeros(batch_size, max_n_past, 1, state_action_dim)
    current_traj_padded = torch.zeros(batch_size, max_current_len, state_action_dim)
    current_states = torch.stack([item["current_state"] for item in batch])
    true_actions = torch.stack([item["true_actions"] for item in batch])

    # Fill padded tensors
    for i, item in enumerate(batch):
        n_past = item["past_trajectories"].size(0)
        current_len = item["current_trajectory"].size(0)

        past_traj_padded[i, :n_past] = item["past_trajectories"]
        current_traj_padded[i, :current_len] = item["current_trajectory"]

    result = {
        "past_trajectories": past_traj_padded,
        "current_trajectory": current_traj_padded,
        "current_state": current_states,
        "true_actions": true_actions,
        "agent_ids": [item["agent_id"] for item in batch],
        "n_past": [item["n_past"] for item in batch],
    }

    # Add additional fields for Figure 5
    if "true_consumption" in batch[0]:
        result["true_consumption"] = torch.stack(
            [item["true_consumption"] for item in batch]
        )
        result["true_sr"] = torch.stack([item["true_sr"] for item in batch])

    return result
