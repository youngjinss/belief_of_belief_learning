import numpy as np
from typing import Tuple, List, Dict, Optional
from scipy.special import softmax
from environment import GridWorld

class RandomAgent:
    """
    Random agent with fixed stochastic policy for Figure 3 experiments
    Policy sampled from Dirichlet distribution with concentration parameter alpha
    """
    
    def __init__(self, alpha: float, n_actions: int = 5):
        self.alpha = alpha
        self.n_actions = n_actions
        
        # Sample fixed policy from Dirichlet(alpha)
        # Higher alpha = more stochastic, lower alpha = more deterministic
        alpha_vec = np.full(n_actions, alpha)
        self.policy = np.random.dirichlet(alpha_vec)
        
        # Store most frequent action for visualization
        self.dominant_action = np.argmax(self.policy)
    
    def act(self, state: np.ndarray) -> int:
        """Sample action according to fixed policy"""
        return np.random.choice(self.n_actions, p=self.policy)
    
    def get_action_probabilities(self, state: np.ndarray) -> np.ndarray:
        """Get action probabilities for current state"""
        return self.policy.copy()


class GoalDirectedAgent:
    """
    Goal-directed agent with optimal policy via value iteration for Figure 5
    """
    
    def __init__(self, rewards: np.ndarray, movement_cost: float = 0.01, 
                 wall_penalty: float = 0.05, gamma: float = 0.99):
        """
        Args:
            rewards: Array of shape (n_objects,) with reward for consuming each object
            movement_cost: Cost per movement step
            wall_penalty: Additional penalty for hitting walls
            gamma: Discount factor for value iteration
        """
        self.rewards = rewards  # r_i,a for consuming object a
        self.movement_cost = movement_cost
        self.wall_penalty = wall_penalty
        self.gamma = gamma
        
        # Store preferred object for visualization
        self.preferred_object = np.argmax(rewards) + 1  # +1 because objects are 1-indexed
        
        # Value function and policy will be computed for each environment
        self.value_function = None
        self.policy = None
        self.converged = False
    
    def plan(self, env: GridWorld, max_iterations: int = 1000, 
             convergence_threshold: float = 1e-6) -> None:
        """
        Run value iteration to compute optimal policy for given environment
        """
        size = env.size
        n_actions = env.n_actions
        
        # Initialize value function
        self.value_function = np.zeros((size, size))
        self.policy = np.zeros((size, size, n_actions))
        
        for iteration in range(max_iterations):
            old_values = self.value_function.copy()
            
            # Update value function for each state
            for i in range(size):
                for j in range(size):
                    if env.walls[i, j]:
                        continue  # Skip wall positions
                    
                    state_values = []
                    
                    # Evaluate each action
                    for action in range(n_actions):
                        value = self._evaluate_action(env, (i, j), action)
                        state_values.append(value)
                    
                    # Bellman update
                    self.value_function[i, j] = max(state_values)
                    
                    # Update policy (softmax for some stochasticity)
                    state_values = np.array(state_values)
                    self.policy[i, j] = softmax(state_values / 0.1)  # Temperature = 0.1
            
            # Check convergence
            if np.max(np.abs(self.value_function - old_values)) < convergence_threshold:
                self.converged = True
                break
        
        if not self.converged:
            print(f"Warning: Value iteration did not converge after {max_iterations} iterations")
    
    def _evaluate_action(self, env: GridWorld, pos: Tuple[int, int], action: int) -> float:
        """Evaluate expected value of taking action from position"""
        i, j = pos
        delta = env.actions[action]
        new_pos = (i + delta[0], j + delta[1])
        
        # Base movement cost
        reward = -self.movement_cost
        
        # Check if action leads to wall or out of bounds
        if (new_pos[0] < 0 or new_pos[0] >= env.size or 
            new_pos[1] < 0 or new_pos[1] >= env.size or 
            env.walls[new_pos]):
            # Stay in same position with wall penalty
            reward -= self.wall_penalty
            next_value = self.gamma * self.value_function[i, j]
        else:
            # Move to new position
            next_i, next_j = new_pos
            
            # Check if new position has object
            if env.objects[next_i, next_j] > 0:
                obj_id = env.objects[next_i, next_j] - 1  # Convert to 0-indexed
                reward += self.rewards[obj_id]  # Terminal reward
                next_value = 0  # Terminal state
            else:
                next_value = self.gamma * self.value_function[next_i, next_j]
        
        return reward + next_value
    
    def act(self, state: np.ndarray, env: GridWorld) -> int:
        """Select action according to computed policy"""
        if self.policy is None:
            self.plan(env)
        
        # Extract agent position from state
        agent_pos = np.where(state[:, :, 5] == 1)
        if len(agent_pos[0]) == 0:
            return 4  # Stay action if agent position not found
        
        i, j = agent_pos[0][0], agent_pos[1][0]
        action_probs = self.policy[i, j]
        
        return np.random.choice(len(action_probs), p=action_probs)
    
    def get_action_probabilities(self, state: np.ndarray, env: GridWorld) -> np.ndarray:
        """Get action probabilities for current state"""
        if self.policy is None:
            self.plan(env)
        
        # Extract agent position from state
        agent_pos = np.where(state[:, :, 5] == 1)
        if len(agent_pos[0]) == 0:
            # Return uniform distribution if agent position not found
            return np.ones(env.n_actions) / env.n_actions
        
        i, j = agent_pos[0][0], agent_pos[1][0]
        return self.policy[i, j].copy()
    
    def get_successor_representation(self, env: GridWorld, gamma_sr: float = 0.9) -> np.ndarray:
        """
        Compute successor representation: discounted state occupancy
        Returns array of shape (size, size) with expected future occupancy
        """
        if self.policy is None:
            self.plan(env)
        
        size = env.size
        sr = np.zeros((size, size))
        
        # Simulate trajectories to estimate successor representation
        n_simulations = 100
        max_steps = 50
        
        for _ in range(n_simulations):
            # Start from current agent position
            temp_env = env.copy()
            
            for step in range(max_steps):
                if temp_env.done:
                    break
                
                # Add discounted occupancy
                discount = (gamma_sr ** step)
                sr[temp_env.agent_pos] += discount / n_simulations
                
                # Take action according to policy
                state = temp_env.get_state()
                action = self.act(state, temp_env)
                temp_env.step(action)
        
        return sr


def create_random_agents(n_agents: int, alpha: float) -> List[RandomAgent]:
    """Create population of random agents with given alpha parameter"""
    return [RandomAgent(alpha) for _ in range(n_agents)]


def create_goal_directed_agents(n_agents: int, alpha_reward: float = 0.01, 
                               high_cost_ratio: float = 0.2) -> List[GoalDirectedAgent]:
    """
    Create population of goal-directed agents with diverse reward preferences
    
    Args:
        n_agents: Number of agents to create
        alpha_reward: Dirichlet concentration for reward sampling
        high_cost_ratio: Fraction of agents with high movement cost (0.5 vs 0.01)
    """
    agents = []
    n_high_cost = int(n_agents * high_cost_ratio)
    
    for i in range(n_agents):
        # Sample reward vector from Dirichlet
        rewards = np.random.dirichlet([alpha_reward] * 4)
        
        # Set movement cost (some agents are "greedy" with high cost)
        movement_cost = 0.5 if i < n_high_cost else 0.01
        
        agent = GoalDirectedAgent(rewards, movement_cost)
        agents.append(agent)
    
    return agents