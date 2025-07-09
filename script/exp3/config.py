import os


class Config:
    def __init__(self):
        # Environment settings
        self.env_name = "MiniGrid-KeyDoor-{size}-v0"
        self.env_size = "9x9"  # Options: "3x3", "5x5", "9x9", "11x11"
        self.max_steps = 500
        self.seed = 42

        # Agent settings
        self.agent_type = "astar"  # Options: "astar", "random", "value"
        self.observability = "full"  # Options: "full", "partial"
        self.movement_prob = 0.8  # For random agent

        # Visualization settings
        self.episodes = 200
        self.pause = 0.5  # Pause duration between actions in seconds
        self.render = True
        self.debug = True

        # Output settings
        self.output_dir = "script/exp3/results"
        self.gif_output = None  # Filename for saving gif (without .gif extension)

        # Data generation settings
        self.n_games = 20000  # Number of games to generate for ToMnet data
        self.save_dir = "script/exp3/data"  # Directory to save generated data

        # Experiment settings
        self.experiment_name = "exp3"
        self.log_actions = True
        self.log_rewards = True
        self.log_debug = True

        # Environment variants
        self.env_variants = {
            "3x3": {"grid_size": 3, "max_steps": 100},
            "5x5": {"grid_size": 5, "max_steps": 200},
            "9x9": {"grid_size": 9, "max_steps": 500},
            "11x11": {"grid_size": 11, "max_steps": 1000},
        }

        # Agent configurations
        self.agent_configs = {
            "astar": {"observability": "full", "debug": True},
            "random": {"movement_prob": 0.8, "exploration_bias": 0.1},
            "value": {
                "observability": "full",
                "movement_cost": 0.01,
                "wall_penalty": 2.0,
                "gamma": 0.99,
                "temperature": 0.1,
            },
            "value_deterministic": {
                "observability": "full",
                "movement_cost": 0.01,
                "wall_penalty": 2.0,
                "gamma": 0.99,
                "temperature": 0.0,
            },
            "value_stochastic": {
                "observability": "full",
                "movement_cost": 0.01,
                "wall_penalty": 2.0,
                "gamma": 0.99,
                "temperature": 0.5,
            },
        }

        # Successor Representation (SR) settings
        self.sr_settings = {
            "gammas": [0.5, 0.9, 0.99],  # Discount factors for SR calculation
            "grid_size": 9,  # Default grid size for SR
        }

        # Goal reward settings (following ToMnetF pattern)
        self.goal_reward_settings = {
            "use_random_rewards": True,  # Enable random goal rewards
            "total_reward_sum": 4,  # Total sum of all goal rewards (user preference)
            "default_rewards": [
                0.5,
                1.0,
                1.5,
                1.0,
            ],  # Default rewards for goals A, B, C, D (sum=4)
            "min_reward": 0.1,  # Minimum reward value
            "max_reward": 3.0,  # Maximum reward value
        }

    def get_env_name(self):
        """Get full environment name"""
        return self.env_name.format(size=self.env_size)

    def get_env_config(self):
        """Get environment configuration"""
        return {
            "name": self.get_env_name(),
            "max_steps": self.max_steps,
            "seed": self.seed,
            "size": self.env_size,
        }

    def get_agent_config(self):
        """Get agent configuration"""
        base_config = {
            "agent_type": self.agent_type,
            "observability": self.observability,
        }

        # Add agent-specific configurations
        if self.agent_type in self.agent_configs:
            base_config.update(self.agent_configs[self.agent_type])

        return base_config

    def get_visualization_config(self):
        """Get visualization configuration"""
        return {
            "episodes": self.episodes,
            "pause": self.pause,
            "render": self.render,
            "debug": self.debug,
            "gif_output": self.gif_output,
        }

    def get_experiment_config(self):
        """Get experiment configuration"""
        return {
            "name": self.experiment_name,
            "output_dir": self.output_dir,
            "log_actions": self.log_actions,
            "log_rewards": self.log_rewards,
            "log_debug": self.log_debug,
        }

    def generate_random_goal_rewards(self, total_reward=None):
        """
        Generate random goal rewards that sum to total_reward
        Following ToMnetF pattern but with configurable sum

        Returns:
            list: Four goal rewards that sum to total_reward
        """
        import numpy as np

        if total_reward is None:
            total_reward = self.goal_reward_settings["total_reward_sum"]

        min_reward = self.goal_reward_settings["min_reward"]
        max_reward = self.goal_reward_settings["max_reward"]

        # Generate 3 random split points between min and max
        splits = np.random.uniform(0, 1, 3)
        splits = np.sort(splits)

        # Create 4 proportions
        proportions = [
            splits[0],
            splits[1] - splits[0],
            splits[2] - splits[1],
            1 - splits[2],
        ]

        # Scale to total reward
        rewards = [prop * total_reward for prop in proportions]

        # Ensure minimum reward constraint
        for i in range(len(rewards)):
            if rewards[i] < min_reward:
                rewards[i] = min_reward

        # Rescale to maintain sum constraint
        current_sum = sum(rewards)
        if current_sum != total_reward:
            scale_factor = total_reward / current_sum
            rewards = [r * scale_factor for r in rewards]

        # Ensure no reward exceeds maximum
        for i in range(len(rewards)):
            if rewards[i] > max_reward:
                rewards[i] = max_reward

        # Final rescaling to maintain exact sum
        current_sum = sum(rewards)
        if current_sum != total_reward:
            scale_factor = total_reward / current_sum
            rewards = [r * scale_factor for r in rewards]

        return rewards

    def get_goal_rewards(self):
        """Get goal rewards (either random or default)"""
        if self.goal_reward_settings["use_random_rewards"]:
            return self.generate_random_goal_rewards()
        else:
            return self.goal_reward_settings["default_rewards"]

    def update_from_args(self, args):
        """Update configuration from command line arguments"""
        if hasattr(args, "agent_type"):
            self.agent_type = args.agent_type
        if hasattr(args, "seed"):
            self.seed = args.seed
        if hasattr(args, "episodes"):
            self.episodes = args.episodes
        if hasattr(args, "pause"):
            self.pause = args.pause
        if hasattr(args, "max_steps"):
            self.max_steps = args.max_steps
        if hasattr(args, "env_size"):
            self.env_size = args.env_size
        if hasattr(args, "observability"):
            self.observability = args.observability
        if hasattr(args, "gif"):
            self.gif_output = args.gif
        if hasattr(args, "debug"):
            self.debug = args.debug

    def validate(self):
        """Validate configuration"""
        if self.agent_type not in ["astar", "random", "value"]:
            raise ValueError(f"Invalid agent_type: {self.agent_type}")

        if self.env_size not in ["3x3", "5x5", "9x9", "11x11"]:
            raise ValueError(f"Invalid env_size: {self.env_size}")

        if self.observability not in ["full", "partial"]:
            raise ValueError(f"Invalid observability: {self.observability}")

        if self.episodes <= 0:
            raise ValueError(f"Episodes must be positive: {self.episodes}")

        if self.max_steps <= 0:
            raise ValueError(f"Max steps must be positive: {self.max_steps}")

    def __str__(self):
        """String representation of configuration"""
        return f"""KeyDoor Experiment Configuration:
  Environment: {self.get_env_name()}
  Agent Type: {self.agent_type}
  Observability: {self.observability}
  Episodes: {self.episodes}
  Max Steps: {self.max_steps}
  Seed: {self.seed}
  Output Dir: {self.output_dir}
"""
