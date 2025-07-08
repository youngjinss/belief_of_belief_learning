"""
Configuration file for MiniGrid-LockedRoom-v0 A* agent experiment
"""


class Config:
    """
    Configuration class for exp3 - MiniGrid-LockedRoom-v0 with A* agent
    """

    def __init__(self):
        # Environment settings
        self.env_name = "MiniGrid-LockedRoom-v0"
        self.size = 19  # Grid size
        self.max_steps = 1000  # Maximum steps per episode

        # Agent settings
        self.agent_type = "astar"  # Agent type
        self.max_exploration_steps = 1000  # Maximum exploration steps
        self.debug = False  # Debug mode

        # Data generation settings
        self.n_episodes = 1000  # Number of episodes to generate
        self.save_dir = "./data/exp3"  # Directory to save data
        self.random_seed = 42  # Base random seed
        self.n_processes = None  # Number of parallel processes (None = auto)

        # Data processing settings
        self.use_percentage = 0.9  # Percentage of data to use for training
        self.time_step = 20  # Maximum trajectory length for processing
        self.experiment_no = 3  # Experiment number

        # Training settings (for future ToMnet integration)
        self.batch_size = 32
        self.learning_rate = 0.001
        self.num_epochs = 100
        self.device = "cudaL3" if self._cuda_available() else "cpu"

        # Evaluation settings
        self.eval_episodes = 100  # Number of episodes for evaluation
        self.eval_seed = 123  # Seed for evaluation

    def _cuda_available(self):
        """Check if CUDA is available"""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def update_from_args(self, args):
        """Update configuration from command line arguments"""
        for key, value in vars(args).items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)

    def __str__(self):
        """String representation of configuration"""
        config_str = "Configuration:\n"
        config_str += "=" * 20 + "\n"
        for key, value in self.__dict__.items():
            if not key.startswith("_"):
                config_str += f"{key}: {value}\n"
        return config_str
