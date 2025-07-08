class Config:
    def __init__(self):
        import os

        # Agent and environment settings
        self.agent_type = "a_star"
        self.observability = "full"
        self.sight = 3
        self.n_games = int(os.getenv("N_GAMES", 20000))
        self.rows = 13
        self.cols = 13
        self.width = 13
        self.height = 13
        self.max_moves = 50
        self.output_dir = "data"
        self.shuffle = False
        self.no_walls = True
        self.random_positions = True  # Enable random player positions
        self.random_goal_rewards = True  # Enable random goal rewards

        # Model hyperparameters
        self.batch_size = int(os.getenv("BATCH_SIZE", 512))
        self.residual_blocks = 5
        self.e_char = 8
        self.out_channels = 32
        self.time_step = 20  # slicing time step = same with max_moves
        self.depth = 10

        # Training settings
        self.epochs = int(os.getenv("EPOCHS", 200))
        self.lr = 1e-4
        self.training_proportion = 0.9
        self.device = "cuda:2"

        # Early stopping settings
        self.early_stopping_patience = int(os.getenv("EARLY_STOPPING_PATIENCE", 50))
        self.early_stopping_min_delta = float(
            os.getenv("EARLY_STOPPING_MIN_DELTA", 0.001)
        )
        self.early_stopping_restore_best = True

        # Data directories
        self.data_dir = "../../data/experiment4"
        self.test_data_dir = "../../data/experiment4/test"
        self.model_dir = "../../models/experiment4"
        self.result_dir = "../../result/experiment4"
        self.plot_dir = "../../plots/experiment4"
        self.log_dir = "../../log/training"

        # Experiment specific
        self.experiment_no = 4
        self.use_percentage = 0.9

        # N_past settings for character embedding
        self.n_past_min = 1  # Minimum number of past episodes
        self.n_past_max = 1  # Maximum number of past episodes
        self.n_past_infer = 1  # Maximum number for inference
        self.use_n_past = True  # Whether to use past episodes for character embedding

        # Goal rank matching settings
        self.rank_threshold = int(
            os.getenv("RANK_THRESHOLD", 4)
        )  # How many top ranks to consider for matching
        # 1 = only rank 1 (highest reward goal)
        # 2 = rank 1 and 2 (top 2 goals)
        # 3 = rank 1, 2, and 3 (top 3 goals)
        # 4 = all ranks (full matching)

    def get_model_kwargs(self):
        """Return model parameters for ToMnet initialization"""
        return {
            "Batch": self.batch_size,
            "ResidualBlocks": self.residual_blocks,
            "N_echar": self.e_char,
            "out_channels": self.out_channels,
            "time_step": self.time_step,
            "Width": self.width,
            "Height": self.height,
            "Depth": self.depth,
            "max_n_past": self.n_past_max,
            "use_n_past": self.use_n_past,
        }

    def get_training_kwargs(self):
        """Return training parameters"""
        return {
            "data_dir": self.data_dir,
            "model_dir": self.model_dir,
            "result_dir": self.result_dir,
            "plot_dir": self.plot_dir,
            "log_dir": self.log_dir,
            "experiment_no": self.experiment_no,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "time_step": self.time_step,
            "height": self.height,
            "width": self.width,
            "depth": self.depth,
            "training_proportion": self.training_proportion,
            "use_percentage": self.use_percentage,
            "device": self.device,
            "max_n_past": self.n_past_max,
            "use_n_past": self.use_n_past,
            "rank_threshold": self.rank_threshold,
        }
