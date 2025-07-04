class Config:
    def __init__(self):
        import os

        # Agent and environment settings
        self.agent_type = "a_star"
        self.observability = "full"
        self.sight = 3
        self.n_games = int(os.getenv("N_GAMES", 5000))
        self.rows = 13
        self.cols = 13
        self.width = 13
        self.height = 13
        self.max_moves = 50
        self.output_dir = "data"
        self.shuffle = False
        self.no_walls = False

        # Model hyperparameters
        self.batch_size = int(os.getenv("BATCH_SIZE", 1024))
        self.residual_blocks = 5
        self.e_char = 8
        self.out_channels = 32
        self.time_step = 10  # Unified trajectory/time parameter
        self.depth = 10

        # Training settings
        self.epochs = int(os.getenv("EPOCHS", 50))
        self.lr = 1e-4
        self.training_proportion = 0.9
        self.device = "cuda:3"

        # Data directories
        self.data_dir = "../../data/experiment3"
        self.model_dir = "../../models/experiment3"
        self.result_dir = "../../result/experiment3"
        self.plot_dir = "../../plots/experiment3"
        self.log_dir = "../../log/training"

        # Experiment specific
        self.experiment_no = 3
        self.use_percentage = 0.9
        
        # N_past settings for character embedding
        self.n_past_min = 0  # Minimum number of past episodes
        self.n_past_max = 10  # Maximum number of past episodes
        self.use_n_past = True  # Whether to use past episodes for character embedding

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
        }
