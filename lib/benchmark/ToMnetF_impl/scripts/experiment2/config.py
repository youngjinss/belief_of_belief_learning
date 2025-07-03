class Config:
    def __init__(self):
        # Agent and environment settings
        self.agent_type = "a_star"
        self.observability = "full"
        self.sight = 3
        self.n_games = 5000
        self.rows = 13
        self.cols = 13
        self.width = 13
        self.height = 13
        self.max_moves = 50
        self.output_dir = "experiment"
        self.shuffle = False
        self.no_walls = False
        
        # Model hyperparameters
        self.batch = 512
        self.batch_size = 512
        self.residual_blocks = 5
        self.e_char = 8
        self.out_channels = 32
        self.time_frame = 10
        self.depth = 10
        
        # Training settings
        self.epoch = 50
        self.epochs = 50
        self.lr = 1e-4
        self.max_trajectory_size = 10
        self.ts = 10
        self.training_proportion = 0.9
        self.device = "cuda:3"
        
        # Data directories
        self.data_dir = "../../data/experiment2"
        self.model_dir = "../../models/experiment2"
        self.result_dir = "../../result/experiment2"
        self.plot_dir = "../../plots/experiment2"
        self.log_dir = "../../log/training"
        
        # Experiment specific
        self.experiment_no = 2
        self.use_percentage = 0.9
        
    def get_model_kwargs(self):
        """Return model parameters for ToMnet initialization"""
        return {
            "Batch": self.batch_size,
            "ResidualBlocks": self.residual_blocks,
            "N_echar": self.e_char,
            "out_channels": self.out_channels,
            "Max_trajectory_size": self.max_trajectory_size,
            "Width": self.width,
            "Height": self.height,
            "Depth": self.depth,
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
            "ts": self.ts,
            "height": self.height,
            "width": self.width,
            "depth": self.depth,
            "training_proportion": self.training_proportion,
            "use_percentage": self.use_percentage,
            "device": self.device,
        }