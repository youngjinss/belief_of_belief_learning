class Config:
    def __init__(self):
        # Environment settings
        self.env_name = "MiniGrid-AchieverBlocker-{size}-v1"
        self.width = 9
        self.height = 9
        self.env_size = (
            f"{self.width}x{self.height}"  # Options: "3x3", "5x5", "9x9", "11x11"
        )
        self.max_steps = 500
        self.seed = 42

        # Agent settings
        self.achiever_type = "value"  # Options: "astar", "random", "value"
        self.blocker_type = "goal_direct"  # Options: "random", "goal_direct"
        self.observability = "full"  # Options: "full", "partial"
        self.movement_prob = 0.8  # For random agent

        # Visualization settings
        self.episodes = 1
        self.pause = 0.1  # Pause duration between actions in seconds
        self.render = True
        self.debug = False

        # Output settings
        self.output_dir = "script/exp4/results"
        self.gif_output = None  # Filename for saving gif (without .gif extension)

        # Data generation settings
        self.n_games = 100000  # Number of games to generate for ToMnet data
        self.save_dir = "data"  # Base directory to save generated data

        # Experiment settings
        self.experiment_name = "exp4"
        self.experiment_no = 4
        self.log_actions = False
        self.log_rewards = False
        self.log_debug = False

        # Directory settings for evaluation and visualization
        self.model_dir = "results/exp4"
        self.test_data_dir = (
            None  # Will be set dynamically using get_data_path(is_test=True)
        )
        self.result_dir = "results/exp4"
        self.plot_dir = "results/exp4/plots"
        self.log_dir = "log/exp4"

        # Environment variants
        self.env_variants = {
            "3x3": {"grid_size": 3, "max_steps": 30},
            "5x5": {"grid_size": 5, "max_steps": 30},
            "9x9": {"grid_size": 9, "max_steps": 30},
            "11x11": {"grid_size": 11, "max_steps": 70},
        }

        # Achiever configurations
        self.achiever_configs = {
            "astar": {"observability": "full", "debug": False, "action_space": 7},
            "random": {
                "movement_prob": 0.8,
                "exploration_bias": 0.1,
                "action_space": 7,
            },
            "value": {
                "observability": "full",
                "movement_cost": 0.05,
                "wall_penalty": 10.0,
                "gamma": 0.99,
                "temperature": 0.1,
                "action_space": 7,
            },
            "value_deterministic": {
                "observability": "full",
                "movement_cost": 0.05,
                "wall_penalty": 10.0,
                "gamma": 0.99,
                "temperature": 0.0,
                "action_space": 7,
            },
            "value_stochastic": {
                "observability": "full",
                "movement_cost": 0.05,
                "wall_penalty": 10.0,
                "gamma": 0.99,
                "temperature": 0.5,
                "action_space": 7,
            },
        }

        # Blocker configurations
        self.blocker_configs = {
            "random": {
                "movement_prob": 0.8,
                "exploration_bias": 0.1,
                "action_space": 6,
            },
            "goal_direct": {
                "observability": "full",
                "movement_cost": 0.05,
                "wall_penalty": 10.0,
                "gamma": 0.99,
                "temperature": 0.1,
                "action_space": 6,
            },
        }

        # Successor Representation (SR) settings
        self.sr_settings = {
            "gammas": [0.5, 0.9, 0.99],  # Discount factors for SR calculation
            "grid_size": self.width,  # Default grid size for SR
        }

        # Goal reward settings (following ToMnetF pattern)
        self.goal_reward_settings = {
            "use_random_rewards": True,  # Enable random goal rewards
            "total_reward_sum": 2,  # Total sum of all goal rewards (user preference)
            "default_rewards": [
                1.0,
                0.4,
                0.1,
                0.2,
            ],  # Default rewards for goals A, B, C, D (sum=4)
            "min_reward": 0.1,  # Minimum reward value
            "max_reward": 1.0,  # Maximum reward value
        }

        # Cost settings (constraint: total sum must be 1)
        self.cost_settings = {
            "use_random_costs": True,  # Enable random cost generation
            "total_cost_sum": 1.0,  # Total sum of all costs (fixed constraint)
            "default_costs": [
                0.1,
                0.1,
                0.1,
                0.1,
            ],  # Default costs for red, green, blue, yellow (sum=1)
            "min_cost": 0.05,  # Minimum cost value (5%)
            "max_cost": 0.7,  # Maximum cost value (70%)
        }

        # Training configuration
        self.training_config = {
            "batch_size": 512,
            "epochs": 200,
            "lr": 0.0001,
            "weight_decay": 0.001,
            "training_proportion": 0.9,
            "device": "cuda:3",
            "optimizer": "adam",
        }

        # Model architecture
        self.model_config = {
            "use_mentalnet": True,  # False: experiment5-style (CharNet→PredNet), True: original 3-stage (CharNet→MentalNet→PredNet)
            "residual_blocks": 5,
            "n_echar": 128,
            "n_ement": 128,
            "out_channels": 32,
            "channels_in": 9,  # 8 original channels + 1 heading direction channel
            "current_state_channels": 8,  # For MentalNet: 8 original channels (no heading direction)
            "achiever_action_space": 7,  # up, right, down, left, stay, pickup, toggle
            "blocker_action_space": 6,  # up, right, down, left, stay, broken
            "goal_space": 4,
            "env_width": self.width,
            "env_height": self.height,
            "hidden_size_lstm": 32,
            "fc_layer_sizes": [32, 32],
            "kernel_size": 3,
            "padding": 1,
            "stride": 1,
        }

        # Data processing configuration
        self.data_config = {
            "max_moves": 50,  # Maximum moves per trajectory (equivalent to experiment5)
            "time_step": 10,  # Time step for model processing (equivalent to experiment5)
            "max_n_past": 1,  # Maximum past episodes (matching experiment5)
            "n_past_min": 1,  # Minimum past episodes (matching experiment5)
            "n_past_max": 1,  # Maximum past episodes for sampling (matching experiment5)
            "rank_threshold": 4,  # How many top ranks to consider for matching (1=only highest, 2=top 2, etc.)
            "maze_width": self.width,
            "maze_height": self.height,
            "maze_depth": 9,  # 8 original channels + 1 heading direction channel
        }

        # Training process configuration
        self.training_process_config = {
            "early_stopping_patience": 30,
            "early_stopping_min_delta": 0.001,
            "max_grad_norm": 1.0,
            "action_weight": 1.0,
            "goal_weight": 1.0,
            "agent_weight": 1.0,
            "consumption_weight": 1.0,
            "sr_weight": 1.0,
        }

        # Evaluation configuration
        self.evaluation_config = {
            "batch_size": 32,
            "device": "auto",
            "n_samples": 1000,
            "save_predictions": True,
            "use_percentage": 1.0,
        }

        # N_past evaluation settings
        self.n_past_evaluation = {
            "n_past_min": 0,
            "n_past_max": 5,
            "n_past_infer": 5,
        }

    def get_env_name(self):
        """Get full environment name"""
        return self.env_name.format(size=self.env_size)

    def get_data_path(self, is_test=False):
        """
        Get data path based on environment name and agent types

        Args:
            is_test (bool): If True, returns path for test data with /test suffix

        Returns:
            str: Data path in format ./data/{env_name}/{achiever_type}_{blocker_type}/ or ./data/{env_name}/{achiever_type}_{blocker_type}/test/
        """
        import os

        env_name = self.get_env_name()
        agent_combination = f"{self.achiever_type}_{self.blocker_type}"
        base_path = os.path.join(self.save_dir, env_name, agent_combination)

        if is_test:
            return os.path.join(base_path, "test")
        else:
            return base_path

    def get_test_data_dir(self):
        """Get test data directory path"""
        return self.get_data_path(is_test=True)

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
            "achiever_type": self.achiever_type,
            "blocker_type": self.blocker_type,
            "observability": self.observability,
        }

        # Add agent-specific configurations
        if self.achiever_type in self.achiever_configs:
            base_config.update(self.achiever_configs[self.achiever_type])

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

        # Generate 4 random rewards from uniform [0,1]
        rewards = np.random.uniform(0, 1, 4).tolist()

        # Find the maximum value
        max_value = max(rewards)

        # Find all indices with the maximum value
        max_indices = [i for i, val in enumerate(rewards) if val == max_value]

        # If there are ties, randomly select one
        if len(max_indices) > 1:
            selected_index = np.random.choice(max_indices)
        else:
            selected_index = max_indices[0]

        # Set only the selected maximum to 1.0
        rewards[selected_index] = 1.0

        return rewards

    def generate_random_costs(self):
        """
        Generate random costs that sum to 1.0 (required constraint)

        Returns:
            dict: Mapping of door colors to cost values with sum=1.0
        """
        import numpy as np

        min_cost = self.cost_settings["min_cost"]
        max_cost = self.cost_settings["max_cost"]
        total_cost_sum = self.cost_settings["total_cost_sum"]

        # Generate 3 random split points between 0 and 1
        splits = np.random.uniform(0, 1, 3)
        splits = np.sort(splits)

        # Create 4 proportions from splits
        proportions = [
            splits[0],
            splits[1] - splits[0],
            splits[2] - splits[1],
            1 - splits[2],
        ]

        # Scale to total cost sum (1.0)
        costs = [prop * total_cost_sum for prop in proportions]

        # Ensure minimum cost constraint
        for i in range(len(costs)):
            if costs[i] < min_cost:
                costs[i] = min_cost

        # Rescale to maintain sum constraint
        current_sum = sum(costs)
        if current_sum != total_cost_sum:
            scale_factor = total_cost_sum / current_sum
            costs = [c * scale_factor for c in costs]

        # Ensure no cost exceeds maximum
        for i in range(len(costs)):
            if costs[i] > max_cost:
                costs[i] = max_cost

        # Final rescaling to maintain exact sum
        current_sum = sum(costs)
        if current_sum != total_cost_sum:
            scale_factor = total_cost_sum / current_sum
            costs = [c * scale_factor for c in costs]

        # Map to door colors
        door_colors = ["red", "green", "blue", "yellow"]
        cost_dict = {color: cost for color, cost in zip(door_colors, costs)}

        return cost_dict

    def get_costs(self):
        """Get costs (either random or default)"""
        if self.cost_settings["use_random_costs"]:
            return self.generate_random_costs()
        else:
            door_colors = ["red", "green", "blue", "yellow"]
            default_costs = self.cost_settings["default_costs"]
            return {color: cost for color, cost in zip(door_colors, default_costs)}

    def get_goal_rewards(self):
        """Get goal rewards (either random or default)"""
        if self.goal_reward_settings["use_random_rewards"]:
            return self.generate_random_goal_rewards()
        else:
            return self.goal_reward_settings["default_rewards"]

    def get_training_config(self):
        """Get training configuration"""
        return self.training_config.copy()

    def get_model_config(self):
        """Get model configuration"""
        return self.model_config.copy()

    def get_data_config(self):
        """Get data processing configuration"""
        return self.data_config.copy()

    def get_training_process_config(self):
        """Get training process configuration"""
        return self.training_process_config.copy()

    def get_evaluation_config(self):
        """Get evaluation configuration"""
        return self.evaluation_config.copy()

    def get_n_past_evaluation_config(self):
        """Get N_past evaluation configuration"""
        return self.n_past_evaluation.copy()

    def get_model_kwargs(self):
        """Get model initialization parameters"""
        return {
            "use_mentalnet": self.model_config["use_mentalnet"],
            "batch": self.training_config["batch_size"],
            "residual_blocks": self.model_config["residual_blocks"],
            "n_echar": self.model_config["n_echar"],
            "n_ement": self.model_config["n_ement"],
            "out_channels": self.model_config["out_channels"],
            "channels_in": self.model_config["channels_in"],
            "current_state_channels": self.model_config["current_state_channels"],
            "time_step": self.data_config["time_step"],
            "action_space": max(
                self.model_config["achiever_action_space"],
                self.model_config["blocker_action_space"],
            ),  # Use max action space
            "goal_space": self.model_config["goal_space"],
            "max_n_past": self.data_config["max_n_past"],
            "use_n_past": True,
            "hidden_size_lstm": self.model_config["hidden_size_lstm"],
            "env_width": self.model_config["env_width"],
            "env_height": self.model_config["env_height"],
        }

    def get_training_kwargs(self):
        """Get training function parameters"""
        return {
            "batch_size": self.training_config["batch_size"],
            "epochs": self.training_config["epochs"],
            "lr": self.training_config["lr"],
            "training_proportion": self.training_config["training_proportion"],
            "max_moves": self.data_config["max_moves"],
            "time_step": self.data_config["time_step"],
            "max_n_past": self.data_config["max_n_past"],
            "device": self.training_config["device"],
            "patience": self.training_process_config["early_stopping_patience"],
            "min_delta": self.training_process_config["early_stopping_min_delta"],
        }

    def update_from_args(self, args):
        """Update configuration from command line arguments"""
        # Environment and agent settings
        if hasattr(args, "achiever_type") and args.achiever_type is not None:
            self.achiever_type = args.achiever_type
        if hasattr(args, "blocker_type") and args.blocker_type is not None:
            self.blocker_type = args.blocker_type
        # Backward compatibility
        if hasattr(args, "agent_type") and args.agent_type is not None:
            self.achiever_type = args.agent_type
        if hasattr(args, "seed") and args.seed is not None:
            self.seed = args.seed
        if hasattr(args, "episodes") and args.episodes is not None:
            self.episodes = args.episodes
        if hasattr(args, "pause") and args.pause is not None:
            self.pause = args.pause
        if hasattr(args, "max_steps") and args.max_steps is not None:
            self.max_steps = args.max_steps
        if hasattr(args, "env_size") and args.env_size is not None:
            self.env_size = args.env_size
            # Update width and height based on env_size
            if self.env_size == "3x3":
                self.width = 3
                self.height = 3
            elif self.env_size == "5x5":
                self.width = 5
                self.height = 5
            elif self.env_size == "9x9":
                self.width = 9
                self.height = 9
            elif self.env_size == "11x11":
                self.width = 11
                self.height = 11
        if hasattr(args, "observability") and args.observability is not None:
            self.observability = args.observability
        if hasattr(args, "gif") and args.gif is not None:
            self.gif_output = args.gif
        if hasattr(args, "debug") and args.debug is not None:
            self.debug = args.debug

        # Training configuration
        if hasattr(args, "batch_size") and args.batch_size is not None:
            self.training_config["batch_size"] = args.batch_size
        if hasattr(args, "epochs") and args.epochs is not None:
            self.training_config["epochs"] = args.epochs
        if hasattr(args, "lr") and args.lr is not None:
            self.training_config["lr"] = args.lr
        if hasattr(args, "weight_decay") and args.weight_decay is not None:
            self.training_config["weight_decay"] = args.weight_decay
        if (
            hasattr(args, "training_proportion")
            and args.training_proportion is not None
        ):
            self.training_config["training_proportion"] = args.training_proportion
        if hasattr(args, "device") and args.device is not None:
            self.training_config["device"] = args.device
        if hasattr(args, "optimizer") and args.optimizer is not None:
            self.training_config["optimizer"] = args.optimizer

        # Model architecture
        if hasattr(args, "residual_blocks") and args.residual_blocks is not None:
            self.model_config["residual_blocks"] = args.residual_blocks
        if hasattr(args, "n_echar") and args.n_echar is not None:
            self.model_config["n_echar"] = args.n_echar
        if hasattr(args, "n_ement") and args.n_ement is not None:
            self.model_config["n_ement"] = args.n_ement
        if hasattr(args, "out_channels") and args.out_channels is not None:
            self.model_config["out_channels"] = args.out_channels
        if hasattr(args, "channels_in") and args.channels_in is not None:
            self.model_config["channels_in"] = args.channels_in
        if (
            hasattr(args, "achiever_action_space")
            and args.achiever_action_space is not None
        ):
            self.model_config["achiever_action_space"] = args.achiever_action_space
        if (
            hasattr(args, "blocker_action_space")
            and args.blocker_action_space is not None
        ):
            self.model_config["blocker_action_space"] = args.blocker_action_space
        if hasattr(args, "goal_space") and args.goal_space is not None:
            self.model_config["goal_space"] = args.goal_space
        if hasattr(args, "hidden_size_lstm") and args.hidden_size_lstm is not None:
            self.model_config["hidden_size_lstm"] = args.hidden_size_lstm

        # Data processing configuration
        if hasattr(args, "max_moves") and args.max_moves is not None:
            self.data_config["max_moves"] = args.max_moves
        if hasattr(args, "time_step") and args.time_step is not None:
            self.data_config["time_step"] = args.time_step
        if hasattr(args, "max_n_past") and args.max_n_past is not None:
            self.data_config["max_n_past"] = args.max_n_past
        if hasattr(args, "n_past_min") and args.n_past_min is not None:
            self.data_config["n_past_min"] = args.n_past_min
        if hasattr(args, "n_past_max") and args.n_past_max is not None:
            self.data_config["n_past_max"] = args.n_past_max
        if hasattr(args, "rank_threshold") and args.rank_threshold is not None:
            self.data_config["rank_threshold"] = args.rank_threshold

        # Training process configuration
        if (
            hasattr(args, "early_stopping_patience")
            and args.early_stopping_patience is not None
        ):
            self.training_process_config["early_stopping_patience"] = (
                args.early_stopping_patience
            )
        if (
            hasattr(args, "early_stopping_min_delta")
            and args.early_stopping_min_delta is not None
        ):
            self.training_process_config["early_stopping_min_delta"] = (
                args.early_stopping_min_delta
            )
        if hasattr(args, "max_grad_norm") and args.max_grad_norm is not None:
            self.training_process_config["max_grad_norm"] = args.max_grad_norm
        if hasattr(args, "action_weight") and args.action_weight is not None:
            self.training_process_config["action_weight"] = args.action_weight
        if hasattr(args, "goal_weight") and args.goal_weight is not None:
            self.training_process_config["goal_weight"] = args.goal_weight

        # Data generation settings
        if hasattr(args, "n_games") and args.n_games is not None:
            self.n_games = args.n_games
        if hasattr(args, "save_dir") and args.save_dir is not None:
            self.save_dir = args.save_dir

        # Evaluation configuration
        if hasattr(args, "test_data_dir") and args.test_data_dir is not None:
            self.test_data_dir = args.test_data_dir
        if hasattr(args, "model_dir") and args.model_dir is not None:
            self.model_dir = args.model_dir
        if hasattr(args, "result_dir") and args.result_dir is not None:
            self.result_dir = args.result_dir
        if hasattr(args, "plot_dir") and args.plot_dir is not None:
            self.plot_dir = args.plot_dir
        if hasattr(args, "experiment_no") and args.experiment_no is not None:
            self.experiment_no = args.experiment_no
        if hasattr(args, "n_samples") and args.n_samples is not None:
            self.evaluation_config["n_samples"] = args.n_samples
        if hasattr(args, "save_predictions") and args.save_predictions is not None:
            self.evaluation_config["save_predictions"] = args.save_predictions

        # N_past evaluation settings
        if hasattr(args, "n_past_min") and args.n_past_min is not None:
            self.n_past_evaluation["n_past_min"] = args.n_past_min
        if hasattr(args, "n_past_max") and args.n_past_max is not None:
            self.n_past_evaluation["n_past_max"] = args.n_past_max
        if hasattr(args, "n_past_infer") and args.n_past_infer is not None:
            self.n_past_evaluation["n_past_infer"] = args.n_past_infer

    def validate(self):
        """Validate configuration"""
        if self.achiever_type not in ["astar", "random", "value"]:
            raise ValueError(f"Invalid achiever_type: {self.achiever_type}")
        if self.blocker_type not in ["random", "goal_direct"]:
            raise ValueError(f"Invalid blocker_type: {self.blocker_type}")

        if self.env_size not in ["3x3", "5x5", "9x9", "11x11"]:
            raise ValueError(f"Invalid env_size: {self.env_size}")

        if self.observability not in ["full", "partial"]:
            raise ValueError(f"Invalid observability: {self.observability}")

        if self.episodes <= 0:
            raise ValueError(f"Episodes must be positive: {self.episodes}")

        if self.max_steps <= 0:
            raise ValueError(f"Max steps must be positive: {self.max_steps}")

    def get_goal_config(self):
        """Get goal configuration for visualization"""
        return {
            "num_goals": self.model_config.get("goal_space", 4),
            "goal_colors": ["red", "green", "blue", "yellow"],
            "goal_names": [
                "Goal A (Red)",
                "Goal B (Green)",
                "Goal C (Blue)",
                "Goal D (Yellow)",
            ],
        }

    def get_action_config(self):
        """Get action configuration for visualization"""
        return {
            "achiever_num_actions": self.model_config.get("achiever_action_space", 7),
            "blocker_num_actions": self.model_config.get("blocker_action_space", 6),
            "achiever_action_names": [
                "Up",
                "Right",
                "Down",
                "Left",
                "Stay",
                "Pickup",
                "Toggle",
            ],
            "blocker_action_names": ["Up", "Right", "Down", "Left", "Stay", "Broken"],
        }

    def get_history_config(self):
        """Get history file configuration"""
        return {
            "history_files": [
                "training_history.json",
                "history.json",
                "train_history.json",
            ]
        }

    def get_prediction_config(self):
        """Get prediction file configuration"""
        return {
            "prediction_files": [
                "predictions.pkl",
                "test_predictions.pkl",
                "eval_predictions.pkl",
            ]
        }

    def get_n_past_config(self):
        """Get N_past evaluation configuration"""
        return {"n_past_results_file": "n_past_evaluation_results.json"}

    def __str__(self):
        """String representation of configuration"""
        return f"""AchieverBlocker Experiment Configuration:
  Environment: {self.get_env_name()}
  Achiever Type: {self.achiever_type}
  Blocker Type: {self.blocker_type}
  Observability: {self.observability}
  Episodes: {self.episodes}
  Max Steps: {self.max_steps}
  Seed: {self.seed}
  Output Dir: {self.output_dir}
"""
