import torch


class Config:
    def __init__(self):
        # Environment settings
        self.env_name = "MiniGrid-AchieverBlocker-{size}-v1"
        self.width = 9
        self.height = 9
        self.env_size = (
            f"{self.width}x{self.height}"  # Options: "3x3", "5x5", "9x9", "11x11"
        )

        self.seed = 42

        # Agent settings
        self.n_games_per_type = 3000  # Number of games to generate for ToMnet data
        self.achiever_types = {
            "lv0va": self.n_games_per_type,
            # "lv1va": self.n_games_per_type,
        }  # Options: "lv0va", "lv1va", "astar", "random", "value"
        self.blocker_types = {
            # "lv0vb": self.n_games_per_type,
            # "lv1vb": self.n_games_per_type,
        }  # Options: "lv0vb", "lv1vb", "random", "goal_direct", "randomly_selected", "rule_based"
        self.observability = "full"  # Options: "full", "partial"
        self.movement_prob = 0.8  # For random agent

        # Visualization settings
        self.episodes = 1
        self.pause = 0.1  # Pause duration between actions in seconds
        self.render = True
        self.debug = False

        # Output settings
        self.output_dir = "results/exp6/"
        self.gif_output = None  # Filename for saving gif (without .gif extension)

        # Data generation settings
        self.save_dir = "data"  # Base directory to save generated data

        # Debug/test settings
        self.debug_mode = False  # Enable for small-scale testing

        # Experiment settings
        self.experiment_name = "exp6"
        self.experiment_no = 6
        self.log_actions = False
        self.log_rewards = False
        self.log_debug = False

        # Directory settings for evaluation and visualization
        self.model_dir = "results/exp6"
        self.test_data_dir = (
            None  # Will be set dynamically using get_data_path(is_test=True)
        )
        self.result_dir = "results/exp6"
        self.plot_dir = "results/exp6/plots"
        self.log_dir = "log/exp6"

        # Environment variants
        self.env_variants = {
            "3x3": {"grid_size": 3, "max_steps": 20},
            "5x5": {"grid_size": 5, "max_steps": 30},
            "9x9": {"grid_size": 9, "max_steps": 50},
            "11x11": {"grid_size": 11, "max_steps": 70},
        }

        # Set max_steps based on current env_size
        self.max_steps = self.env_variants[self.env_size]["max_steps"]

        # Achiever configurations
        self.achiever_configs = {
            "astar": {"observability": "full", "debug": False, "action_space": 7},
            "random": {
                "movement_prob": 0.8,
                "exploration_bias": 0.1,
                "action_space": 7,
            },
            "lv0va": {
                "observability": "full",
                "movement_cost": 1.0,
                "wall_penalty": 10.0,
                "conflict_penalty": 10.0,
                "consumption_penalty": 1.0,
                "gamma": 0.99,
                "temperature": 0.1,
                "q_value_clip": 100,
                "action_space": 7,
            },
            "lv1va": {
                "observability": "full",
                "movement_cost": 1.0,
                "wall_penalty": 10.0,
                "conflict_penalty": 10.0,
                "consumption_penalty": 1.0,
                "gamma": 0.99,
                "temperature": 0.1,
                "q_value_clip": 100,
                "action_space": 7,
            },
            "value": {
                "observability": "full",
                "movement_cost": 0.1,
                "wall_penalty": 10.0,
                "consumption_penalty": 1.0,
                "gamma": 0.99,
                "temperature": 0.1,
                "action_space": 7,
            },
            "value_deterministic": {
                "observability": "full",
                "movement_cost": 0.1,
                "wall_penalty": 10.0,
                "consumption_penalty": 1.0,
                "gamma": 0.99,
                "temperature": 0.0,
                "action_space": 7,
            },
            "value_stochastic": {
                "observability": "full",
                "movement_cost": 0.1,
                "wall_penalty": 10.0,
                "consumption_penalty": 1.0,
                "gamma": 0.99,
                "temperature": 0.5,
                "action_space": 7,
            },
        }

        # Achiever type mapping for output format
        self.achiever_type_map = {
            "lv0va": 0,
            "lv1va": 1,
            "astar": 2,
            "random": 3,
            "value": 4,
        }

        # Blocker type mapping for output format
        self.blocker_type_map = {
            "lv0vb": 0,
            "lv1vb": 1,
            "randomly_selected": 2,
            "rule_based": 3,
            "random": 4,
            "goal_direct": 5,
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
            "randomly_selected": {
                "observability": "full",
                "movement_cost": 0.05,
                "wall_penalty": 10.0,
                "gamma": 0.99,
                "temperature": 0.1,
                "action_space": 6,
                "stay_probability": 0.1,
            },
            "lv0vb": {
                "observability": "full",
                "movement_cost": 0.1,
                "wall_penalty": 10.0,
                "conflict_penalty": 0.0,
                "gamma": 0.99,
                "temperature": 0.1,
                "q_value_clip": 100,
                "action_space": 6,
            },
            "lv1vb": {
                "observability": "full",
                "movement_cost": 0.1,
                "wall_penalty": 10.0,
                "conflict_penalty": 0.0,
                "gamma": 0.99,
                "temperature": 0.1,
                "q_value_clip": 100,
                "action_space": 6,
            },
            "rule_based": {
                "observability": "full",
                "movement_cost": 0.05,
                "wall_penalty": 10.0,
                "gamma": 0.99,
                "temperature": 0.1,
                "action_space": 6,
                "stay_probability": 0.1,
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
            "epochs": 300,
            "lr": 0.0001,
            "weight_decay": 0.001,
            "training_proportion": 0.9,
            "device": "cuda:3",
            "device_ids": [3, 2],  # GPU IDs for parallel training
            "use_parallel": True,  # Enable parallel GPU training
            "use_amp": True,  # Automatic Mixed Precision for memory and speed
            "gradient_accumulation_steps": 2,  # Accumulate gradients over multiple batches
            "pin_memory": True,  # Pin memory for faster data transfer
            "num_workers": 0,  # Number of dataloader workers (0 = auto-detect CPU count)
            "optimizer": "adam",
        }

        # Model architecture
        self.model_config = {
            "use_mentalnet": True,  # False: experiment5-style (CharNet→PredNet), True: original 3-stage (CharNet→MentalNet→PredNet)
            "residual_blocks": 5,
            "n_echar": 128,
            "n_ement": 128,
            "out_channels": 64,
            "channels_in": 10,  # 8 original channels + 1 self position + 1 opponent position
            "current_state_channels": 8,  # For MentalNet: 8 original channels (no position channels)
            "achiever_action_space": 7,  # up, right, down, left, stay, pickup, toggle
            "blocker_action_space": 6,  # up, right, down, left, stay, broken
            "goal_space": 4,
            "env_width": self.width,
            "env_height": self.height,
            "hidden_size_lstm": 64,
            "fc_layer_sizes": [64, 32],
            "kernel_size": 3,
            "padding": 1,
            "stride": 1,
        }

        # Data processing configuration
        self.data_config = {
            "max_moves": 50,  # Maximum moves per trajectory (equivalent to experiment5)
            "time_step": 50,  # Time step for model processing (equivalent to experiment5)
            "min_time_steps": 5,  # Minimum timestep to start trajectory slicing from
            "max_n_past": 1,  # Maximum past episodes (matching experiment5)
            "n_past_min": 1,  # Minimum past episodes (matching experiment5)
            "n_past_max": 1,  # Maximum past episodes for sampling (matching experiment5)
            "rank_threshold": 4,  # How many top ranks to consider for matching (1=only highest, 2=top 2, etc.)
            "maze_width": self.width,
            "maze_height": self.height,
            "maze_depth": 10,  # 8 original channels + 1 self position + 1 opponent position
            "chunk_size": 5000,  # Number of samples to process per chunk for memory efficiency
        }

        # Training process configuration
        self.training_process_config = {
            "early_stopping_patience": 100,
            "early_stopping_min_delta": 0.001,
            "max_grad_norm": 1.0,
            "action_weight": 1,
            "goal_weight": 1,
            "agent_weight": 0.5,
            "type_weight": 0.05,
            "consumption_weight": 1,
            "sr_weight": 1,
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
            "n_past_min": 1,
            "n_past_max": 1,
            "n_past_infer": 1,
        }

    def is_single_agent_mode(self):
        """Check if running in single-agent mode (no blockers)"""
        return not self.blocker_types or len(self.blocker_types) == 0
    
    def get_test_data_proportion(self):
        """Get test data proportion (1 - training_proportion)"""
        return 1.0 - self.training_config["training_proportion"]

    def get_env_name(self):
        """Get full environment name based on agent configuration"""
        if self.is_single_agent_mode():
            # Use KeyDoor environment for single-agent mode
            return f"MiniGrid-KeyDoor-{self.env_size}-v1"
        else:
            # Use AchieverBlocker environment for multi-agent mode
            return self.env_name.format(size=self.env_size)

    def get_agent_pair_name(self, achiever_type, blocker_type=None):
        """Get agent pair name for directory structure"""
        if self.is_single_agent_mode() or blocker_type is None:
            # For single-agent mode, use only achiever type
            return achiever_type
        else:
            # For multi-agent mode, use both types
            return f"{achiever_type}_{blocker_type}"

    def get_data_path(self, achiever_type, blocker_type=None, is_test=False):
        """
        Get data path based on environment name and agent types

        Args:
            achiever_type (str): Type of achiever agent
            blocker_type (str): Type of blocker agent (None for single-agent mode)
            is_test (bool): If True, returns path for test data with /test suffix

        Returns:
            str: Data path in format ./data/{env_name}/{achiever_type}/ for single-agent or 
                 ./data/{env_name}/{achiever_type}_{blocker_type}/ for multi-agent
        """
        import os

        env_name = self.get_env_name()
        agent_combination = self.get_agent_pair_name(achiever_type, blocker_type)
        base_path = os.path.join(self.save_dir, env_name, agent_combination)

        if is_test:
            return os.path.join(base_path, "test")
        else:
            return base_path

    def get_training_data_path(self, achiever_type, blocker_type=None, is_test=False):
        """
        Get training data path that always uses 'data' as base directory, regardless of save_dir modifications

        Args:
            achiever_type (str): Type of achiever agent
            blocker_type (str): Type of blocker agent (None for single-agent mode)
            is_test (bool): If True, returns path for test data with /test suffix

        Returns:
            str: Training data path in format ./data/{env_name}/{achiever_type}/ for single-agent or
                 ./data/{env_name}/{achiever_type}_{blocker_type}/ for multi-agent
        """
        import os

        env_name = self.get_env_name()
        agent_combination = self.get_agent_pair_name(achiever_type, blocker_type)
        base_path = os.path.join("data", env_name, agent_combination)

        if is_test:
            return os.path.join(base_path, "test")
        else:
            return base_path

    def get_test_data_dir(self, achiever_type, blocker_type=None):
        """Get test data directory path"""
        return self.get_data_path(achiever_type, blocker_type, is_test=True)

    def get_env_config(self):
        """Get environment configuration"""
        return {
            "name": self.get_env_name(),
            "max_steps": self.max_steps,
            "seed": self.seed,
            "size": self.env_size,
        }

    def get_agent_config(self, achiever_type=None, blocker_type=None):
        """Get agent configuration"""
        base_config = {
            "achiever_types": self.achiever_types,
            "blocker_types": self.blocker_types,
            "observability": self.observability,
        }

        # Add agent-specific configurations if specific types are provided
        if achiever_type and achiever_type in self.achiever_configs:
            base_config.update(self.achiever_configs[achiever_type])

        return base_config

    def get_visualization_config(self):
        """Get visualization configuration"""
        return {
            "episodes": self.episodes,
            "pause": self.pause,
            "render": self.render,
            "debug": self.debug,
            "gif_output": self.gif_output,
            # Character embedding visualization settings
            "agent_colors": ["blue", "orange"],  # achiever, blocker
            "agent_names": ["Achiever", "Blocker"],
            "goal_colors": ["red", "green", "blue", "yellow"],
            "goal_names": ["Red", "Green", "Blue", "Yellow"],
            "goal_letters": ["A", "B", "C", "D"],
            "embedding_plots": {
                "pca_figsize": (20, 6),
                "tsne_figsize": (20, 6),
                "combined_figsize": (20, 12),
                "alpha": 0.6,
                "marker_size": 50,
            },
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

    def enable_debug_mode(self):
        """Enable debug mode with smaller scale settings for testing"""
        self.debug_mode = True

        # Reduce training settings for faster testing
        self.training_config.update(
            {
                "batch_size": 64,  # Reduced from 1024
                "epochs": 2,  # Reduced from 200 (faster test)
                "gradient_accumulation_steps": 1,  # Reduced from 2
            }
        )

        # Auto-detect device for debug mode
        if not torch.cuda.is_available():
            self.training_config["device"] = "cpu"
            self.training_config["use_parallel"] = False
            self.training_config["use_amp"] = False  # Disable AMP for CPU

        # Reduce data processing settings
        self.data_config.update(
            {
                "max_moves": 20,  # Reduced from 50
                "min_time_steps": 2,
                "time_step": 6,  # Reduced from 10
            }
        )

        # Reduce model complexity slightly
        self.model_config.update(
            {
                "residual_blocks": 1,  # Reduced from 5
                "n_echar": 16,  # Reduced from 128
                "n_ement": 16,  # Reduced from 128
            }
        )

        print("Debug mode enabled: Using smaller scale settings for testing")

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
            self.achiever_types = {
                args.achiever_type: self.n_games_per_type
            }  # Convert single type to dict for backward compatibility
        if hasattr(args, "blocker_type") and args.blocker_type is not None:
            if args.blocker_type == "none":
                # Single-agent mode
                self.blocker_types = {}
            else:
                self.blocker_types = {
                    args.blocker_type: self.n_games_per_type
                }  # Convert single type to dict for backward compatibility
        # Backward compatibility
        if hasattr(args, "agent_type") and args.agent_type is not None:
            self.achiever_types = {args.agent_type: self.n_games_per_type}
        if hasattr(args, "seed") and args.seed is not None:
            self.seed = args.seed
        if hasattr(args, "episodes") and args.episodes is not None:
            self.episodes = args.episodes
        if hasattr(args, "pause") and args.pause is not None:
            self.pause = args.pause
        if hasattr(args, "env_size") and args.env_size is not None:
            self.env_size = args.env_size
            # Update max_steps based on new env_size
            self.max_steps = self.env_variants[self.env_size]["max_steps"]
        if hasattr(args, "max_steps") and args.max_steps is not None:
            # Allow explicit max_steps override
            self.max_steps = args.max_steps
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
        if hasattr(args, "device_ids") and args.device_ids is not None:
            self.training_config["device_ids"] = args.device_ids

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
        if hasattr(args, "min_time_steps") and args.min_time_steps is not None:
            self.data_config["min_time_steps"] = args.min_time_steps
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
        valid_achiever_types = ["lv0va", "lv1va", "astar", "random", "value"]
        for achiever_type in self.achiever_types.keys():
            if achiever_type not in valid_achiever_types:
                raise ValueError(f"Invalid achiever_type: {achiever_type}")
        valid_blocker_types = [
            "lv0vb",
            "lv1vb",
            "random",
            "goal_direct",
            "randomly_selected",
            "rule_based",
        ]
        for blocker_type in self.blocker_types.keys():
            if blocker_type not in valid_blocker_types:
                raise ValueError(f"Invalid blocker_type: {blocker_type}")

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
  Achiever Types: {', '.join(self.achiever_types.keys())}
  Blocker Types: {', '.join(self.blocker_types.keys())}
  Observability: {self.observability}
  Episodes: {self.episodes}
  Max Steps: {self.max_steps}
  Seed: {self.seed}
  Output Dir: {self.output_dir}
"""
