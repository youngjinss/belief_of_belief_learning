"""Configuration behaviour shared by every experiment.

exp5..exp8 each carried their own Config class, but 21 of its 34 methods were
byte-identical in all four. Those live here; each experiment's Config subclasses
this and keeps only its own values plus the methods that genuinely differ
(__init__, get_env_name, get_data_path, get_model_kwargs, get_training_kwargs,
get_agent_pair_name, get_test_data_dir, update_from_args, enable_debug_mode).

The methods below read attributes the subclass sets in __init__; this class
supplies no defaults of its own, so behaviour is unchanged by the move.
"""


class BaseConfig:
    """Shared accessors over an experiment's configuration attributes."""

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

    def get_costs(self):
        """Get costs (either random or default)"""
        if self.cost_settings["use_random_costs"]:
            return self.generate_random_costs()
        else:
            door_colors = ["red", "green", "blue", "yellow"]
            default_costs = self.cost_settings["default_costs"]
            return {color: cost for color, cost in zip(door_colors, default_costs)}

    def get_data_config(self):
        """Get data processing configuration"""
        return self.data_config.copy()

    def get_env_config(self):
        """Get environment configuration"""
        return {
            "name": self.get_env_name(),
            "max_steps": self.max_steps,
            "seed": self.seed,
            "size": self.env_size,
        }

    def get_evaluation_config(self):
        """Get evaluation configuration"""
        return self.evaluation_config.copy()

    def get_experiment_config(self):
        """Get experiment configuration"""
        return {
            "name": self.experiment_name,
            "output_dir": self.output_dir,
            "log_actions": self.log_actions,
            "log_rewards": self.log_rewards,
            "log_debug": self.log_debug,
        }

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

    def get_goal_rewards(self):
        """Get goal rewards (either random or default)"""
        if self.goal_reward_settings["use_random_rewards"]:
            return self.generate_random_goal_rewards()
        else:
            return self.goal_reward_settings["default_rewards"]

    def get_history_config(self):
        """Get history file configuration"""
        return {
            "history_files": [
                "training_history.json",
                "history.json",
                "train_history.json",
            ]
        }

    def get_model_config(self):
        """Get model configuration"""
        return self.model_config.copy()

    def get_n_past_config(self):
        """Get N_past evaluation configuration"""
        return {"n_past_results_file": "n_past_evaluation_results.json"}

    def get_n_past_evaluation_config(self):
        """Get N_past evaluation configuration"""
        return self.n_past_evaluation.copy()

    def get_prediction_config(self):
        """Get prediction file configuration"""
        return {
            "prediction_files": [
                "predictions.pkl",
                "test_predictions.pkl",
                "eval_predictions.pkl",
            ]
        }

    def get_training_config(self):
        """Get training configuration"""
        return self.training_config.copy()

    def get_training_process_config(self):
        """Get training process configuration"""
        return self.training_process_config.copy()

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
