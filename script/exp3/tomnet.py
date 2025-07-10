import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

"""
ToMnet architecture for KeyDoor environment (experiment 3)
Adapted from ToMnetF experiment5 for KeyDoor environment

Channel structure (9 channels total):
- Channels 0-7: Original game state channels (walls, keys, doors, agent position, etc.)
- Channel 8: Agent heading direction (0=north, 1=east, 2=south, 3=west)

@author: Based on ToMnetF implementation, adapted for KeyDoor
"""


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        super(ResidualBlock, self).__init__()

        # Block structure
        self.convBlock_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.convBlock_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
            ),
            nn.BatchNorm2d(out_channels),
        )

        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        x = self.convBlock_1(x)
        x = self.convBlock_2(x)

        return self.relu(x + residual)  # skipping connection


class LSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, x):
        # x.shape -> (batch_size, seq_len, input_size)
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))

        return out[:, -1, :]  # Return last timestep output


class CharNet(nn.Module):
    def __init__(
        self,
        batch: int,
        residual_blocks: int,
        n_echar: int,
        out_channels: int,
        channels_in: int,
        time_step: int,
        max_n_past: int = 10,
        use_n_past: bool = True,
        hidden_size_lstm: int = 64,
    ):
        super(CharNet, self).__init__()

        self.n = residual_blocks
        self.n_echar = n_echar
        self.out_channels = out_channels
        self.channels_in = channels_in
        self.batch = batch
        self.time_step = time_step
        self.hidden_size_lstm = hidden_size_lstm  # Fixed like experiment 5
        self.max_n_past = max_n_past
        self.use_n_past = use_n_past

        if self.use_n_past:
            # Past episode processing architecture - following experiment 5 exactly
            self.past_conv_1 = nn.Conv2d(
                in_channels=channels_in,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
            )
            self.past_res_blocks = nn.ModuleList()

            for _ in range(residual_blocks):
                self.past_res_blocks.append(
                    ResidualBlock(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        padding=1,
                        stride=1,
                    )
                )

            self.past_lstm = LSTM(out_channels, self.hidden_size_lstm)
            self.past_e_char = nn.Linear(self.hidden_size_lstm, n_echar)

        # Always create default_embedding as a fallback
        self.default_embedding = nn.Parameter(torch.zeros(n_echar))

    def forward(self, past_trajectories):
        """
        Forward pass for character network - following experiment 5 exactly

        Args:
            past_trajectories: (batch_size, n_past, seq_len, channels, height, width)

        Returns:
            Character embeddings: (batch_size, n_echar)
        """
        if self.use_n_past and past_trajectories is not None:
            batch_size, n_past_max = past_trajectories.size(0), past_trajectories.size(
                1
            )

            # Initialize character embedding from past episodes
            e_char_past = torch.zeros(batch_size, self.n_echar).to(
                past_trajectories.device
            )
            # Track number of valid episodes per sample for averaging
            valid_episode_counts = torch.zeros(batch_size).to(past_trajectories.device)

            # Process each past episode and accumulate for averaging
            for ep_idx in range(n_past_max):
                # Get episode ep_idx for all samples in batch
                episode_batch = past_trajectories[
                    :, ep_idx
                ]  # (batch, seq_len, channels, height, width)

                # Reshape to (batch * seq_len, channels, height, width) for Conv2d processing
                batch_size_local, seq_len_local, channels, height, width = (
                    episode_batch.shape
                )
                episode_batch = episode_batch.reshape(
                    batch_size_local * seq_len_local, channels, height, width
                )

                # Check if episode is non-zero (not masked)
                episode_check = episode_batch.view(batch_size_local, seq_len_local, -1)
                episode_mask = torch.sum(episode_check, dim=[1, 2]) > 0  # (batch,)

                if episode_mask.any():
                    # Process through conv layers directly
                    ep_x = episode_batch  # (batch * seq_len, channels, height, width)

                    # Apply first conv layer
                    ep_x = self.past_conv_1(ep_x)  # Direct Conv2d

                    # Apply residual blocks
                    for i in range(self.n):
                        ep_x = self.past_res_blocks[i](ep_x)

                    # Reshape back to (batch, seq_len, height, width, out_channels)
                    _, out_channels, out_height, out_width = ep_x.shape
                    ep_x = ep_x.view(
                        batch_size_local,
                        seq_len_local,
                        out_channels,
                        out_height,
                        out_width,
                    )
                    ep_x = ep_x.permute(
                        0, 1, 3, 4, 2
                    )  # (batch, seq_len, height, width, out_channels)

                    # Average over spatial dimensions
                    ep_x = torch.mean(ep_x, [2, 3])  # (batch, seq_len, out_channels)

                    # Apply LSTM
                    ep_x = self.past_lstm(ep_x)

                    # Get character embedding for this episode
                    ep_e_char = self.past_e_char(ep_x)  # (batch, n_echar)

                    # Add to cumulative sum only for non-masked episodes (vectorized)
                    e_char_past += ep_e_char * episode_mask.unsqueeze(-1)
                    # Track valid episode counts
                    valid_episode_counts += episode_mask.float()

            # Average the character embeddings over valid episodes
            # Avoid division by zero by using maximum of counts and 1
            valid_episode_counts = torch.maximum(
                valid_episode_counts, torch.ones_like(valid_episode_counts)
            )
            e_char_past = e_char_past / valid_episode_counts.unsqueeze(-1)

            return e_char_past
        else:
            # Return default embedding if not using past episodes
            batch_size = (
                past_trajectories.size(0)
                if past_trajectories is not None
                else self.batch
            )
            return self.default_embedding.unsqueeze(0).expand(batch_size, -1)


class MentalNet(nn.Module):
    """Simplified MentalNet that processes current state instead of trajectory"""

    def __init__(
        self,
        batch: int,
        residual_blocks: int,
        n_ement: int,
        out_channels: int,
        channels_in: int,
        time_step: int,
        n_echar: int,
    ):
        super(MentalNet, self).__init__()

        self.n = residual_blocks
        self.n_ement = n_ement
        self.out_channels = out_channels
        self.original_channels_in = channels_in  # This is current_state_channels (8)
        self.channels_in = (
            channels_in + n_echar
        )  # Current state channels + character embedding
        self.batch = batch
        self.n_echar = n_echar

        # Convolutional layers for spatial feature extraction
        self.conv_layers = nn.ModuleList()

        # Initial convolution
        self.conv_layers.append(
            nn.Conv2d(self.channels_in, out_channels, kernel_size=3, padding=1)
        )

        # Residual blocks
        for _ in range(residual_blocks):
            self.conv_layers.append(
                ResidualBlock(out_channels, out_channels, kernel_size=3, padding=1)
            )

        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Mental state network (directly from spatial features)
        self.mental_state_net = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, n_ement),
        )

    def forward(self, recent_trajectory_with_char):
        """
        Forward pass for mental state network

        Args:
            recent_trajectory_with_char: (batch_size, seq_len, channels_in + n_echar, height, width)

        Returns:
            Mental state embedding: (batch_size, n_ement)
            Spatial features: (batch_size, out_channels, height, width)
        """
        batch_size, seq_len, channels, height, width = recent_trajectory_with_char.shape

        # Reshape to process all timesteps through conv layers
        x = recent_trajectory_with_char.view(
            batch_size * seq_len, channels, height, width
        )

        # Extract spatial features through conv layers
        for layer in self.conv_layers:
            x = layer(x)

        # Reshape back to sequence format
        _, out_channels, out_height, out_width = x.shape
        x = x.view(batch_size, seq_len, out_channels, out_height, out_width)

        # Global average pooling over spatial dimensions for each timestep
        x_pooled = x.mean(dim=[3, 4])  # (batch_size, seq_len, out_channels)

        # Use the last timestep for mental state prediction
        final_features = x_pooled[:, -1, :]  # (batch_size, out_channels)

        # Mental state prediction
        mental_state = self.mental_state_net(final_features)

        # Return spatial features from last timestep for SR prediction
        spatial_features = x[
            :, -1, :, :, :
        ]  # (batch_size, out_channels, height, width)

        return mental_state, spatial_features


class PredNet(nn.Module):
    def __init__(
        self,
        batch: int,
        n_ement: int,
        n_echar: int,
        action_space: int = 7,
        out_channels: int = 64,
        goal_space: int = 4,
        env_width: int = 9,
        env_height: int = 9,
    ):
        super(PredNet, self).__init__()

        self.batch = batch
        self.n_ement = n_ement
        self.n_echar = n_echar
        self.action_space = action_space
        self.goal_space = goal_space
        self.env_width = env_width
        self.env_height = env_height

        # Action prediction network
        self.action_predictor = nn.Sequential(
            nn.Linear(n_ement + n_echar, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, action_space),
        )

        # Goal prediction network
        self.goal_predictor = nn.Sequential(
            nn.Linear(n_ement + n_echar, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, goal_space),
        )

        # Consumption prediction network (8 outputs: 4 keys + 4 doors)
        self.consumption_predictor = nn.Sequential(
            nn.Linear(n_ement + n_echar, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, 8),  # 4 keys + 4 doors
        )

        # SR prediction network using spatial convolutions (matching experiment 5)
        # Uses spatial convolutional layers to maintain spatial structure
        self.conv_sr = nn.Conv2d(
            in_channels=out_channels,  # This should match out_channels from MentalNet
            out_channels=out_channels,
            kernel_size=1,
        )
        self.conv_sr_out = nn.Conv2d(
            in_channels=out_channels,
            out_channels=3,  # 3 discount factors
            kernel_size=1,
        )

    def forward(self, mental_state, character_embedding, spatial_features):
        """
        Forward pass for prediction network

        Args:
            mental_state: (batch_size, n_ement)
            character_embedding: (batch_size, n_echar)
            spatial_features: (batch_size, out_channels, height, width)

        Returns:
            action_logits: (batch_size, action_space)
            goal_logits: (batch_size, goal_space)
            consumption_logits: (batch_size, 8)
            sr_pred: (batch_size, 3, env_height, env_width)
        """
        # Combine mental state and character embedding
        combined = torch.cat([mental_state, character_embedding], dim=1)

        # Predict actions, goals, and consumption
        action_logits = self.action_predictor(combined)
        goal_logits = self.goal_predictor(combined)
        consumption_logits = self.consumption_predictor(combined)

        # Predict SR maps using spatial convolutions (matching experiment 5)
        sr_features = self.conv_sr(
            spatial_features
        )  # (batch_size, out_channels, height, width)
        sr_features = F.relu(sr_features)
        sr_pred = self.conv_sr_out(sr_features)  # (batch_size, 3, height, width)

        # Apply softmax to each SR channel independently to get probability distributions
        batch_size, channels, height, width = sr_pred.shape
        sr_pred = sr_pred.view(
            batch_size, channels, -1
        )  # (batch_size, 3, spatial_size)
        sr_pred = F.softmax(sr_pred, dim=2)
        sr_pred = sr_pred.view(
            batch_size, channels, height, width
        )  # Back to spatial format

        return action_logits, goal_logits, consumption_logits, sr_pred


class ToMnet(nn.Module):
    def __init__(
        self,
        batch: int = 32,
        residual_blocks: int = 3,
        n_echar: int = 64,
        n_ement: int = 64,
        out_channels: int = 32,
        channels_in: int = 9,  # 8 original channels + 1 heading direction channel (for CharNet)
        current_state_channels: int = 8,  # For MentalNet (without heading direction)
        time_step: int = 500,
        action_space: int = 7,
        goal_space: int = 4,
        max_n_past: int = 10,
        use_n_past: bool = True,
        env_width: int = 9,
        env_height: int = 9,
        hidden_size_lstm: int = 64,
    ):
        super(ToMnet, self).__init__()

        self.batch = batch
        self.n_echar = n_echar
        self.n_ement = n_ement
        self.time_step = time_step
        self.action_space = action_space
        self.goal_space = goal_space
        self.max_n_past = max_n_past
        self.use_n_past = use_n_past
        self.channels_in = channels_in
        self.env_width = env_width
        self.env_height = env_height

        # Character network - processes past episodes
        self.char_net = CharNet(
            batch=batch,
            residual_blocks=residual_blocks,
            n_echar=n_echar,
            out_channels=out_channels,
            channels_in=channels_in,
            time_step=time_step,
            max_n_past=max_n_past,
            use_n_past=use_n_past,
            hidden_size_lstm=hidden_size_lstm,
        )

        # Mental state network - processes current state + character embedding
        self.mental_net = MentalNet(
            batch=batch,
            residual_blocks=residual_blocks,
            n_ement=n_ement,
            out_channels=out_channels,
            channels_in=current_state_channels,  # Use current_state_channels for MentalNet
            time_step=time_step,
            n_echar=n_echar,
        )

        # Prediction network - processes mental state + character embedding
        self.pred_net = PredNet(
            batch=batch,
            n_ement=n_ement,
            n_echar=n_echar,
            action_space=action_space,
            out_channels=out_channels,
            goal_space=goal_space,
            env_width=env_width,
            env_height=env_height,
        )

    def forward(self, past_trajectories, recent_trajectory, current_state):
        """
        Forward pass for ToMnet (3-stage architecture)

        Args:
            past_trajectories: (batch_size, n_past, seq_len, channels, height, width) - for CharNet
            recent_trajectory: (batch_size, seq_len, channels, height, width) - for MentalNet
            current_state: (batch_size, channels, height, width) - for PredNet

        Returns:
            action_logits: (batch_size, action_space)
            goal_logits: (batch_size, goal_space)
            consumption_logits: (batch_size, 8)
            sr_pred: (batch_size, 3, env_height, env_width)
            character_embedding: (batch_size, n_echar)
            mental_state: (batch_size, n_ement)
        """
        # 1. Character network - processes past episodes
        if self.use_n_past and past_trajectories is not None:
            character_embedding = self.char_net(past_trajectories)
        else:
            # Use zero embedding if no past trajectories
            batch_size = current_state.size(0)
            character_embedding = torch.zeros(
                batch_size, self.n_echar, device=current_state.device
            )

        # 2. Mental state network - processes recent_trajectory + character embedding
        batch_size, seq_len, channels, height, width = recent_trajectory.shape

        # Reshape character embedding to spatial format for trajectory concatenation
        e_char_spatial = (
            character_embedding.unsqueeze(1).unsqueeze(3).unsqueeze(4)
        )  # (batch, 1, n_echar, 1, 1)
        e_char_spatial = e_char_spatial.expand(
            batch_size, seq_len, self.n_echar, height, width
        )  # (batch, seq_len, n_echar, height, width)

        # Concatenate recent_trajectory with character embedding
        recent_trajectory_with_char = torch.cat(
            [recent_trajectory, e_char_spatial], dim=2
        )  # (batch, seq_len, channels + n_echar, height, width)

        # Process through MentalNet
        mental_state, spatial_features = self.mental_net(recent_trajectory_with_char)

        # 3. Prediction network - processes current_state + character embedding + mental state
        action_logits, goal_logits, consumption_logits, sr_pred = self.pred_net(
            mental_state, character_embedding, spatial_features
        )

        return (
            action_logits,
            goal_logits,
            consumption_logits,
            sr_pred,
            character_embedding,
            mental_state,
        )

    def predict_action(self, past_trajectories, recent_trajectory, current_state):
        """
        Predict next action

        Args:
            past_trajectories: Past episode trajectories
            recent_trajectory: Recent trajectory for MentalNet
            current_state: Current state for PredNet

        Returns:
            Predicted action probabilities
        """
        with torch.no_grad():
            action_logits, _, _, _, _, _ = self.forward(
                past_trajectories, recent_trajectory, current_state
            )
            return F.softmax(action_logits, dim=1)

    def predict_goal(self, past_trajectories, recent_trajectory, current_state):
        """
        Predict goal

        Args:
            past_trajectories: Past episode trajectories
            recent_trajectory: Recent trajectory for MentalNet
            current_state: Current state for PredNet

        Returns:
            Predicted goal probabilities
        """
        with torch.no_grad():
            _, goal_logits, _, _, _, _ = self.forward(
                past_trajectories, recent_trajectory, current_state
            )
            return F.softmax(goal_logits, dim=1)

    def get_character_embedding(self, past_trajectories):
        """
        Get character embedding from past trajectories

        Args:
            past_trajectories: Past episode trajectories

        Returns:
            Character embedding
        """
        with torch.no_grad():
            if self.use_n_past and past_trajectories is not None:
                return self.char_net(past_trajectories)
            else:
                batch_size = 1
                return torch.zeros(batch_size, self.n_echar)

    def get_mental_state(self, recent_trajectory, character_embedding):
        """
        Get mental state from recent trajectory and character embedding

        Args:
            recent_trajectory: Recent trajectory for MentalNet
            character_embedding: Character embedding

        Returns:
            Mental state embedding
        """
        with torch.no_grad():
            # Need to concatenate trajectory with character embedding like in forward()
            batch_size, seq_len, channels, height, width = recent_trajectory.shape
            e_char_spatial = character_embedding.unsqueeze(1).unsqueeze(3).unsqueeze(4)
            e_char_spatial = e_char_spatial.expand(
                batch_size, seq_len, self.n_echar, height, width
            )
            recent_trajectory_with_char = torch.cat(
                [recent_trajectory, e_char_spatial], dim=2
            )
            mental_state, _ = self.mental_net(recent_trajectory_with_char)
            return mental_state


class ToMnetLoss(nn.Module):
    def __init__(
        self, action_weight=1.0, goal_weight=1.0, consumption_weight=1.0, sr_weight=1.0
    ):
        super(ToMnetLoss, self).__init__()
        self.action_weight = action_weight
        self.goal_weight = goal_weight
        self.consumption_weight = consumption_weight
        self.sr_weight = sr_weight
        self.action_loss = nn.CrossEntropyLoss()
        self.goal_loss = nn.CrossEntropyLoss()
        self.consumption_loss = nn.BCEWithLogitsLoss()  # For consumption prediction

    def forward(
        self,
        action_logits,
        goal_logits,
        consumption_logits,
        sr_pred,
        action_targets,
        goal_targets,
        consumption_targets,
        sr_targets,
    ):
        """
        Compute combined loss (following experiment 5 approach)

        Args:
            action_logits: Predicted action logits
            goal_logits: Predicted goal logits
            consumption_logits: Predicted consumption logits (batch_size, 8)
            sr_pred: Predicted SR maps (batch_size, 3, height, width)
            action_targets: True action labels
            goal_targets: True goal labels
            consumption_targets: True consumption labels (batch_size, 8)
            sr_targets: True SR maps (batch_size, 3, height, width)

        Returns:
            total_loss: Combined loss
            action_loss: Action prediction loss
            goal_loss: Goal prediction loss
            consumption_loss: Consumption prediction loss
            sr_loss: SR prediction loss
        """
        action_loss = self.action_loss(action_logits, action_targets)
        goal_loss = self.goal_loss(goal_logits, goal_targets)

        # Consumption loss (use sigmoid + BCE for multi-label classification)
        consumption_loss = self.consumption_loss(
            consumption_logits, consumption_targets
        )

        # SR loss using KL divergence
        from train import calculate_sr_loss_kl_divergence

        sr_loss = calculate_sr_loss_kl_divergence(sr_pred, sr_targets)

        total_loss = (
            self.action_weight * action_loss
            + self.goal_weight * goal_loss
            + self.consumption_weight * consumption_loss
            + self.sr_weight * sr_loss
        )

        return total_loss, action_loss, goal_loss, consumption_loss, sr_loss


# Utility functions
def create_model(config):
    """
    Create ToMnet model from configuration

    Args:
        config: Configuration dictionary

    Returns:
        ToMnet model
    """
    model = ToMnet(
        batch=config.get("batch", 32),
        residual_blocks=config.get("residual_blocks", 3),
        n_echar=config.get("n_echar", 64),
        n_ement=config.get("n_ement", 64),
        out_channels=config.get("out_channels", 32),
        channels_in=config.get("channels_in", 8),
        current_state_channels=config.get("current_state_channels", 8),
        time_step=config.get("time_step", 500),
        action_space=config.get("action_space", 7),
        goal_space=config.get("goal_space", 4),
        max_n_past=config.get("max_n_past", 10),
        use_n_past=config.get("use_n_past", True),
        env_width=config.get("env_width", 9),
        env_height=config.get("env_height", 9),
        hidden_size_lstm=config.get("hidden_size_lstm", 64),
    )

    return model


def count_parameters(model):
    """Count the number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Example usage
if __name__ == "__main__":
    # Example configuration
    config = {
        "batch_size": 32,
        "residual_blocks": 3,
        "n_echar": 64,
        "n_ement": 64,
        "out_channels": 32,
        "channels_in": 9,  # 8 original channels + 1 heading direction channel
        "time_step": 100,
        "action_space": 7,
        "goal_space": 4,
        "max_n_past": 5,
        "use_n_past": True,
    }

    # Create model
    model = create_model(config)
    print(f"Model created with {count_parameters(model)} parameters")

    # Example input shapes
    batch_size = 8
    n_past = 3
    seq_len = 50
    channels = 9  # 8 original channels + 1 heading direction channel
    height = 9
    width = 9

    # Create dummy data
    past_trajectories = torch.randn(
        batch_size, n_past, seq_len, channels, height, width
    )
    current_trajectory = torch.randn(batch_size, seq_len, channels, height, width)

    # Forward pass
    action_logits, goal_logits, consumption_logits, sr_pred, char_emb, mental_state = (
        model(past_trajectories, current_trajectory)
    )

    print(f"Action logits shape: {action_logits.shape}")
    print(f"Goal logits shape: {goal_logits.shape}")
    print(f"Consumption logits shape: {consumption_logits.shape}")
    print(f"SR prediction shape: {sr_pred.shape}")
    print(f"Character embedding shape: {char_emb.shape}")
    print(f"Mental state shape: {mental_state.shape}")

    # Test loss computation
    action_targets = torch.randint(0, 7, (batch_size,))
    goal_targets = torch.randint(0, 4, (batch_size,))
    consumption_targets = torch.randint(0, 2, (batch_size, 8)).float()  # Binary targets
    sr_targets = torch.rand(batch_size, 3, height, width)  # Random SR targets

    loss_fn = ToMnetLoss()
    total_loss, action_loss, goal_loss, consumption_loss, sr_loss = loss_fn(
        action_logits,
        goal_logits,
        consumption_logits,
        sr_pred,
        action_targets,
        goal_targets,
        consumption_targets,
        sr_targets,
    )

    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Action loss: {action_loss.item():.4f}")
    print(f"Goal loss: {goal_loss.item():.4f}")
    print(f"Consumption loss: {consumption_loss.item():.4f}")
    print(f"SR loss: {sr_loss.item():.4f}")
