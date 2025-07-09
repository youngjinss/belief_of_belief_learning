import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

"""
ToMnet architecture for KeyDoor environment (experiment 3)
Adapted from ToMnetF experiment5 for KeyDoor environment
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

        # Skip connection adjustment if needed
        if in_channels != out_channels:
            self.skip_connection = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride
            )
        else:
            self.skip_connection = None

    def forward(self, x):
        residual = x

        # Apply skip connection if channels don't match
        if self.skip_connection is not None:
            residual = self.skip_connection(residual)

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
        Batch: int,
        ResidualBlocks: int,
        N_echar: int,
        out_channels: int,
        channels_in: int,
        time_step: int,
        max_n_past: int = 10,
        use_n_past: bool = True,
    ):
        super(CharNet, self).__init__()

        self.n = ResidualBlocks
        self.N_echar = N_echar
        self.out_channels = out_channels
        self.channels_in = channels_in
        self.B = Batch
        self.time_step = time_step
        self.hidden_size_lstm = 64
        self.max_n_past = max_n_past
        self.use_n_past = use_n_past

        # Convolutional layers for spatial feature extraction
        self.conv_layers = nn.ModuleList()

        # Initial convolution
        self.conv_layers.append(
            nn.Conv2d(channels_in, out_channels, kernel_size=3, padding=1)
        )

        # Residual blocks
        for _ in range(ResidualBlocks):
            self.conv_layers.append(
                ResidualBlock(out_channels, out_channels, kernel_size=3, padding=1)
            )

        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # LSTM for temporal processing
        self.lstm = LSTM(out_channels, self.hidden_size_lstm)

        # Character embedding output
        self.char_embedding = nn.Linear(self.hidden_size_lstm, N_echar)

    def forward(self, past_trajectories):
        """
        Forward pass for character network

        Args:
            past_trajectories: (batch_size, n_past, seq_len, channels, height, width)

        Returns:
            Character embeddings: (batch_size, N_echar)
        """
        batch_size = past_trajectories.size(0)
        n_past = past_trajectories.size(1)
        seq_len = past_trajectories.size(2)

        # Reshape for processing
        # (batch_size * n_past * seq_len, channels, height, width)
        x = past_trajectories.reshape(-1, self.channels_in, 9, 9)

        # Extract spatial features
        for layer in self.conv_layers:
            x = layer(x)

        # Global average pooling
        x = self.global_avg_pool(
            x
        )  # (batch_size * n_past * seq_len, out_channels, 1, 1)
        x = x.reshape(batch_size * n_past, seq_len, self.out_channels)

        # LSTM processing for each past episode
        episode_embeddings = []
        for i in range(n_past):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            episode_data = x[start_idx:end_idx]  # (batch_size, seq_len, out_channels)
            episode_emb = self.lstm(episode_data)  # (batch_size, hidden_size)
            episode_embeddings.append(episode_emb)

        # Stack and average episode embeddings
        episode_embeddings = torch.stack(
            episode_embeddings, dim=1
        )  # (batch_size, n_past, hidden_size)
        char_features = torch.mean(
            episode_embeddings, dim=1
        )  # (batch_size, hidden_size)

        # Character embedding
        char_embedding = self.char_embedding(char_features)

        return char_embedding


class MentalNet(nn.Module):
    def __init__(
        self,
        Batch: int,
        ResidualBlocks: int,
        N_ement: int,
        out_channels: int,
        channels_in: int,
        time_step: int,
        N_echar: int,
    ):
        super(MentalNet, self).__init__()

        self.n = ResidualBlocks
        self.N_ement = N_ement
        self.out_channels = out_channels
        self.channels_in = channels_in
        self.B = Batch
        self.time_step = time_step
        self.N_echar = N_echar
        self.hidden_size_lstm = 64

        # Convolutional layers for spatial feature extraction
        self.conv_layers = nn.ModuleList()

        # Initial convolution
        self.conv_layers.append(
            nn.Conv2d(channels_in, out_channels, kernel_size=3, padding=1)
        )

        # Residual blocks
        for _ in range(ResidualBlocks):
            self.conv_layers.append(
                ResidualBlock(out_channels, out_channels, kernel_size=3, padding=1)
            )

        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # LSTM for temporal processing
        self.lstm = LSTM(out_channels, self.hidden_size_lstm)

        # Mental state network
        self.mental_state_net = nn.Sequential(
            nn.Linear(self.hidden_size_lstm + N_echar, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, N_ement),
        )

    def forward(self, current_trajectory, character_embedding):
        """
        Forward pass for mental state network

        Args:
            current_trajectory: (batch_size, seq_len, channels, height, width)
            character_embedding: (batch_size, N_echar)

        Returns:
            Mental state embedding: (batch_size, N_ement)
        """
        batch_size = current_trajectory.size(0)
        seq_len = current_trajectory.size(1)

        # Reshape for processing
        x = current_trajectory.reshape(-1, self.channels_in, 9, 9)

        # Extract spatial features
        for layer in self.conv_layers:
            x = layer(x)

        # Global average pooling
        x = self.global_avg_pool(x)  # (batch_size * seq_len, out_channels, 1, 1)
        x = x.reshape(batch_size, seq_len, self.out_channels)

        # LSTM processing
        trajectory_features = self.lstm(x)  # (batch_size, hidden_size)

        # Combine with character embedding
        combined_features = torch.cat([trajectory_features, character_embedding], dim=1)

        # Mental state prediction
        mental_state = self.mental_state_net(combined_features)

        return mental_state


class PredNet(nn.Module):
    def __init__(
        self,
        Batch: int,
        N_ement: int,
        N_echar: int,
        action_space: int = 7,
        goal_space: int = 4,
    ):
        super(PredNet, self).__init__()

        self.B = Batch
        self.N_ement = N_ement
        self.N_echar = N_echar
        self.action_space = action_space
        self.goal_space = goal_space

        # Action prediction network
        self.action_predictor = nn.Sequential(
            nn.Linear(N_ement + N_echar, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_space),
        )

        # Goal prediction network
        self.goal_predictor = nn.Sequential(
            nn.Linear(N_ement + N_echar, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, goal_space),
        )

    def forward(self, mental_state, character_embedding):
        """
        Forward pass for prediction network

        Args:
            mental_state: (batch_size, N_ement)
            character_embedding: (batch_size, N_echar)

        Returns:
            action_logits: (batch_size, action_space)
            goal_logits: (batch_size, goal_space)
        """
        # Combine mental state and character embedding
        combined = torch.cat([mental_state, character_embedding], dim=1)

        # Predict actions and goals
        action_logits = self.action_predictor(combined)
        goal_logits = self.goal_predictor(combined)

        return action_logits, goal_logits


class ToMnet(nn.Module):
    def __init__(
        self,
        Batch: int = 32,
        ResidualBlocks: int = 3,
        N_echar: int = 64,
        N_ement: int = 64,
        out_channels: int = 32,
        channels_in: int = 8,
        time_step: int = 500,
        action_space: int = 7,
        goal_space: int = 4,
        max_n_past: int = 10,
        use_n_past: bool = True,
    ):
        super(ToMnet, self).__init__()

        self.B = Batch
        self.N_echar = N_echar
        self.N_ement = N_ement
        self.time_step = time_step
        self.action_space = action_space
        self.goal_space = goal_space
        self.max_n_past = max_n_past
        self.use_n_past = use_n_past

        # Character network
        self.char_net = CharNet(
            Batch=Batch,
            ResidualBlocks=ResidualBlocks,
            N_echar=N_echar,
            out_channels=out_channels,
            channels_in=channels_in,
            time_step=time_step,
            max_n_past=max_n_past,
            use_n_past=use_n_past,
        )

        # Mental state network
        self.mental_net = MentalNet(
            Batch=Batch,
            ResidualBlocks=ResidualBlocks,
            N_ement=N_ement,
            out_channels=out_channels,
            channels_in=channels_in,
            time_step=time_step,
            N_echar=N_echar,
        )

        # Prediction network
        self.pred_net = PredNet(
            Batch=Batch,
            N_ement=N_ement,
            N_echar=N_echar,
            action_space=action_space,
            goal_space=goal_space,
        )

    def forward(self, past_trajectories, current_trajectory):
        """
        Forward pass for ToMnet

        Args:
            past_trajectories: (batch_size, n_past, seq_len, channels, height, width)
            current_trajectory: (batch_size, seq_len, channels, height, width)

        Returns:
            action_logits: (batch_size, action_space)
            goal_logits: (batch_size, goal_space)
            character_embedding: (batch_size, N_echar)
            mental_state: (batch_size, N_ement)
        """
        # Character network
        if self.use_n_past and past_trajectories is not None:
            character_embedding = self.char_net(past_trajectories)
        else:
            # Use zero embedding if no past trajectories
            batch_size = current_trajectory.size(0)
            character_embedding = torch.zeros(
                batch_size, self.N_echar, device=current_trajectory.device
            )

        # Mental state network
        mental_state = self.mental_net(current_trajectory, character_embedding)

        # Prediction network
        action_logits, goal_logits = self.pred_net(mental_state, character_embedding)

        return action_logits, goal_logits, character_embedding, mental_state

    def predict_action(self, past_trajectories, current_trajectory):
        """
        Predict next action

        Args:
            past_trajectories: Past episode trajectories
            current_trajectory: Current episode trajectory

        Returns:
            Predicted action probabilities
        """
        with torch.no_grad():
            action_logits, _, _, _ = self.forward(past_trajectories, current_trajectory)
            return F.softmax(action_logits, dim=1)

    def predict_goal(self, past_trajectories, current_trajectory):
        """
        Predict goal

        Args:
            past_trajectories: Past episode trajectories
            current_trajectory: Current episode trajectory

        Returns:
            Predicted goal probabilities
        """
        with torch.no_grad():
            _, goal_logits, _, _ = self.forward(past_trajectories, current_trajectory)
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
                return torch.zeros(batch_size, self.N_echar)

    def get_mental_state(self, current_trajectory, character_embedding):
        """
        Get mental state from current trajectory and character embedding

        Args:
            current_trajectory: Current episode trajectory
            character_embedding: Character embedding

        Returns:
            Mental state embedding
        """
        with torch.no_grad():
            return self.mental_net(current_trajectory, character_embedding)


class ToMnetLoss(nn.Module):
    def __init__(self, action_weight=1.0, goal_weight=1.0):
        super(ToMnetLoss, self).__init__()
        self.action_weight = action_weight
        self.goal_weight = goal_weight
        self.action_loss = nn.CrossEntropyLoss()
        self.goal_loss = nn.CrossEntropyLoss()

    def forward(self, action_logits, goal_logits, action_targets, goal_targets):
        """
        Compute combined loss

        Args:
            action_logits: Predicted action logits
            goal_logits: Predicted goal logits
            action_targets: True action labels
            goal_targets: True goal labels

        Returns:
            total_loss: Combined loss
            action_loss: Action prediction loss
            goal_loss: Goal prediction loss
        """
        action_loss = self.action_loss(action_logits, action_targets)
        goal_loss = self.goal_loss(goal_logits, goal_targets)

        total_loss = self.action_weight * action_loss + self.goal_weight * goal_loss

        return total_loss, action_loss, goal_loss


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
        Batch=config.get("batch_size", 32),
        ResidualBlocks=config.get("residual_blocks", 3),
        N_echar=config.get("n_echar", 64),
        N_ement=config.get("n_ement", 64),
        out_channels=config.get("out_channels", 32),
        channels_in=config.get("channels_in", 8),
        time_step=config.get("time_step", 500),
        action_space=config.get("action_space", 7),
        goal_space=config.get("goal_space", 4),
        max_n_past=config.get("max_n_past", 10),
        use_n_past=config.get("use_n_past", True),
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
        "channels_in": 8,
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
    channels = 8
    height = 9
    width = 9

    # Create dummy data
    past_trajectories = torch.randn(
        batch_size, n_past, seq_len, channels, height, width
    )
    current_trajectory = torch.randn(batch_size, seq_len, channels, height, width)

    # Forward pass
    action_logits, goal_logits, char_emb, mental_state = model(
        past_trajectories, current_trajectory
    )

    print(f"Action logits shape: {action_logits.shape}")
    print(f"Goal logits shape: {goal_logits.shape}")
    print(f"Character embedding shape: {char_emb.shape}")
    print(f"Mental state shape: {mental_state.shape}")

    # Test loss computation
    action_targets = torch.randint(0, 7, (batch_size,))
    goal_targets = torch.randint(0, 4, (batch_size,))

    loss_fn = ToMnetLoss()
    total_loss, action_loss, goal_loss = loss_fn(
        action_logits, goal_logits, action_targets, goal_targets
    )

    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Action loss: {action_loss.item():.4f}")
    print(f"Goal loss: {goal_loss.item():.4f}")
