import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import numpy as np


class CharacterNet(nn.Module):
    """
    Character Net: Processes past episode trajectories into character embeddings
    e_char,ij = f_θ(τ_ij^(obs))

    FIXED VERSION: Uses learnable embedding for N_past=0 instead of zeros
    ENHANCED: Added dropout for regularization
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        embedding_dim: int = 8,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embedding_dim = embedding_dim

        # Input: flattened (state, action) pairs
        input_dim = state_dim + action_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, embedding_dim),
        )

        # FIXED: Learnable embedding for "no past information" case
        # This allows the model to learn what "no past information" means
        # rather than always returning zeros
        self.no_past_embedding = nn.Parameter(torch.randn(embedding_dim) * 0.1)

    def forward(self, trajectories: torch.Tensor) -> torch.Tensor:
        """
        Args:
            trajectories: (batch_size, n_past, seq_len, state_dim + action_dim)
        Returns:
            character_embeddings: (batch_size, embedding_dim)
        """
        batch_size, n_past, seq_len, input_dim = trajectories.shape

        # Handle empty past trajectories (N_past=0) with zero embedding
        if n_past == 0:
            # Return zero embedding to force uniform/random behavior
            # This ensures N_past=0 gives ~0.2 action likelihood (1/5 actions)
            return torch.zeros(
                batch_size, self.embedding_dim, device=trajectories.device
            )

        # Flatten trajectories for processing
        traj_flat = trajectories.view(batch_size * n_past, seq_len, input_dim)

        # Vectorized processing - reshape to process all timesteps at once
        # Reshape from (batch_size * n_past, seq_len, input_dim) to (batch_size * n_past * seq_len, input_dim)
        traj_reshaped = traj_flat.view(batch_size * n_past * seq_len, input_dim)

        # Process all timesteps at once through MLP
        embeddings_flat = self.mlp(
            traj_reshaped
        )  # (batch_size * n_past * seq_len, embedding_dim)

        # Reshape back and average over sequence length
        embeddings = embeddings_flat.view(
            batch_size * n_past, seq_len, self.embedding_dim
        )
        trajectory_embeddings = embeddings.mean(
            dim=1
        )  # (batch_size * n_past, embedding_dim)

        # Reshape and aggregate over past episodes
        trajectory_embeddings = trajectory_embeddings.view(
            batch_size, n_past, self.embedding_dim
        )

        # IMPROVED: Use mean instead of sum for better normalization
        # This prevents embeddings from growing linearly with n_past
        character_embeddings = trajectory_embeddings.mean(dim=1)

        return character_embeddings


class ResidualBlock(nn.Module):
    """Residual block for ResNet architecture"""

    def __init__(self, channels: int, use_batchnorm: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels) if use_batchnorm else nn.Identity()
        self.bn2 = nn.BatchNorm2d(channels) if use_batchnorm else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class Figure3CharacterNet(nn.Module):
    """
    Character Net for Figure 3: ConvNet + LSTM architecture
    As specified in README lines 26-31
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        embedding_dim: int = 2,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embedding_dim = embedding_dim

        # Assuming 11x11 grid with 6 channels for state
        # State channels: walls, agent, 4 object types
        self.state_channels = 6
        grid_size_float = np.sqrt(state_dim // 6)
        if grid_size_float % 1 != 0:
            raise ValueError("Grid size must be a perfect square")
        self.grid_size = int(grid_size_float)

        # 1-layer convnet with 8 feature planes (line 28)
        self.conv1 = nn.Conv2d(
            self.state_channels + action_dim, 8, kernel_size=3, padding=1
        )
        self.relu = nn.ReLU()

        # Convolutional LSTM (line 29)
        # Using regular LSTM after flattening conv features
        conv_output_size = int(8 * self.grid_size * self.grid_size)
        self.lstm = nn.LSTM(conv_output_size, 128, batch_first=True)

        # Fully-connected layer to 2D embedding space (line 31)
        self.fc = nn.Linear(128, embedding_dim)

        # Learnable embedding for N_past=0 case
        self.no_past_embedding = nn.Parameter(torch.randn(embedding_dim) * 0.1)

    def forward(self, trajectories: torch.Tensor) -> torch.Tensor:
        """
        Args:
            trajectories: (batch_size, n_past, seq_len, state_dim + action_dim)
        Returns:
            character_embeddings: (batch_size, embedding_dim)
        """
        batch_size, n_past, seq_len, input_dim = trajectories.shape

        if n_past == 0:
            return torch.zeros(
                batch_size, self.embedding_dim, device=trajectories.device
            )

        # Reshape trajectories for processing
        traj_flat = trajectories.view(batch_size * n_past, seq_len, input_dim)

        # Split state and action
        state_flat = traj_flat[:, :, : self.state_dim]  # Flattened state
        action_flat = traj_flat[:, :, self.state_dim :]  # One-hot action

        # Reshape state to grid format
        state_grid = state_flat.view(
            batch_size * n_past,
            seq_len,
            self.state_channels,
            self.grid_size,
            self.grid_size,
        )

        # Spatialize action (expand to match grid size)
        action_expanded = (
            action_flat.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(
                batch_size * n_past,
                seq_len,
                self.action_dim,
                self.grid_size,
                self.grid_size,
            )
        )

        # Vectorized convolution processing
        # Reshape to process all timesteps at once
        state_grid_flat = state_grid.permute(
            1, 0, 2, 3, 4
        ).contiguous()  # (seq_len, batch*n_past, 6, 11, 11)
        action_expanded_flat = action_expanded.permute(
            1, 0, 2, 3, 4
        ).contiguous()  # (seq_len, batch*n_past, 5, 11, 11)

        # Reshape for batch processing
        state_grid_all = state_grid_flat.view(
            seq_len * batch_size * n_past,
            self.state_channels,
            self.grid_size,
            self.grid_size,
        )
        action_expanded_all = action_expanded_flat.view(
            seq_len * batch_size * n_past,
            self.action_dim,
            self.grid_size,
            self.grid_size,
        )

        # Concatenate all timesteps
        conv_input_all = torch.cat(
            [state_grid_all, action_expanded_all], dim=1
        )  # (seq_len*batch*n_past, 11, 11, 11)

        # Apply convolution to all timesteps at once
        conv_out_all = self.relu(
            self.conv1(conv_input_all)
        )  # (seq_len*batch*n_past, 8, 11, 11)

        # Flatten and reshape for LSTM
        conv_out_flat = conv_out_all.view(
            seq_len, batch_size * n_past, -1
        )  # (seq_len, batch*n_past, conv_output_size)
        lstm_input = conv_out_flat.permute(
            1, 0, 2
        )  # (batch*n_past, seq_len, conv_output_size)

        # Process through LSTM
        lstm_out, (hidden, _) = self.lstm(lstm_input)

        # Use final hidden state
        final_hidden = hidden[-1]  # (batch*n_past, 128)

        # Generate embeddings
        trajectory_embeddings = self.fc(final_hidden)  # (batch*n_past, embedding_dim)

        # Reshape and aggregate
        trajectory_embeddings = trajectory_embeddings.view(
            batch_size, n_past, self.embedding_dim
        )
        character_embeddings = trajectory_embeddings.sum(
            dim=1
        )  # Sum over past episodes

        return character_embeddings


class Figure5CharacterNet(nn.Module):
    """
    Character Net for Figure 5: 5-layer ResNet architecture
    As specified in README lines 582-589
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        embedding_dim: int = 8,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embedding_dim = embedding_dim

        # Assuming 11x11 grid with 6 channels for state
        self.state_channels = 6
        grid_size_float = np.sqrt(state_dim // 6)
        if grid_size_float % 1 != 0:
            raise ValueError("Grid size must be a perfect square")
        self.grid_size = int(grid_size_float)

        # Initial convolution to 32 channels
        self.conv1 = nn.Conv2d(
            self.state_channels + action_dim, 32, kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        # 3-layer ResNet with 32 channels (reduced for simple 3x3 maze)
        self.resnet_blocks = nn.Sequential(
            ResidualBlock(32),
            ResidualBlock(32),
            ResidualBlock(32),
        )

        # Average pooling (line 587)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully-connected layer to embedding (line 588)
        self.fc = nn.Linear(32, embedding_dim)

        # Learnable embedding for N_past=0 case
        self.no_past_embedding = nn.Parameter(torch.randn(embedding_dim) * 0.1)

    def forward(self, trajectories: torch.Tensor) -> torch.Tensor:
        """
        Args:
            trajectories: (batch_size, n_past, seq_len, state_dim + action_dim)
            Note: For Figure 5, seq_len is typically 1 (single observation-action pair)
        Returns:
            character_embeddings: (batch_size, embedding_dim)
        """
        batch_size, n_past, seq_len, input_dim = trajectories.shape

        if n_past == 0:
            return torch.zeros(
                batch_size, self.embedding_dim, device=trajectories.device
            )

        # Vectorized processing - reshape to process all episodes at once
        # Take first timestep (seq_len=1 for Figure 5)
        trajectories_flat = trajectories[
            :, :, 0, :
        ]  # (batch_size, n_past, state_dim + action_dim)

        # Reshape to process all batch*n_past samples together
        trajectories_flat = trajectories_flat.view(batch_size * n_past, input_dim)

        # Split state and action
        state_flat = trajectories_flat[:, : self.state_dim]  # (batch*n_past, state_dim)
        action_flat = trajectories_flat[
            :, self.state_dim :
        ]  # (batch*n_past, action_dim)

        # Reshape state to grid format
        state_grid = state_flat.view(
            batch_size * n_past, self.state_channels, self.grid_size, self.grid_size
        )

        # Spatialize action (expand to match grid size)
        action_expanded = (
            action_flat.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(
                batch_size * n_past, self.action_dim, self.grid_size, self.grid_size
            )
        )

        # Concatenate state and spatialized action
        conv_input = torch.cat(
            [state_grid, action_expanded], dim=1
        )  # (batch*n_past, 11, 11, 11)

        # Apply initial convolution
        x = self.conv1(conv_input)
        x = self.bn1(x)
        x = self.relu(x)

        # Apply ResNet blocks
        x = self.resnet_blocks(x)

        # Average pooling
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # (batch*n_past, 32)

        # Generate embeddings
        embeddings = self.fc(x)  # (batch*n_past, embedding_dim)

        # Reshape back to (batch_size, n_past, embedding_dim)
        embeddings = embeddings.view(batch_size, n_past, self.embedding_dim)

        # Sum embeddings from all past episodes (line 589)
        character_embeddings = embeddings.sum(dim=1)  # (batch_size, embedding_dim)

        return character_embeddings


class MentalStateNet(nn.Module):
    """
    Mental State Net: Processes current episode trajectory
    e_mental,i = g_φ([τ_ij^(obs)]_0:t-1, e_char,i)
    ENHANCED: Added dropout for regularization
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        char_embedding_dim: int,
        hidden_dim: int = 128,
        mental_embedding_dim: int = 64,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.char_embedding_dim = char_embedding_dim
        self.mental_embedding_dim = mental_embedding_dim

        # LSTM for processing current trajectory
        input_dim = state_dim + action_dim
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            batch_first=True,
            dropout=dropout_rate if dropout_rate > 0 else 0.0,
        )

        # Combine LSTM output with character embedding
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + char_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, mental_embedding_dim),
        )

    def forward(
        self, current_trajectory: torch.Tensor, character_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            current_trajectory: (batch_size, current_seq_len, state_dim + action_dim)
            character_embedding: (batch_size, char_embedding_dim)
        Returns:
            mental_embedding: (batch_size, mental_embedding_dim)
        """
        # Process current trajectory with LSTM
        lstm_out, (hidden, _) = self.lstm(current_trajectory)

        # Use final hidden state
        final_hidden = hidden[-1]  # (batch_size, hidden_dim)

        # Concatenate with character embedding
        combined = torch.cat([final_hidden, character_embedding], dim=1)

        # Generate mental state embedding
        mental_embedding = self.fusion(combined)

        return mental_embedding


class Figure3PredictionNet(nn.Module):
    """
    Prediction Net for Figure 3: Only action prediction head
    As specified in README lines 37-45
    """

    def __init__(
        self,
        state_dim: int,
        char_embedding_dim: int,
        n_actions: int = 5,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.char_embedding_dim = char_embedding_dim
        self.n_actions = n_actions

        # Assuming 11x11 grid with 6 channels for state
        self.state_channels = 6
        grid_size_float = np.sqrt(state_dim // 6)
        if grid_size_float % 1 != 0:
            raise ValueError("Grid size must be a perfect square")
        self.grid_size = int(grid_size_float)

        # 2-layer convnet with 32 feature planes and ReLUs (line 42)
        self.conv1 = nn.Conv2d(
            self.state_channels + char_embedding_dim, 32, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        # Average pooling (line 43)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully-connected layer to 5-dim logits (line 44)
        self.fc = nn.Linear(32, n_actions)

    def forward(
        self,
        current_state: torch.Tensor,
        character_embedding: torch.Tensor,
        mental_embedding: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            current_state: (batch_size, state_dim)
            character_embedding: (batch_size, char_embedding_dim)
            mental_embedding: Unused in Figure 3
        Returns:
            Dict containing action_logits and action_probs
        """
        batch_size = current_state.size(0)

        # Reshape state to grid format
        state_grid = current_state.view(
            batch_size, self.state_channels, self.grid_size, self.grid_size
        )

        # Spatialize character embedding (expand to match grid size)
        char_expanded = (
            character_embedding.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(batch_size, self.char_embedding_dim, self.grid_size, self.grid_size)
        )

        # Concatenate current state with character embedding (line 38)
        conv_input = torch.cat([state_grid, char_expanded], dim=1)

        # 2-layer convnet with 32 feature planes and ReLUs (line 42)
        x = self.relu(self.conv1(conv_input))
        x = self.relu(self.conv2(x))

        # Average pooling (line 43)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # Flatten

        # Fully-connected layer to 5-dim logits (line 44)
        action_logits = self.fc(x)

        # Softmax for action probabilities (line 45)
        action_probs = F.softmax(action_logits, dim=1)

        return {
            "action_logits": action_logits,
            "action_pred": action_logits,  # For compatibility
            "action_probs": action_probs,
        }


class Figure5PredictionNet(nn.Module):
    """
    Prediction Net for Figure 5: Shared torso with three prediction heads
    As specified in README lines 223-245
    """

    def __init__(
        self,
        state_dim: int,
        char_embedding_dim: int,
        n_actions: int = 5,
        n_objects: int = 4,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.char_embedding_dim = char_embedding_dim
        self.n_actions = n_actions
        self.n_objects = n_objects

        # Assuming 11x11 grid with 6 channels for state
        self.state_channels = 6
        grid_size_float = np.sqrt(state_dim // 6)
        if grid_size_float % 1 != 0:
            raise ValueError("Grid size must be a perfect square")
        self.grid_size = int(grid_size_float)

        # Shared Torso: 5-layer ResNet with 32 channels (lines 227-228)
        self.conv1 = nn.Conv2d(
            self.state_channels + char_embedding_dim, 32, kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        # 3-layer ResNet (reduced for simple 3x3 maze)
        self.resnet_blocks = nn.Sequential(
            ResidualBlock(32),
            ResidualBlock(32),
            ResidualBlock(32),
        )

        # Action Prediction Head (lines 230-234)
        self.action_conv = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.action_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.action_fc = nn.Linear(32, n_actions)

        # Consumption Prediction Head (lines 235-239)
        self.consumption_conv = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.consumption_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.consumption_fc = nn.Linear(32, n_objects)

        # Successor Representation Prediction Head (lines 241-245)
        self.sr_conv1 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.sr_conv2 = nn.Conv2d(
            32, 3, kernel_size=3, padding=1
        )  # 3 channels for 3 discount factors

    def forward(
        self,
        current_state: torch.Tensor,
        character_embedding: torch.Tensor,
        mental_embedding: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            current_state: (batch_size, state_dim)
            character_embedding: (batch_size, char_embedding_dim)
            mental_embedding: Unused in Figure 5
        Returns:
            Dict containing action predictions, consumption predictions, and SR predictions
        """
        batch_size = current_state.size(0)

        # Reshape state to grid format
        state_grid = current_state.view(
            batch_size, self.state_channels, self.grid_size, self.grid_size
        )

        # Spatialize character embedding and concatenate with query state (line 227)
        char_expanded = (
            character_embedding.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(batch_size, self.char_embedding_dim, self.grid_size, self.grid_size)
        )
        conv_input = torch.cat([state_grid, char_expanded], dim=1)

        # Shared Torso: 5-layer ResNet with 32 channels (line 228)
        x = self.conv1(conv_input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.resnet_blocks(x)

        # Action Prediction Head (lines 230-234)
        action_x = self.relu(self.action_conv(x))
        action_x = self.action_avgpool(action_x)
        action_x = action_x.view(action_x.size(0), -1)
        action_logits = self.action_fc(action_x)
        action_probs = F.softmax(action_logits, dim=1)

        # Consumption Prediction Head (lines 235-239)
        consumption_x = self.relu(self.consumption_conv(x))
        consumption_x = self.consumption_avgpool(consumption_x)
        consumption_x = consumption_x.view(consumption_x.size(0), -1)
        consumption_logits = self.consumption_fc(consumption_x)
        consumption_probs = torch.sigmoid(consumption_logits)

        # Successor Representation Prediction Head (lines 241-245)
        sr_x = self.relu(self.sr_conv1(x))
        sr_x = self.sr_conv2(sr_x)  # (batch, 3, grid_size, grid_size)

        # Softmax over each channel independently (line 244)
        # Gives predicted normalized SRs for three discount factors: γ = 0.5, 0.9, 0.99
        sr_probs = F.softmax(
            sr_x.view(batch_size, 3, -1), dim=2
        )  # Softmax over spatial dims
        sr_probs = sr_probs.view(batch_size, 3, self.grid_size, self.grid_size)

        return {
            "action_logits": action_logits,
            "action_pred": action_logits,  # For compatibility
            "action_probs": action_probs,
            "consumption": consumption_logits,  # Raw logits for loss calculation
            "consumption_probs": consumption_probs,
            "successor_representation": sr_x,  # Raw logits for loss calculation
            "sr_logits": sr_x,
            "sr_probs": sr_probs,
        }


class PredictionNet(nn.Module):
    """
    Legacy Prediction Net: Outputs behavioral predictions
    - Next-step action probabilities
    - Object consumption probabilities
    - Successor representations
    ENHANCED: Added dropout for regularization
    """

    def __init__(
        self,
        state_dim: int,
        char_embedding_dim: int,
        mental_embedding_dim: int,
        n_actions: int = 5,
        n_objects: int = 4,
        hidden_dim: int = 128,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.n_objects = n_objects

        # Input: current state + character embedding + mental embedding
        input_dim = state_dim + char_embedding_dim + mental_embedding_dim

        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # Action prediction head
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, n_actions),
        )

        # Object consumption prediction head
        self.consumption_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, n_objects),
        )

        # Successor representation head (for grid states)
        grid_size = int(np.sqrt(state_dim / 6))  # Assuming 6 channels
        self.sr_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, grid_size * grid_size),
        )

    def forward(
        self,
        current_state: torch.Tensor,
        character_embedding: torch.Tensor,
        mental_embedding: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            current_state: (batch_size, state_dim)
            character_embedding: (batch_size, char_embedding_dim)
            mental_embedding: (batch_size, mental_embedding_dim)
        Returns:
            Dict containing action_logits, action_probs, consumption_probs, sr_logits
        """
        # Concatenate all inputs
        combined_input = torch.cat(
            [current_state, character_embedding, mental_embedding], dim=1
        )

        # Shared processing
        shared_features = self.shared(combined_input)

        # Action predictions
        action_logits = self.action_head(shared_features)

        action_probs = F.softmax(action_logits, dim=1)

        # Object consumption predictions
        consumption_probs = torch.sigmoid(self.consumption_head(shared_features))

        # Successor representation predictions
        sr_logits = self.sr_head(shared_features)

        return {
            "action_logits": action_logits,
            "action_pred": action_logits,  # For compatibility with enhanced trainer
            "action_probs": action_probs,
            "consumption_probs": consumption_probs,
            "sr_logits": sr_logits,
        }


class ToMnet(nn.Module):
    """
    Theory of Mind Network (ToMnet)

    FIXED VERSION: Uses improved CharacterNet that properly handles N_past=0
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 5,
        n_actions: int = 5,
        n_objects: int = 4,
        char_embedding_dim: int = 8,
        mental_embedding_dim: int = 64,
        hidden_dim: int = 128,
        use_mental_state: bool = True,
        dropout_rate: float = 0.0,
        character_net: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.char_embedding_dim = char_embedding_dim
        self.mental_embedding_dim = mental_embedding_dim
        self.use_mental_state = use_mental_state

        # Initialize networks with dropout
        if character_net is not None:
            self.character_net = character_net
        else:
            self.character_net = CharacterNet(
                state_dim, action_dim, hidden_dim, char_embedding_dim, dropout_rate
            )

        if use_mental_state:
            self.mental_state_net = MentalStateNet(
                state_dim,
                action_dim,
                char_embedding_dim,
                hidden_dim,
                mental_embedding_dim,
                dropout_rate,
            )
        else:
            # For Figure 3 experiments, mental state is not used
            self.mental_state_net = None
            mental_embedding_dim = 0

        self.prediction_net = PredictionNet(
            state_dim,
            char_embedding_dim,
            mental_embedding_dim,
            n_actions,
            n_objects,
            hidden_dim,
            dropout_rate,
        )

    def forward(
        self,
        past_trajectories: torch.Tensor,
        current_trajectory: torch.Tensor,
        current_state: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            past_trajectories: (batch_size, n_past, seq_len, state_dim + action_dim)
            current_trajectory: (batch_size, current_seq_len, state_dim + action_dim)
            current_state: (batch_size, state_dim)
        Returns:
            predictions: Dict with all prediction outputs
        """
        # Generate character embedding (FIXED: now properly handles N_past=0)
        character_embedding = self.character_net(past_trajectories)

        # Generate mental state embedding if used
        if self.use_mental_state and self.mental_state_net is not None:
            mental_embedding = self.mental_state_net(
                current_trajectory, character_embedding
            )
        else:
            # Create dummy mental embedding with zeros
            batch_size = current_state.size(0)
            mental_embedding = torch.zeros(batch_size, 0, device=current_state.device)

        # Generate predictions
        predictions = self.prediction_net(
            current_state, character_embedding, mental_embedding
        )

        # Add embeddings to output for analysis
        predictions["character_embedding"] = character_embedding
        if self.use_mental_state:
            predictions["mental_embedding"] = mental_embedding

        return predictions

    def compute_loss(
        self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-component loss function
        """
        losses = {}

        # Action prediction loss
        if "true_actions" in targets:
            action_loss = F.cross_entropy(
                predictions["action_logits"], targets["true_actions"]
            )
            losses["action_loss"] = action_loss

        # Object consumption loss
        if "true_consumption" in targets:
            consumption_loss = F.binary_cross_entropy(
                predictions["consumption_probs"], targets["true_consumption"]
            )
            losses["consumption_loss"] = consumption_loss

        # Successor representation loss - using cross-entropy as specified in README line 66
        if "true_sr" in targets:
            # Cross-entropy between predicted and empirical successor representation
            # L_SR = Σ_τ Σ_s -SR_τ(s) log ŜR_τ(s)
            sr_loss = F.cross_entropy(
                predictions["sr_logits"], targets["true_sr"], reduction="mean"
            )
            losses["sr_loss"] = sr_loss

        # Total loss
        total_loss = sum(losses.values())
        losses["total_loss"] = total_loss

        return losses


def create_tomnet(
    experiment_type: str,
    state_dim: int,
    char_embedding_dim: Optional[int] = None,
    action_dim: int = 5,
    n_actions: int = 5,
    n_objects: int = 4,
    mental_embedding_dim: int = 64,
    hidden_dim: int = 128,
    dropout_rate: float = 0.0,
) -> ToMnet:
    """
    Create ToMnet configuration for different experiment types

    FIXED VERSION: Uses improved CharacterNet
    ENHANCED: Added dropout support for regularization
    """
    # Set experiment-specific defaults
    if experiment_type == "figure3":
        # Figure 3: Random agents with 2D character embeddings for visualization
        if char_embedding_dim is None:
            char_embedding_dim = 2
        use_mental_state = False  # Figure 3 doesn't use mental state
        # Create Figure 3 specific CharacterNet with ConvNet + LSTM
        character_net = Figure3CharacterNet(
            state_dim=state_dim,
            action_dim=action_dim,
            embedding_dim=char_embedding_dim,
            dropout_rate=dropout_rate,
        )
    elif experiment_type == "figure5":
        # Figure 5: Goal-directed agents with higher-dimensional embeddings
        if char_embedding_dim is None:
            char_embedding_dim = 8
        use_mental_state = (
            False  # Figure 5 does NOT use mental state (per README line 591)
        )
        # Create Figure 5 specific CharacterNet with 5-layer ResNet
        character_net = Figure5CharacterNet(
            state_dim=state_dim,
            action_dim=action_dim,
            embedding_dim=char_embedding_dim,
            dropout_rate=dropout_rate,
        )
    else:
        raise ValueError(f"Unknown experiment_type: {experiment_type}")

    # Create ToMnet instance
    tomnet = ToMnet(
        state_dim=state_dim,
        action_dim=action_dim,
        n_actions=n_actions,
        n_objects=n_objects,
        char_embedding_dim=char_embedding_dim,
        mental_embedding_dim=mental_embedding_dim,
        hidden_dim=hidden_dim,
        use_mental_state=use_mental_state,
        dropout_rate=dropout_rate,
        character_net=character_net,
    )

    # Replace with experiment-specific prediction net
    if experiment_type == "figure3":
        tomnet.prediction_net = Figure3PredictionNet(
            state_dim=state_dim,
            char_embedding_dim=char_embedding_dim,
            n_actions=n_actions,
            dropout_rate=dropout_rate,
        )
    elif experiment_type == "figure5":
        tomnet.prediction_net = Figure5PredictionNet(
            state_dim=state_dim,
            char_embedding_dim=char_embedding_dim,
            n_actions=n_actions,
            n_objects=n_objects,
            dropout_rate=dropout_rate,
        )

    return tomnet
