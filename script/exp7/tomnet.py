import os
import sys

import torch
from torch import nn
import torch.nn.functional as F

# Add lib to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from utils import set_seed, calculate_sr_loss_kl_divergence

# Add current directory for utils import
from utils import spatialize_action

# Add current directory for config import
sys.path.append(os.path.dirname(__file__))
from config import Config

# Set seed using Config default value
config = Config()
set_seed(config.seed)

"""
Integrated ToMnet architecture for KeyDoor environment with second-order belief modeling (exp7)
Supports multiple architectures including novel second belief integration

Architectures:
- use_mentalnet=False: 2-stage (CharNet → PredNet) - direct prediction
- use_mentalnet=True: 3-stage (CharNet → MentalNet → PredNet) - with mental state modeling
- use_second_belief=True: Enhanced with SecondBeliefNet for modeling beliefs about others' beliefs

Channel structure (10 channels total):
- Channels 0-7: Original game state channels (walls, keys, doors, etc.)
- Channel 8: Self position (agent whose action is being predicted)
- Channel 9: Opponent position (0 for single-agent mode)

@author: exp7 implementation with second-order belief modeling capabilities
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

        # Block structure with improved initialization
        self.conv_block_1 = nn.Sequential(
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
        self.conv_block_2 = nn.Sequential(
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return self.relu(x + residual)


class LSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape -> (batch_size, seq_len, input_size)
        device = x.device
        batch_size = x.size(0)
        
        h0 = torch.zeros(1, batch_size, self.hidden_size, device=device, dtype=x.dtype)
        c0 = torch.zeros(1, batch_size, self.hidden_size, device=device, dtype=x.dtype)
        
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
        self.hidden_size_lstm = hidden_size_lstm
        self.max_n_past = max_n_past
        self.use_n_past = use_n_past

        if self.use_n_past:
            # Past episode processing architecture
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

    def forward(self, past_trajectories: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for character network

        Args:
            past_trajectories: (batch_size, n_past, seq_len, channels, height, width)

        Returns:
            Character embeddings: (batch_size, n_echar)
        """
        if self.use_n_past and past_trajectories is not None:
            batch_size, n_past_max = past_trajectories.size(0), past_trajectories.size(1)
            device = past_trajectories.device
            dtype = past_trajectories.dtype

            # Initialize character embedding from past episodes
            e_char_past = torch.zeros(batch_size, self.n_echar, device=device, dtype=dtype)
            # Track number of valid episodes per sample for averaging
            valid_episode_counts = torch.zeros(batch_size, device=device, dtype=dtype)

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
                episode_batch = episode_batch.contiguous().view(
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

                    # Reshape back to (batch, seq_len, out_channels, height, width)
                    _, out_channels, out_height, out_width = ep_x.shape
                    ep_x = ep_x.view(
                        batch_size_local,
                        seq_len_local,
                        out_channels,
                        out_height,
                        out_width,
                    )

                    # Average over spatial dimensions
                    ep_x = torch.mean(ep_x, [3, 4])  # (batch, seq_len, out_channels)

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
            valid_episode_counts = torch.clamp(valid_episode_counts, min=1.0)
            e_char_past = e_char_past / valid_episode_counts.unsqueeze(-1)

            return e_char_past
        else:
            # Return default embedding if not using past episodes
            batch_size = (
                past_trajectories.size(0)
                if past_trajectories is not None
                else self.batch
            )
            device = past_trajectories.device if past_trajectories is not None else self.default_embedding.device
            return self.default_embedding.unsqueeze(0).expand(batch_size, -1).to(device=device)


class ConvLSTM2d(nn.Module):
    """Convolutional LSTM implementation"""

    def __init__(self, input_channels, hidden_channels, kernel_size=3, padding=1):
        super(ConvLSTM2d, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = padding

        # Gates: input, forget, output, candidate
        self.conv_gates = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, x: torch.Tensor, hidden_state=None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Vectorized ConvLSTM forward pass

        Args:
            x: Input tensor (batch_size, seq_len, channels, height, width)
            hidden_state: Tuple of (h, c) or None

        Returns:
            output: (batch_size, seq_len, hidden_channels, height, width)
            (h, c): Final hidden and cell states
        """
        batch_size, seq_len, input_channels, height, width = x.size()
        device = x.device
        dtype = x.dtype

        if hidden_state is None:
            h = torch.zeros(
                batch_size, self.hidden_channels, height, width, device=device, dtype=dtype
            )
            c = torch.zeros(
                batch_size, self.hidden_channels, height, width, device=device, dtype=dtype
            )
        else:
            h, c = hidden_state

        # Initialize output storage
        outputs = torch.zeros(
            batch_size, seq_len, self.hidden_channels, height, width, device=device, dtype=dtype
        )

        # Process each timestep (minimal sequential processing for LSTM dependencies)
        for t in range(seq_len):
            x_t = x[:, t]  # (batch_size, input_channels, height, width)

            # Combine input and previous hidden state
            combined = torch.cat([x_t, h], dim=1)

            # Compute all gates in one pass
            gates = self.conv_gates(combined)

            # Split gates efficiently
            i_gate, f_gate, o_gate, g_gate = torch.split(
                gates, self.hidden_channels, dim=1
            )

            # Apply activations (vectorized)
            i_gate = torch.sigmoid(i_gate)
            f_gate = torch.sigmoid(f_gate)
            o_gate = torch.sigmoid(o_gate)
            g_gate = torch.tanh(g_gate)

            # Update cell and hidden states (vectorized)
            c = f_gate * c + i_gate * g_gate
            h = o_gate * torch.tanh(c)

            # Store output
            outputs[:, t] = h

        return outputs, (h, c)


class MentalNet(nn.Module):
    """MentalNet following the exact paper specification"""

    def __init__(
        self,
        batch: int,
        residual_blocks: int,
        n_ement: int,
        out_channels: int,
        channels_in: int,
        time_step: int,
        n_echar: int,
        action_space: int = 7,
    ):
        super(MentalNet, self).__init__()

        self.batch = batch
        self.time_step = time_step
        self.n_echar = n_echar

        # State channels (now includes all channels) + action channel (1) = total input channels
        self.state_channels = channels_in  # Use channels_in directly (now 10 channels including position info)
        self.action_space = action_space  # Number of possible actions from config
        self.input_channels = (
            self.state_channels + 1
        )  # State + spatialized action (channels_in+1)

        # Use configurable n_ement channels throughout
        self.resnet_channels = n_ement

        # Output is spatial mental state embedding (n_ement channels)
        self.output_channels = n_ement

        # Initial conv layer to get to n_ement channels
        self.input_conv = nn.Conv2d(
            self.input_channels, self.resnet_channels, kernel_size=3, padding=1
        )
        self.input_bn = nn.BatchNorm2d(self.resnet_channels)

        # Paper spec: 5-layer ResNet with n_ement channels, ReLU, BatchNorm
        self.resnet_layers = nn.ModuleList()
        for _ in range(residual_blocks):  # Exactly 5 layers as specified
            self.resnet_layers.append(
                ResidualBlock(
                    in_channels=self.resnet_channels,
                    out_channels=self.resnet_channels,
                    kernel_size=3,
                    padding=1,
                )
            )

        # Paper spec: Convolutional LSTM with n_ement channels
        self.conv_lstm = ConvLSTM2d(
            input_channels=self.resnet_channels,
            hidden_channels=self.resnet_channels,
            kernel_size=3,
            padding=1,
        )

        # Paper spec: 1-layer convnet with n_ement channels for final output
        self.output_conv = nn.Conv2d(
            self.resnet_channels, self.output_channels, kernel_size=3, padding=1
        )



    def forward(self, self_states: torch.Tensor, self_actions: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass following paper specification

        Args:
            self_states: (batch_size, seq_len, channels, height, width) - state channels only
            self_actions: (batch_size, seq_len) - action indices (optional)

        Returns:
            Mental state embedding: (batch_size, output_channels, height, width) - SPATIAL output
        """
        batch_size, seq_len, _, height, width = self_states.shape

        # Check if self states is empty (query state is initial state)
        self_states_sum = torch.sum(self_states, dim=(2, 3, 4))  # (batch_size, seq_len)
        is_empty = torch.all(self_states_sum == 0, dim=1)  # (batch_size,)

        if torch.all(is_empty):
            raise ValueError("self_states cannot be Zero tensor")
        
        # Create fallback actions if not available
        if self_actions is None:
            raise ValueError("self_actions cannot be Zero tensor")

        # Pre-process each timestep: spatialize action + concatenate with state
        processed_timesteps = []

        for t in range(seq_len):
            state_t = self_states[:, t]  # (batch_size, channels, height, width)
            action_t = self_actions[:, t]  # (batch_size,)

            # Spatialize action
            action_spatial_t = spatialize_action(
                action_t, height, width, self.action_space
            )  # (batch_size, 1, height, width)

            # Concatenate state and spatialized action
            combined_t = torch.cat(
                [state_t, action_spatial_t], dim=1
            )  # (batch_size, channels+1, height, width)

            processed_timesteps.append(combined_t)

        # Stack timesteps
        processed_trajectory = torch.stack(
            processed_timesteps, dim=1
        )  # (batch_size, seq_len, channels+1, height, width)

        # Pass through initial conv to get to 32 channels
        # Reshape for processing
        trajectory_flat = processed_trajectory.view(
            batch_size * seq_len, self.input_channels, height, width
        )

        # Initial conv + batch norm + ReLU
        x = self.input_conv(trajectory_flat)
        x = self.input_bn(x)
        x = F.relu(x)

        # Pass through 5-layer ResNet with 32 channels, ReLU, BatchNorm
        for resnet_layer in self.resnet_layers:
            x = resnet_layer(x)

        # Reshape back to sequence format for ConvLSTM
        x = x.view(batch_size, seq_len, self.resnet_channels, height, width)

        # Feed into convolutional LSTM with 32 channels
        _, (final_h, _) = self.conv_lstm(x)

        # Use final hidden state from LSTM
        final_features = final_h  # (batch_size, 32, height, width)

        # LSTM output through 1-layer convnet with n_ement channels
        mental_state = self.output_conv(
            final_features
        )  # (batch_size, n_ement, height, width)

        # Handle empty trajectories - set their output to zero
        for i in range(batch_size):
            if is_empty[i]:
                mental_state[i] = 0

        return mental_state


class SecondBeliefNet(nn.Module):
    """
    SecondBeliefNet for modeling second-order beliefs (e_opp2)
    
    Takes mental state embedding and opponent trajectory to produce
    an embedding representing what the agent believes about others' beliefs.
    
    Uses the SAME LOGIC as MentalNet: processes state + spatialized actions
    """
    
    def __init__(
        self,
        n_ement: int,
        n_eopp2: int,
        channels_in: int,
        time_step: int,
        hidden_size: int = 128,
        residual_blocks: int = 5,  # Match MentalNet's 5 residual blocks
        action_space: int = 7,
    ):
        super(SecondBeliefNet, self).__init__()
        
        self.n_ement = n_ement
        self.n_eopp2 = n_eopp2
        self.channels_in = channels_in
        self.time_step = time_step
        self.hidden_size = hidden_size
        self.action_space = action_space
        
        # SAME AS MENTALNET: State channels (channels_in) + action channel (1) = total input channels
        self.state_channels = channels_in  # channels_in value
        self.input_channels = self.state_channels + 1  # State + spatialized action (channels_in+1)
        
        # SAME AS MENTALNET: Initial conv to process combined state+action
        self.input_conv = nn.Conv2d(
            self.input_channels, hidden_size, kernel_size=3, padding=1
        )
        self.input_bn = nn.BatchNorm2d(hidden_size)
        
        # Mental state processor: handles spatial mental state for fusion
        self.mental_conv = nn.Conv2d(
            n_ement, hidden_size, kernel_size=3, padding=1
        )
        self.mental_bn = nn.BatchNorm2d(hidden_size)
        
        # Fusion layer: combines trajectory and mental state features
        self.fusion_conv = nn.Conv2d(
            hidden_size * 2, hidden_size, kernel_size=3, padding=1
        )
        self.fusion_bn = nn.BatchNorm2d(hidden_size)
        
        # SAME AS MENTALNET: Residual blocks for feature refinement
        self.resnet_layers = nn.ModuleList()
        for _ in range(residual_blocks):
            self.resnet_layers.append(
                ResidualBlock(
                    in_channels=hidden_size,
                    out_channels=hidden_size,
                    kernel_size=3,
                    padding=1,
                )
            )
        
        # SAME AS MENTALNET: ConvLSTM for temporal processing (maintains spatial structure)
        self.conv_lstm = ConvLSTM2d(hidden_size, hidden_size)
        
        # Output conv to project to n_eopp2 channels (spatial output like MentalNet)
        self.output_conv = nn.Conv2d(
            hidden_size, n_eopp2, kernel_size=3, padding=1
        )
        
        # Global pooling to get vector embedding from spatial output
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
    
    def forward(self, mental_state: torch.Tensor, oppo_states: torch.Tensor, oppo_actions: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for SecondBeliefNet - USING SAME LOGIC AS MENTALNET
        
        Args:
            mental_state: (batch_size, n_ement, height, width) - spatial mental state
            oppo_states: (batch_size, seq_len, channels_in, height, width) - opponent's state channels
            oppo_actions: (batch_size, seq_len) - opponent's action indices (optional)
            
        Returns:
            e_opp2: (batch_size, n_eopp2) - second belief embedding
        """
        if oppo_states is None:
            raise ValueError("oppo_states cannot be None")
            
        batch_size, seq_len, _, height, width = oppo_states.shape
        
        # Create fallback actions if not available
        if oppo_actions is None:
            raise ValueError("oppo_actions cannot be None")
        
        # Check if opponent states is empty - SAME AS MENTALNET
        oppo_states_sum = torch.sum(oppo_states, dim=(2, 3, 4))  # (batch_size, seq_len)
        is_empty = torch.all(oppo_states_sum == 0, dim=1)  # (batch_size,)
        
        # Process each timestep with state + spatialized action - SAME AS MENTALNET
        processed_timesteps = []
        for t in range(seq_len):
            state_t = oppo_states[:, t]  # (batch_size, channels, height, width)
            action_t = oppo_actions[:, t]  # (batch_size,)
            
            # Spatialize action - SAME AS MENTALNET
            action_spatial_t = spatialize_action(
                action_t, height, width, self.action_space
            )  # (batch_size, 1, height, width)
            
            # Concatenate state and spatialized action - SAME AS MENTALNET
            combined_t = torch.cat(
                [state_t, action_spatial_t], dim=1
            )  # (batch_size, channels+1, height, width)
            
            processed_timesteps.append(combined_t)
        
        # Stack timesteps
        processed_trajectory = torch.stack(
            processed_timesteps, dim=1
        )  # (batch_size, seq_len, channels+1, height, width)
        
        # Flatten for processing - SAME AS MENTALNET
        trajectory_flat = processed_trajectory.view(
            batch_size * seq_len, self.input_channels, height, width
        )
        
        # Initial conv + batch norm + ReLU - SAME AS MENTALNET
        x = self.input_conv(trajectory_flat)
        x = self.input_bn(x)
        x = F.relu(x)
        
        # Pass through ResNet blocks - SAME AS MENTALNET
        for resnet_layer in self.resnet_layers:
            x = resnet_layer(x)
        
        # Reshape back to sequence format for ConvLSTM - SAME AS MENTALNET
        x = x.view(batch_size, seq_len, self.hidden_size, height, width)
        
        # Feed into ConvLSTM - SAME AS MENTALNET
        lstm_output, (final_h, final_c) = self.conv_lstm(x)
        
        # Use final hidden state from ConvLSTM - SAME AS MENTALNET
        final_features = final_h  # (batch_size, hidden_size, height, width)
        
        # NOW FUSE WITH MENTAL STATE (unique to SecondBeliefNet)
        # Process mental state
        mental_features = self.mental_conv(mental_state)  # (batch, hidden_size, H, W)
        mental_features = self.mental_bn(mental_features)
        mental_features = F.relu(mental_features)
        
        # Fuse trajectory features with mental state
        combined = torch.cat([final_features, mental_features], dim=1)  # (batch, hidden_size*2, H, W)
        fused = self.fusion_conv(combined)  # (batch, hidden_size, H, W)
        fused = self.fusion_bn(fused)
        fused = F.relu(fused)
        
        # Output through final conv - produces spatial output like MentalNet
        spatial_output = self.output_conv(fused)  # (batch_size, n_eopp2, height, width)
        
        # Handle empty trajectories - SAME AS MENTALNET
        for i in range(batch_size):
            if is_empty[i]:
                spatial_output[i] = 0
        
        # Global pooling to get vector embedding
        pooled = self.global_pool(spatial_output)  # (batch_size, n_eopp2, 1, 1)
        e_opp2 = pooled.squeeze(-1).squeeze(-1)  # (batch_size, n_eopp2)
        
        # Apply dropout
        e_opp2 = self.dropout(e_opp2)
        
        return e_opp2


class CrossAttentionModule(nn.Module):
    """
    Cross-attention module for combining character, mental, and second belief embeddings
    """
    
    def __init__(
        self,
        n_echar: int,
        n_ement: int,
        n_eopp2: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super(CrossAttentionModule, self).__init__()
        
        self.n_echar = n_echar
        self.n_ement = n_ement
        self.n_eopp2 = n_eopp2
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Project all embeddings to same dimension for attention
        self.char_proj = nn.Linear(n_echar, hidden_dim)
        self.mental_proj = nn.Linear(n_ement, hidden_dim)  # For spatial mental state, will be adapted
        self.opp2_proj = nn.Linear(n_eopp2, hidden_dim)
        
        # Multi-head attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, e_char: torch.Tensor, e_mental: torch.Tensor, e_opp2: torch.Tensor, current_state_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for cross-attention
        
        Args:
            e_char: (batch_size, n_echar) - character embedding
            e_mental: (batch_size, n_ement, height, width) or (batch_size, n_ement) - mental state
            e_opp2: (batch_size, n_eopp2) - second belief embedding
            current_state_features: (batch_size, feature_dim) - query features from current state
            
        Returns:
            attended_features: (batch_size, hidden_dim) - attention-weighted features
            attention_weights: attention weights tensor
        """
        batch_size = e_char.size(0)
        
        # Handle spatial mental state by global average pooling
        if e_mental.dim() == 4:  # Spatial mental state
            e_mental_pooled = torch.mean(e_mental, [2, 3])  # (batch, n_ement)
        else:
            e_mental_pooled = e_mental
        
        # Project embeddings to attention space
        char_proj = self.char_proj(e_char)  # (batch, hidden_dim)
        mental_proj = self.mental_proj(e_mental_pooled)  # (batch, hidden_dim)
        opp2_proj = self.opp2_proj(e_opp2)  # (batch, hidden_dim)
        
        # Stack embeddings for attention (values and keys)
        embeddings = torch.stack([char_proj, mental_proj, opp2_proj], dim=1)  # (batch, 3, hidden_dim)
        
        # Use current state features as query
        if current_state_features.dim() == 1:
            current_state_features = current_state_features.unsqueeze(0)
        query = current_state_features.unsqueeze(1)  # (batch, 1, feature_dim)
        
        # Project query to attention space if needed
        if current_state_features.size(-1) != self.hidden_dim:
            query_proj = nn.Linear(current_state_features.size(-1), self.hidden_dim, device=current_state_features.device)
            query = query_proj(query)  # (batch, 1, hidden_dim)
        
        # Apply multi-head attention
        attended, attention_weights = self.multihead_attn(
            query=query,  # (batch, 1, hidden_dim)
            key=embeddings,  # (batch, 3, hidden_dim)
            value=embeddings,  # (batch, 3, hidden_dim)
        )
        
        # Process output
        attended = attended.squeeze(1)  # (batch, hidden_dim)
        attended = self.dropout(attended)
        attended = self.layer_norm(self.output_proj(attended))
        
        return attended, attention_weights


class PredNet(nn.Module):
    def __init__(
        self,
        batch: int,
        n_ement: int,
        n_echar: int,
        channels_in: int,
        residual_blocks: int,
        action_space: int = 7,
        out_channels: int = 64,
        goal_space: int = 4,
        env_width: int = 9,
        env_height: int = 9,
        use_mentalnet: bool = False,
        use_second_belief: bool = False,
        n_eopp2: int = 64,
        attention_hidden: int = 256,
        attention_heads: int = 8,
    ):
        super(PredNet, self).__init__()

        self.batch = batch
        self.n_ement = n_ement
        self.n_echar = n_echar
        self.channels_in = channels_in
        self.action_space = action_space
        self.goal_space = goal_space
        self.env_width = env_width
        self.env_height = env_height
        self.out_channels = out_channels
        self.n = residual_blocks
        self.use_mentalnet = use_mentalnet
        self.use_second_belief = use_second_belief
        self.n_eopp2 = n_eopp2

        # Determine input channels based on architecture
        if use_second_belief:
            # With second belief: use cross-attention
            self.cross_attention = CrossAttentionModule(
                n_echar=n_echar,
                n_ement=n_ement,
                n_eopp2=n_eopp2,
                hidden_dim=attention_hidden,
                num_heads=attention_heads,
            )
            # Input channels: current_state + attention output
            input_channels = channels_in + attention_hidden
        elif use_mentalnet:
            # Original 3-stage: current_state + mental_state + character_embedding
            # MentalNet outputs n_ement channels (spatial)
            input_channels = channels_in + n_ement + n_echar
        else:
            # 2-stage architecture: current_state + character_embedding - direct prediction
            input_channels = channels_in + n_echar

        # Shared torso - processes input data
        self.conv_1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.res_blocks = nn.ModuleList()
        for _ in range(self.n):
            self.res_blocks.append(
                ResidualBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                )
            )

        self.conv_2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # Shared feature extraction
        self.fc1 = nn.Linear(out_channels, out_channels)
        self.fc2 = nn.Linear(out_channels, out_channels)

        # Action prediction head
        self.fc3_action = nn.Linear(out_channels, action_space)

        # Goal prediction head
        self.fc3_goal = nn.Linear(out_channels, goal_space)

        # Agent prediction head (2 outputs: 0=achiever, 1=blocker)
        self.fc3_agent = nn.Linear(out_channels, 2)

        # Type prediction head (2 outputs: level, depth)
        self.fc3_type = nn.Linear(out_channels, 2)

        # Consumption prediction head (8 outputs: 4 keys + 4 doors)
        self.fc3_consumption = nn.Linear(out_channels, 8)

        # SR prediction heads for different discount factors
        self.conv_sr = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
        )
        self.conv_sr_out = nn.Conv2d(
            in_channels=out_channels,
            out_channels=3,  # 3 discount factors
            kernel_size=1,
        )

    def forward(self, mental_state: torch.Tensor, character_embedding: torch.Tensor, current_state: torch.Tensor, e_opp2: torch.Tensor = None) -> tuple[torch.Tensor, ...]:
        """
        Forward pass for prediction network with optional second belief

        Args:
            mental_state: (batch_size, n_ement, height, width) - spatial mental state from MentalNet
            character_embedding: (batch_size, n_echar)
            current_state: (batch_size, channels_in, height, width)
            e_opp2: (batch_size, n_eopp2) - second belief embedding (optional)

        Returns:
            tuple of prediction tensors: (action_logits, goal_logits, agent_logits, type_logits, consumption_logits, sr_pred)
        """
        batch_size, _, height, width = current_state.shape

        if self.use_second_belief and e_opp2 is not None:
            # Use cross-attention to combine all embeddings
            # Extract features from current state for query
            current_features = torch.mean(current_state, [2, 3])  # (batch, channels_in)
            
            # Apply cross-attention
            attended_features, attention_weights = self.cross_attention(
                e_char=character_embedding,
                e_mental=mental_state,
                e_opp2=e_opp2,
                current_state_features=current_features
            )
            
            # Broadcast attended features to spatial dimensions
            attended_spatial = (
                attended_features.unsqueeze(2)
                .unsqueeze(3)
                .expand(batch_size, attended_features.size(-1), height, width)
            )
            
            # Concatenate current state with attended features
            x = torch.cat([current_state, attended_spatial], dim=1)
        else:
            # Original architecture logic
            if self.use_mentalnet:
                # Handle spatial mental state (already spatial from MentalNet)
                if mental_state.dim() == 4:
                    # Mental state is already spatial (batch_size, n_ement, height, width)
                    mental_state_spatial = mental_state
                else:
                    # Fallback: broadcast mental_state if it's 1D
                    mental_state_spatial = (
                        mental_state.unsqueeze(2)
                        .unsqueeze(3)
                        .expand(batch_size, self.n_ement, height, width)
                    )

                # Spatially broadcast character_embedding
                character_embedding_spatial = (
                    character_embedding.unsqueeze(2)
                    .unsqueeze(3)
                    .expand(batch_size, self.n_echar, height, width)
                )

                # Concatenate all inputs
                x = torch.cat(
                    [current_state, mental_state_spatial, character_embedding_spatial], dim=1
                )
            else:
                # 2-stage architecture: just current_state + character_embedding
                character_embedding_spatial = (
                    character_embedding.unsqueeze(2)
                    .unsqueeze(3)
                    .expand(batch_size, self.n_echar, height, width)
                )
                x = torch.cat([current_state, character_embedding_spatial], dim=1)

        return self._forward_shared(x)

    def forward_direct(self, mixed_data: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        Direct forward pass for 2-stage architecture - bypasses mental state modeling

        Args:
            mixed_data: (batch_size, channels_in + n_echar, height, width)

        Returns:
            tuple of prediction tensors: (action_logits, goal_logits, agent_logits, type_logits, consumption_logits, sr_pred)
        """
        return self._forward_shared(mixed_data)

    def _forward_shared(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Shared forward pass implementation"""
        # Shared torso
        x = self.conv_1(x)

        for i in range(self.n):
            x = self.res_blocks[i](x)

        x = self.conv_2(x)
        x = F.relu(x)

        # Store spatial features for SR prediction
        spatial_features = x

        # Global pooling for action and consumption predictions
        x_pooled = torch.mean(x, [2, 3])  # (batch_size, out_channels)

        # Shared feature extraction
        x_pooled = self.fc1(x_pooled)
        x_pooled = F.relu(x_pooled)

        x_pooled = self.fc2(x_pooled)
        x_pooled = F.relu(x_pooled)

        # Prediction heads
        action_logits = self.fc3_action(x_pooled)
        goal_logits = self.fc3_goal(x_pooled)
        agent_logits = self.fc3_agent(x_pooled)
        type_logits = self.fc3_type(x_pooled)
        consumption_logits = self.fc3_consumption(x_pooled)

        # SR prediction (using spatial features)
        sr_features = self.conv_sr(spatial_features)
        sr_features = F.relu(sr_features)
        sr_pred = self.conv_sr_out(sr_features)

        # Apply softmax to each SR channel independently
        batch_size, channels, height, width = sr_pred.shape
        sr_pred = sr_pred.view(batch_size, channels, -1)
        sr_pred = F.softmax(sr_pred, dim=2)
        sr_pred = sr_pred.view(batch_size, channels, height, width)

        return (
            action_logits,
            goal_logits,
            agent_logits,
            type_logits,
            consumption_logits,
            sr_pred,
        )


class ToMnet(nn.Module):
    def __init__(
        self,
        use_mentalnet: bool = False,
        use_second_belief: bool = False,
        batch: int = 32,
        residual_blocks: int = 3,
        n_echar: int = 64,
        n_ement: int = 64,
        n_eopp2: int = 64,
        out_channels: int = 32,
        channels_in: int = 10,  # 8 original channels + 1 self position + 1 opponent position (for CharNet)
        time_step: int = 500,
        action_space: int = 7,
        goal_space: int = 4,
        max_n_past: int = 10,
        use_n_past: bool = True,
        env_width: int = 9,
        env_height: int = 9,
        hidden_size_lstm: int = 64,
        second_belief_hidden: int = 128,
        attention_hidden: int = 256,
        attention_heads: int = 8,
    ):
        super(ToMnet, self).__init__()

        self.use_mentalnet = use_mentalnet
        self.use_second_belief = use_second_belief
        self.batch = batch
        self.n_echar = n_echar
        self.n_ement = n_ement
        self.n_eopp2 = n_eopp2
        self.time_step = time_step
        self.action_space = action_space
        self.goal_space = goal_space
        self.max_n_past = max_n_past
        self.use_n_past = use_n_past
        self.channels_in = channels_in
        self.env_width = env_width
        self.env_height = env_height

        # Character network - processes past episodes (same for both architectures)
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

        # Mental state network - only used in original 3-stage architecture
        if use_mentalnet:
            self.mental_net = MentalNet(
                batch=batch,
                residual_blocks=residual_blocks,
                n_ement=n_ement,
                out_channels=out_channels,
                channels_in=channels_in,  # Use full channels_in (10)
                time_step=time_step,
                n_echar=n_echar,
                action_space=action_space,
            )

        # Second belief network - only used when second belief is enabled
        if use_second_belief:
            self.second_belief_net = SecondBeliefNet(
                n_ement=n_ement,
                n_eopp2=n_eopp2,
                channels_in=channels_in,  # Keep channels_in (10) for opponent states which include position info
                time_step=time_step,
                hidden_size=second_belief_hidden,
                residual_blocks=residual_blocks,
                action_space=action_space,
            )

        # Prediction network - processes inputs based on architecture
        self.pred_net = PredNet(
            batch=batch,
            n_ement=n_ement,
            n_echar=n_echar,
            channels_in=channels_in,
            residual_blocks=residual_blocks,
            action_space=action_space,
            out_channels=out_channels,
            goal_space=goal_space,
            env_width=env_width,
            env_height=env_height,
            use_mentalnet=use_mentalnet,
            use_second_belief=use_second_belief,
            n_eopp2=n_eopp2,
            attention_hidden=attention_hidden,
            attention_heads=attention_heads,
        )


    def forward(self, past_trajectories: torch.Tensor, self_states: torch.Tensor, self_actions: torch.Tensor,
                current_state: torch.Tensor, oppo_states: torch.Tensor = None, oppo_actions: torch.Tensor = None, **kwargs) -> dict:
        """
        Forward pass for ToMnet (supports all architectures including second belief)

        Args:
            past_trajectories: (batch_size, n_past, seq_len, channels, height, width) - for CharNet (past self states)
            self_states: (batch_size, seq_len, channels, height, width) - recent self states for MentalNet (if used)
            self_actions: (batch_size, seq_len) - recent self actions for MentalNet (if used)
            current_state: (batch_size, channels, height, width) - for PredNet
            oppo_states: (batch_size, seq_len, channels, height, width) - opponent states for SecondBeliefNet (optional)
            oppo_actions: (batch_size, seq_len) - opponent actions for SecondBeliefNet (optional)
            **kwargs: Additional keyword arguments for flexible parameter passing

        Returns:
            Dict with keys: action_logits, goal_logits, agent_logits, type_logits, consumption_logits, sr_pred, 
                           character_embedding, mental_state, second_belief (optional)
        """
        # Validate inputs for second belief functionality
        if self.use_second_belief and oppo_states is None:
            pass
        # 1. Character network - processes past episodes (same for both architectures)
        if self.use_n_past and past_trajectories is not None:
            character_embedding = self.char_net(past_trajectories)
        else:
            batch_size = current_state.size(0)
            device = current_state.device
            dtype = current_state.dtype
            character_embedding = torch.zeros(
                batch_size, self.n_echar, device=device, dtype=dtype
            )

        # Use full state for PredNet (no channel slicing)
        current_state_for_pred = current_state
        batch_size, _, height, width = current_state_for_pred.shape

        # Initialize second_belief as None
        second_belief = None
        device = current_state.device
        dtype = current_state.dtype

        if self.use_mentalnet:
            # Process recent trajectory through MentalNet
            # Use all channels (no slicing needed anymore)
            state_channels_only = self_states  # (batch_size, seq_len, channels_in, height, width)
            mental_state = self.mental_net(state_channels_only, self_actions)

            # Generate second belief if enabled
            if self.use_second_belief and oppo_states is not None:
                # Use all channels from opponent states as well
                oppo_state_channels_only = oppo_states  # (batch_size, seq_len, channels_in, height, width)
                second_belief = self.second_belief_net(mental_state, oppo_state_channels_only, oppo_actions)
            else:
                second_belief = torch.zeros(batch_size, self.n_eopp2, device=device, dtype=dtype)

            # PredNet with all embeddings
            (
                action_logits,
                goal_logits,
                agent_logits,
                type_logits,
                consumption_logits,
                sr_pred,
            ) = self.pred_net(mental_state, character_embedding, current_state_for_pred, second_belief)
        else:
            # 2-stage architecture: CharNet → PredNet - direct prediction without mental modeling
            # Create dummy mental_state since MentalNet is not used in 2-stage architecture
            device = current_state.device
            dtype = current_state.dtype
            mental_state = torch.zeros(
                batch_size, self.n_ement, device=device, dtype=dtype
            )

            if self.use_second_belief:
                # 2-stage mode with second belief: use cross-attention
                dummy_mental_state = torch.zeros(
                    batch_size, self.n_ement, height, width, device=device, dtype=dtype
                )
                
                if oppo_states is not None:
                    second_belief = self.second_belief_net(dummy_mental_state, oppo_states, oppo_actions)
                else:
                    second_belief = torch.zeros(batch_size, self.n_eopp2, device=device, dtype=dtype)
                
                # Use PredNet with all embeddings (it will use cross-attention internally)
                (
                    action_logits,
                    goal_logits,
                    agent_logits,
                    type_logits,
                    consumption_logits,
                    sr_pred,
                ) = self.pred_net(dummy_mental_state, character_embedding, current_state_for_pred, second_belief)
            else:
                # Original 2-stage architecture without second belief
                # Reshape character embedding to spatial format
                e_char_spatial = (
                    character_embedding.unsqueeze(2)
                    .unsqueeze(3)
                    .expand(batch_size, self.n_echar, height, width)
                )

                # Concatenate current_state with character embedding
                mixed_data = torch.cat([current_state_for_pred, e_char_spatial], dim=1)

                # PredNet processes mixed data directly
                (
                    action_logits,
                    goal_logits,
                    agent_logits,
                    type_logits,
                    consumption_logits,
                    sr_pred,
                ) = self.pred_net.forward_direct(mixed_data)
                
                second_belief = torch.zeros(batch_size, self.n_eopp2, device=device, dtype=dtype)

        return {
            "action_logits": action_logits,
            "goal_logits": goal_logits,
            "agent_logits": agent_logits,
            "type_logits": type_logits,
            "consumption_logits": consumption_logits,
            "sr_pred": sr_pred,
            "character_embedding": character_embedding,
            "mental_state": mental_state,
            "second_belief": second_belief,
        }

    def predict_action(self, past_trajectories: torch.Tensor, self_states: torch.Tensor, self_actions: torch.Tensor,
                      current_state: torch.Tensor, oppo_states: torch.Tensor = None, oppo_actions: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """
        Predict next action probabilities using softmax on action logits.
        
        Args:
            past_trajectories: Past episode trajectories for character embedding
            self_states: Recent self states for mental state modeling
            self_actions: Recent self actions for mental state modeling
            current_state: Current state for prediction
            oppo_states: Opponent states for second belief (optional)
            oppo_actions: Opponent actions for second belief (optional)
            **kwargs: Additional arguments
            
        Returns:
            torch.Tensor: Action probabilities (batch_size, action_space)
        """
        with torch.no_grad():
            outputs = self.forward(
                past_trajectories, self_states, self_actions, current_state, oppo_states, oppo_actions, **kwargs
            )
            return F.softmax(outputs["action_logits"], dim=1)

    def predict_goal(self, past_trajectories, self_states, self_actions, current_state, oppo_states=None, oppo_actions=None, **kwargs):
        """Predict goal"""
        with torch.no_grad():
            outputs = self.forward(
                past_trajectories, self_states, self_actions, current_state, oppo_states, oppo_actions, **kwargs
            )
            return F.softmax(outputs["goal_logits"], dim=1)

    def get_character_embedding(self, past_trajectories):
        """Get character embedding from past trajectories"""
        with torch.no_grad():
            if self.use_n_past and past_trajectories is not None:
                return self.char_net(past_trajectories)
            else:
                batch_size = 1
                return torch.zeros(batch_size, self.n_echar)

    def get_mental_state(self, self_states, self_actions):
        """Get mental state from recent trajectory states and actions"""
        with torch.no_grad():
            if self.use_mentalnet:
                # Use new MentalNet interface - pass states and actions separately
                mental_state = self.mental_net(self_states, self_actions)
                return mental_state
            else:
                # Return dummy mental state for fixed architecture
                batch_size = (
                    1 if self_states is None else self_states.size(0)
                )
                device = self_states.device if self_states is not None else torch.device('cpu')
                return torch.zeros(batch_size, self.n_ement, device=device)
    
    def validate_configuration(self) -> bool:
        """
        Validate the model configuration for common issues.
        
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        issues = []
        
        # Check second belief configuration
        if self.use_second_belief and not hasattr(self, 'second_belief_net'):
            issues.append("SecondBeliefNet enabled but not initialized")
        
        # Check mental net configuration  
        if self.use_mentalnet and not hasattr(self, 'mental_net'):
            issues.append("MentalNet enabled but not initialized")
        
        # Check embedding dimensions
        if self.n_echar <= 0:
            issues.append(f"Invalid character embedding dimension: {self.n_echar}")
        
        if self.n_ement <= 0:
            issues.append(f"Invalid mental embedding dimension: {self.n_ement}")
        
        if self.use_second_belief and self.n_eopp2 <= 0:
            issues.append(f"Invalid second belief embedding dimension: {self.n_eopp2}")
        
        # Report issues
        if issues:
            print("Model configuration issues:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        
        return True
    
    def get_model_info(self) -> dict:
        """
        Get detailed information about the model configuration.
        
        Returns:
            dict: Model configuration and capability information
        """
        return {
            "architecture": "3-stage" if self.use_mentalnet else "2-stage",
            "second_belief_enabled": self.use_second_belief,
            "character_embedding_dim": self.n_echar,
            "mental_embedding_dim": self.n_ement,
            "second_belief_dim": self.n_eopp2 if self.use_second_belief else None,
            "action_space": self.action_space,
            "goal_space": self.goal_space,
            "use_past_episodes": self.use_n_past,
            "max_past_episodes": self.max_n_past,
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }


class ToMnetLoss(nn.Module):
    def __init__(
        self,
        action_weight=1.0,
        goal_weight=1.0,
        agent_weight=1.0,
        type_weight=1.0,
        consumption_weight=1.0,
        sr_weight=1.0,
    ):
        super(ToMnetLoss, self).__init__()
        self.action_weight = action_weight
        self.goal_weight = goal_weight
        self.agent_weight = agent_weight
        self.type_weight = type_weight
        self.consumption_weight = consumption_weight
        self.sr_weight = sr_weight
        self.action_loss = nn.CrossEntropyLoss(ignore_index=-1)  # Ignore padded actions
        self.goal_loss = nn.CrossEntropyLoss()
        self.agent_loss = nn.CrossEntropyLoss()
        self.type_loss = nn.CrossEntropyLoss()
        self.consumption_loss = nn.BCEWithLogitsLoss()

    def forward(
        self,
        action_logits,
        goal_logits,
        agent_logits,
        type_logits,
        consumption_logits,
        sr_pred,
        action_targets,
        goal_targets,
        agent_targets,
        type_targets,
        consumption_targets,
        sr_targets,
    ):
        """Compute combined loss"""
        action_loss = self.action_loss(action_logits, action_targets)
        goal_loss = self.goal_loss(goal_logits, goal_targets)
        agent_loss = self.agent_loss(agent_logits, agent_targets)
        type_loss = self.type_loss(type_logits, type_targets)
        consumption_loss = self.consumption_loss(
            consumption_logits, consumption_targets
        )

        sr_loss = calculate_sr_loss_kl_divergence(sr_pred, sr_targets)

        total_loss = (
            self.action_weight * action_loss
            + self.goal_weight * goal_loss
            + self.agent_weight * agent_loss
            + self.type_weight * type_loss
            + self.consumption_weight * consumption_loss
            + self.sr_weight * sr_loss
        )

        return {
            "loss": total_loss,
            "action_loss": action_loss,
            "goal_loss": goal_loss,
            "agent_loss": agent_loss,
            "type_loss": type_loss,
            "consumption_loss": consumption_loss,
            "sr_loss": sr_loss,
        }


# Utility functions
def create_model(config, save_dir=None):
    """
    Create ToMnet model from configuration.

    Args:
        config: Configuration object or dictionary containing model parameters
        save_dir: Directory where model checkpoint might be saved (optional)

    Returns:
        ToMnet: Model instance with loaded weights if checkpoint exists
    """
    # Handle both Config object and dictionary
    if hasattr(config, "get_model_kwargs"):
        # Config object
        model_kwargs = config.get_model_kwargs()
        model = ToMnet(**model_kwargs)
    else:
        # Dictionary - use provided parameters directly
        model = ToMnet(
            use_mentalnet=config["use_mentalnet"],
            use_second_belief=config["use_second_belief"],
            batch=config.get("batch", 32),
            residual_blocks=config.get("residual_blocks", 3),
            n_echar=config.get("n_echar", 64),
            n_ement=config.get("n_ement", 64),
            n_eopp2=config.get("n_eopp2", 64),
            out_channels=config.get("out_channels", 32),
            channels_in=config.get("channels_in", 10),
            time_step=config.get("time_step", 500),
            action_space=config.get("action_space", 7),
            goal_space=config.get("goal_space", 4),
            max_n_past=config.get("max_n_past", 10),
            use_n_past=config.get("use_n_past", True),
            env_width=config.get("env_width", 9),
            env_height=config.get("env_height", 9),
            hidden_size_lstm=config.get("hidden_size_lstm", 64),
            second_belief_hidden=config.get("second_belief_hidden", 128),
            attention_hidden=config.get("attention_hidden", 256),
            attention_heads=config.get("attention_heads", 8),
        )
    
    print(f"Created model with configuration: {model.get_model_info()}")

    # Load checkpoint if available
    if save_dir is not None:
        checkpoint_path = os.path.join(save_dir, "best_model.pth")
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(checkpoint)
            print("Successfully loaded model parameters from checkpoint")
        else:
            print(f"No checkpoint found at: {checkpoint_path}")
    else:
        print("No save_dir provided, skipping checkpoint loading")

    return model


def count_parameters(model):
    """Count the number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Example usage
if __name__ == "__main__":
    # Test different architectures
    test_configs = [
        {"name": "2-stage without SecondBelief", "use_mentalnet": False, "use_second_belief": False},
        {"name": "3-stage without SecondBelief", "use_mentalnet": True, "use_second_belief": False},
        {"name": "2-stage with SecondBelief", "use_mentalnet": False, "use_second_belief": True},
        {"name": "3-stage with SecondBelief", "use_mentalnet": True, "use_second_belief": True},
    ]
    
    for test_config in test_configs:
        print(f"\n=== Testing {test_config['name']} ===")

        config = {
            "use_mentalnet": test_config["use_mentalnet"],
            "use_second_belief": test_config["use_second_belief"],
            "batch_size": 32,
            "residual_blocks": 3,
            "n_echar": 16,
            "n_ement": 16,
            "n_eopp2": 16,
            "out_channels": 32,
            "channels_in": 10,
            "time_step": 20,
            "action_space": 7,
            "goal_space": 4,
            "max_n_past": 1,
            "use_n_past": True,
            "second_belief_hidden": 64,
            "attention_hidden": 128,
            "attention_heads": 4,
        }

        model = create_model(config)
        print(f"Model created with {count_parameters(model):,} parameters")

        # Test forward pass
        batch_size = 4
        seq_len = 20
        past_trajectories = torch.randn(batch_size, 1, seq_len, 10, 9, 9)
        self_states = torch.randn(batch_size, seq_len, 8, 9, 9)
        self_actions = torch.randint(0, 7, (batch_size, seq_len))
        current_state = torch.randn(batch_size, 8, 9, 9)
        oppo_states = torch.randn(batch_size, seq_len, 10, 9, 9)
        oppo_actions = torch.randint(0, 7, (batch_size, seq_len))

        # Test with opponent trajectory
        outputs = model(past_trajectories, self_states, self_actions, current_state, oppo_states, oppo_actions)

        print(f"Action logits: {outputs['action_logits'].shape}")
        print(f"Goal logits: {outputs['goal_logits'].shape}")
        print(f"Agent logits: {outputs['agent_logits'].shape}")
        print(f"Type logits: {outputs['type_logits'].shape}")
        print(f"Consumption logits: {outputs['consumption_logits'].shape}")
        print(f"SR pred: {outputs['sr_pred'].shape}")
        print(f"Character embedding: {outputs['character_embedding'].shape}")
        print(f"Mental state: {outputs['mental_state'].shape}")
        print(f"Second belief: {outputs['second_belief'].shape}")
        
        # Test without opponent trajectory (single-agent mode)
        if test_config["use_second_belief"]:
            outputs_single = model(past_trajectories, self_states, self_actions, current_state, None, None)
            print(f"Single-agent mode - Second belief: {outputs_single['second_belief'].shape}")
