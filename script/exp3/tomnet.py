import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

"""
Integrated ToMnet architecture for KeyDoor environment (experiment 3)
Supports both original 3-stage and fixed 2-stage architectures via config flag

Architectures:
- use_mentalnet=False: experiment5-style (CharNet → PredNet) - bypasses bottleneck
- use_mentalnet=True: original 3-stage (CharNet → MentalNet → PredNet) - with bottleneck

Channel structure (9 channels total):
- Channels 0-7: Original game state channels (walls, keys, doors, agent position, etc.)
- Channel 8: Agent heading direction (0=north, 1=east, 2=south, 3=west)

@author: Based on ToMnetF experiment5, adapted for KeyDoor environment
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

    def forward(self, past_trajectories):
        """
        Forward pass for character network

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
            padding=padding
        )
        
    def forward(self, x, hidden_state=None):
        """
        Vectorized ConvLSTM forward pass
        
        Args:
            x: Input tensor (batch_size, seq_len, channels, height, width)
            hidden_state: Tuple of (h, c) or None
            
        Returns:
            output: (batch_size, seq_len, hidden_channels, height, width)
            (h, c): Final hidden and cell states
        """
        batch_size, seq_len, _, height, width = x.size()
        
        if hidden_state is None:
            h = torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device)
            c = torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device)
        else:
            h, c = hidden_state
            
        # Initialize output storage
        outputs = torch.zeros(batch_size, seq_len, self.hidden_channels, height, width, device=x.device)
        
        # Process each timestep (minimal sequential processing for LSTM dependencies)
        for t in range(seq_len):
            x_t = x[:, t]  # (batch_size, input_channels, height, width)
            
            # Combine input and previous hidden state
            combined = torch.cat([x_t, h], dim=1)
            
            # Compute all gates in one pass
            gates = self.conv_gates(combined)
            
            # Split gates efficiently
            i_gate, f_gate, o_gate, g_gate = torch.split(gates, self.hidden_channels, dim=1)
            
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
    ):
        super(MentalNet, self).__init__()

        self.batch = batch
        self.time_step = time_step
        self.n_echar = n_echar
        
        # Paper spec: state channels (8) + action channel (1) = 9 total input channels
        self.state_channels = 8  # State without heading direction
        self.action_space = 7    # Number of possible actions
        self.input_channels = self.state_channels + 1  # State + spatialized action
        
        # Use configurable n_ement channels throughout
        self.resnet_channels = n_ement
        
        # Output is spatial mental state embedding (n_ement channels)
        self.output_channels = n_ement
        
        # Initial conv layer to get to n_ement channels
        self.input_conv = nn.Conv2d(
            self.input_channels, 
            self.resnet_channels, 
            kernel_size=3, 
            padding=1
        )
        self.input_bn = nn.BatchNorm2d(self.resnet_channels)
        
        # Paper spec: 5-layer ResNet with n_ement channels, ReLU, BatchNorm
        self.resnet_layers = nn.ModuleList()
        for _ in range(5):  # Exactly 5 layers as specified
            self.resnet_layers.append(
                ResidualBlock(
                    in_channels=self.resnet_channels,
                    out_channels=self.resnet_channels,
                    kernel_size=3,
                    padding=1
                )
            )
        
        # Paper spec: Convolutional LSTM with n_ement channels
        self.conv_lstm = ConvLSTM2d(
            input_channels=self.resnet_channels,
            hidden_channels=self.resnet_channels,
            kernel_size=3,
            padding=1
        )
        
        # Paper spec: 1-layer convnet with n_ement channels for final output
        self.output_conv = nn.Conv2d(
            self.resnet_channels,
            self.output_channels,
            kernel_size=3,
            padding=1
        )

    def spatialize_action(self, action_indices, height, width):
        """
        Convert action indices to spatial representation
        
        Args:
            action_indices: (batch_size,) - action indices for each sample
            height, width: spatial dimensions
            
        Returns:
            Spatialized actions: (batch_size, 1, height, width)
        """
        batch_size = action_indices.size(0)
        device = action_indices.device
        
        # Create spatial action maps
        # Each action gets a unique value across the entire spatial map
        action_maps = torch.zeros(batch_size, 1, height, width, device=device)
        
        for i in range(batch_size):
            action_idx = action_indices[i].item()
            # Normalize action index to [0, 1] range
            action_value = action_idx / (self.action_space - 1) if self.action_space > 1 else 0
            action_maps[i, 0] = action_value
            
        return action_maps

    def forward(self, recent_trajectory, recent_actions):
        """
        Forward pass following paper specification
        
        Args:
            recent_trajectory: (batch_size, seq_len, channels, height, width) - full trajectory
            recent_actions: (batch_size, seq_len) - actions taken at each timestep
            
        Returns:
            Mental state embedding: (batch_size, output_channels, height, width) - SPATIAL output
        """
        batch_size, seq_len, channels, height, width = recent_trajectory.shape
        
        # Extract state channels (first 8 channels, excluding heading direction)
        states = recent_trajectory[:, :, :self.state_channels]  # (batch_size, seq_len, 8, height, width)
        
        # Check if trajectory is empty (query state is initial state)
        trajectory_sum = torch.sum(states, dim=(2, 3, 4))  # (batch_size, seq_len)
        is_empty = torch.all(trajectory_sum == 0, dim=1)  # (batch_size,)
        
        if torch.all(is_empty):
            # Return zero vector for empty trajectories
            return torch.zeros(batch_size, self.output_channels, height, width, device=recent_trajectory.device)
        
        # Pre-process each timestep: spatialize action + concatenate with state
        processed_timesteps = []
        
        for t in range(seq_len):
            state_t = states[:, t]  # (batch_size, 8, height, width)
            action_t = recent_actions[:, t]  # (batch_size,)
            
            # Spatialize action
            action_spatial_t = self.spatialize_action(action_t, height, width)  # (batch_size, 1, height, width)
            
            # Concatenate state and spatialized action
            combined_t = torch.cat([state_t, action_spatial_t], dim=1)  # (batch_size, 9, height, width)
            
            processed_timesteps.append(combined_t)
        
        # Stack timesteps
        processed_trajectory = torch.stack(processed_timesteps, dim=1)  # (batch_size, seq_len, 9, height, width)
        
        # Pass through initial conv to get to 32 channels
        # Reshape for processing
        trajectory_flat = processed_trajectory.view(batch_size * seq_len, self.input_channels, height, width)
        
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
        lstm_output, (final_h, final_c) = self.conv_lstm(x)
        
        # Use final hidden state from LSTM
        final_features = final_h  # (batch_size, 32, height, width)
        
        # LSTM output through 1-layer convnet with 32 channels
        mental_state = self.output_conv(final_features)  # (batch_size, 32, height, width)
        
        # Handle empty trajectories - set their output to zero
        for i in range(batch_size):
            if is_empty[i]:
                mental_state[i] = 0
        
        return mental_state


class PredNet(nn.Module):
    def __init__(
        self,
        batch: int,
        n_ement: int,
        n_echar: int,
        current_state_channels: int,
        residual_blocks: int,
        action_space: int = 7,
        out_channels: int = 64,
        goal_space: int = 4,
        env_width: int = 9,
        env_height: int = 9,
        use_mentalnet: bool = False,
    ):
        super(PredNet, self).__init__()

        self.batch = batch
        self.n_ement = n_ement
        self.n_echar = n_echar
        self.current_state_channels = current_state_channels
        self.action_space = action_space
        self.goal_space = goal_space
        self.env_width = env_width
        self.env_height = env_height
        self.out_channels = out_channels
        self.n = residual_blocks
        self.use_mentalnet = use_mentalnet

        # Determine input channels based on architecture
        if use_mentalnet:
            # Original 3-stage: current_state + mental_state + character_embedding
            # MentalNet outputs n_ement channels (spatial)
            input_channels = current_state_channels + n_ement + n_echar
        else:
            # Fixed 2-stage: current_state + character_embedding (like experiment5)
            input_channels = current_state_channels + n_echar

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

    def forward(self, mental_state, character_embedding, current_state):
        """
        Forward pass for prediction network (original 3-stage architecture)

        Args:
            mental_state: (batch_size, n_ement, height, width) - spatial mental state from MentalNet
            character_embedding: (batch_size, n_echar)
            current_state: (batch_size, current_state_channels, height, width)

        Returns:
            action_logits, goal_logits, consumption_logits, sr_pred
        """
        batch_size, _, height, width = current_state.shape

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

        return self._forward_shared(x)

    def forward_direct(self, mixed_data):
        """
        Direct forward pass like experiment5 PredNet (fixed 2-stage architecture)

        Args:
            mixed_data: (batch_size, current_state_channels + n_echar, height, width)

        Returns:
            action_logits, goal_logits, consumption_logits, sr_pred
        """
        return self._forward_shared(mixed_data)

    def _forward_shared(self, x):
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

        return action_logits, goal_logits, consumption_logits, sr_pred


class ToMnet(nn.Module):
    def __init__(
        self,
        use_mentalnet: bool = False,
        batch: int = 32,
        residual_blocks: int = 3,
        n_echar: int = 64,
        n_ement: int = 64,
        out_channels: int = 32,
        channels_in: int = 9,  # 8 original channels + 1 heading direction channel (for CharNet)
        current_state_channels: int = 8,  # For PredNet (without heading direction)
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

        self.use_mentalnet = use_mentalnet
        self.batch = batch
        self.n_echar = n_echar
        self.n_ement = n_ement
        self.time_step = time_step
        self.action_space = action_space
        self.goal_space = goal_space
        self.max_n_past = max_n_past
        self.use_n_past = use_n_past
        self.channels_in = channels_in
        self.current_state_channels = current_state_channels
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
                channels_in=current_state_channels,
                time_step=time_step,
                n_echar=n_echar,
            )

        # Prediction network - processes inputs based on architecture
        self.pred_net = PredNet(
            batch=batch,
            n_ement=n_ement,
            n_echar=n_echar,
            current_state_channels=current_state_channels,
            residual_blocks=residual_blocks,
            action_space=action_space,
            out_channels=out_channels,
            goal_space=goal_space,
            env_width=env_width,
            env_height=env_height,
            use_mentalnet=use_mentalnet,
        )

    def forward(self, past_trajectories, recent_trajectory, current_state):
        """
        Forward pass for ToMnet (supports both architectures)

        Args:
            past_trajectories: (batch_size, n_past, seq_len, channels, height, width) - for CharNet
            recent_trajectory: (batch_size, seq_len, channels, height, width) - for MentalNet (if used)
            current_state: (batch_size, channels, height, width) - for PredNet

        Returns:
            action_logits, goal_logits, consumption_logits, sr_pred, character_embedding, mental_state
        """
        # 1. Character network - processes past episodes (same for both architectures)
        if self.use_n_past and past_trajectories is not None:
            character_embedding = self.char_net(past_trajectories)
        else:
            batch_size = current_state.size(0)
            character_embedding = torch.zeros(
                batch_size, self.n_echar, device=current_state.device
            )

        # Extract relevant channels for PredNet
        current_state_for_pred = current_state[:, : self.current_state_channels]
        batch_size, _, height, width = current_state_for_pred.shape

        if self.use_mentalnet:
            # 2a. Original 3-stage architecture: CharNet → MentalNet → PredNet

            # Extract actions from recent trajectory for MentalNet
            # For simplicity, assume actions are embedded in trajectory or use dummy actions
            recent_actions = torch.zeros(batch_size, recent_trajectory.size(1), dtype=torch.long, device=recent_trajectory.device)
            
            # Process recent trajectory through MentalNet
            mental_state = self.mental_net(recent_trajectory, recent_actions)

            # PredNet with mental state
            action_logits, goal_logits, consumption_logits, sr_pred = self.pred_net(
                mental_state, character_embedding, current_state_for_pred
            )
        else:
            # 2b. Fixed 2-stage architecture: CharNet → PredNet (like experiment5)

            # Reshape character embedding to spatial format
            e_char_spatial = (
                character_embedding.unsqueeze(2)
                .unsqueeze(3)
                .expand(batch_size, self.n_echar, height, width)
            )

            # Concatenate current_state with character embedding
            mixed_data = torch.cat([current_state_for_pred, e_char_spatial], dim=1)

            # PredNet processes mixed data directly
            action_logits, goal_logits, consumption_logits, sr_pred = (
                self.pred_net.forward_direct(mixed_data)
            )

            # Create dummy mental_state for compatibility
            mental_state = torch.zeros(
                batch_size, self.n_ement, device=current_state.device
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
        """Predict next action"""
        with torch.no_grad():
            action_logits, _, _, _, _, _ = self.forward(
                past_trajectories, recent_trajectory, current_state
            )
            return F.softmax(action_logits, dim=1)

    def predict_goal(self, past_trajectories, recent_trajectory, current_state):
        """Predict goal"""
        with torch.no_grad():
            _, goal_logits, _, _, _, _ = self.forward(
                past_trajectories, recent_trajectory, current_state
            )
            return F.softmax(goal_logits, dim=1)

    def get_character_embedding(self, past_trajectories):
        """Get character embedding from past trajectories"""
        with torch.no_grad():
            if self.use_n_past and past_trajectories is not None:
                return self.char_net(past_trajectories)
            else:
                batch_size = 1
                return torch.zeros(batch_size, self.n_echar)

    def get_mental_state(self, recent_trajectory, character_embedding):
        """Get mental state from recent trajectory and character embedding"""
        with torch.no_grad():
            if self.use_mentalnet:
                # Use new MentalNet interface - pass full trajectory and character embedding
                mental_state = self.mental_net(recent_trajectory, character_embedding)
                return mental_state
            else:
                # Return dummy mental state for fixed architecture
                batch_size = (
                    1 if recent_trajectory is None else recent_trajectory.size(0)
                )
                return torch.zeros(batch_size, self.n_ement)


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
        self.consumption_loss = nn.BCEWithLogitsLoss()

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
        """Compute combined loss"""
        action_loss = self.action_loss(action_logits, action_targets)
        goal_loss = self.goal_loss(goal_logits, goal_targets)
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
    """Create ToMnet model from configuration"""
    # Handle both Config object and dictionary
    if hasattr(config, "get_model_kwargs"):
        # Config object
        model_kwargs = config.get_model_kwargs()
        model = ToMnet(**model_kwargs)
    else:
        # Dictionary
        model = ToMnet(
            use_mentalnet=config.get("use_mentalnet", False),
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
    # Test both architectures
    for use_mentalnet in [False, True]:
        print(
            f"\n=== Testing {'Original' if use_mentalnet else 'Fixed'} Architecture ==="
        )

        config = {
            "use_mentalnet": use_mentalnet,
            "batch_size": 32,
            "residual_blocks": 3,
            "n_echar": 16,
            "n_ement": 16,
            "out_channels": 32,
            "channels_in": 9,
            "current_state_channels": 8,
            "time_step": 20,
            "action_space": 7,
            "goal_space": 4,
            "max_n_past": 1,
            "use_n_past": True,
        }

        model = create_model(config)
        print(f"Model created with {count_parameters(model):,} parameters")

        # Test forward pass
        batch_size = 4
        past_trajectories = torch.randn(batch_size, 1, 20, 9, 9, 9)
        recent_trajectory = torch.randn(batch_size, 20, 9, 9, 9)
        current_state = torch.randn(batch_size, 9, 9, 9)

        outputs = model(past_trajectories, recent_trajectory, current_state)
        (
            action_logits,
            goal_logits,
            consumption_logits,
            sr_pred,
            char_emb,
            mental_state,
        ) = outputs

        print(f"Action logits: {action_logits.shape}")
        print(f"Goal logits: {goal_logits.shape}")
        print(f"Consumption logits: {consumption_logits.shape}")
        print(f"SR pred: {sr_pred.shape}")
        print(f"Character embedding: {char_emb.shape}")
        print(f"Mental state: {mental_state.shape}")
