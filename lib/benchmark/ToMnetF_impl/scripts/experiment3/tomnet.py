import torch
from torch import nn
import torch.nn.functional as F

"""
ToMnetF architecture for experiment1 (CNN-based ToMnet)
@Author Filip Borowiak
"""


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        kernel_size: int,
        padding: int,
        downsample=None,
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
        # x.shape -> (seq_len, batch_size, input_size)
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(
            x.device
        )  # initial hidden state
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(
            x.device
        )  # initial cell state
        out, _ = self.lstm(x, (h0, c0))

        return out[:, -1, :]  # out shape -> (seq_len, batch_size, hidden_size)


class TimeDistributedResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        kernel_size: int,
        padding: int,
    ):
        super(TimeDistributedResidualBlock, self).__init__()
        self.res_block = ResidualBlock(
            in_channels, out_channels, stride, kernel_size, padding
        )

    def forward(self, x):
        # x shape: (batch, channels, height, width, time)
        batch_size, channels, height, width, time_steps = x.size()

        # Reshape to (batch * time, channels, height, width)
        x_reshaped = x.permute(0, 4, 1, 2, 3).contiguous()
        x_reshaped = x_reshaped.view(batch_size * time_steps, channels, height, width)

        # Apply residual block
        x_processed = self.res_block(x_reshaped)

        # Reshape back to (batch, channels, height, width, time)
        _, out_channels, out_height, out_width = x_processed.size()
        x_output = x_processed.view(
            batch_size, time_steps, out_channels, out_height, out_width
        )
        x_output = x_output.permute(0, 2, 3, 4, 1).contiguous()

        return x_output


class TimeDistributedConv2d(nn.Module):
    def __init__(self, time_step: int):
        super(TimeDistributedConv2d, self).__init__()
        self.time_step = time_step
        self.conv = nn.Conv2d(
            in_channels=10,  # depth/channels
            out_channels=32,  # out_channels
            kernel_size=(3, 3),
            stride=1,
            padding=1,
        )

    def forward(self, x):
        # x shape: (batch, channels, height, width, time)
        batch_size, channels, height, width, time_steps = x.size()

        # Reshape to (batch * time, channels, height, width)
        x_reshaped = x.permute(0, 4, 1, 2, 3).contiguous()
        x_reshaped = x_reshaped.view(batch_size * time_steps, channels, height, width)

        # Apply convolution
        x_processed = self.conv(x_reshaped)

        # Reshape back to (batch, channels, height, width, time)
        _, out_channels, out_height, out_width = x_processed.size()
        x_output = x_processed.view(
            batch_size, time_steps, out_channels, out_height, out_width
        )
        x_output = x_output.permute(0, 2, 3, 4, 1).contiguous()

        return x_output


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

        # current_state_shape = (B, 10, 13, 13, 12) # batch, channel, height, width, time_frame
        self.n = ResidualBlocks
        self.N_echar = N_echar
        self.out_channels = out_channels
        self.channels_in = channels_in
        self.B = Batch  # Batch size
        self.time_step = time_step  # sequence length = time frame
        self.hidden_size_lstm = 64  # 128 # 64
        self.max_n_past = max_n_past
        self.use_n_past = use_n_past

        self.conv_1 = TimeDistributedConv2d(
            time_step=self.time_step
        )  # Use time frame conv2d to process different lengths of sequence
        self.res_blocks = nn.ModuleList()

        for i in range(self.n):
            self.res_blocks.append(
                TimeDistributedResidualBlock(
                    in_channels=self.out_channels,
                    out_channels=self.out_channels,
                    kernel_size=(3, 3),
                    padding=1,
                    stride=1,
                )
            )

        self.lstm = LSTM(self.out_channels, self.hidden_size_lstm)

        # Character embedding layers
        if self.use_n_past:
            # Past episode processing network
            # Input: (position_x, position_y, action_one_hot) = 6 dimensions
            self.past_episode_encoder = nn.Sequential(
                nn.Linear(6, 32),
                nn.ReLU(),
                nn.Linear(32, self.N_echar),
            )
            
            # Final character embedding combines current trajectory + past episodes
            self.e_char = nn.Linear(self.hidden_size_lstm, N_echar)
        else:
            self.e_char = nn.Linear(self.hidden_size_lstm, N_echar)

    def forward(self, x, past_episodes=None, n_past=None):
        """
        Args:
            x: Current trajectory tensor (Batch x channels x Width x Height x time_frame)
            past_episodes: Past episodes tensor (Batch x max_n_past x 6)
            n_past: Number of past episodes per sample (Batch,)
        """
        # Process current trajectory
        x = self.conv_1(x)  # (Batch x channels x Width x Height x time frame)

        for i in range(self.n):
            x = self.res_blocks[i](x)

        x = torch.mean(x, [2, 3])

        x = x.reshape([x.size(0), self.time_step, self.out_channels])
        x = self.lstm(x)

        # Get trajectory-based character embedding
        e_char_traj = self.e_char(x)

        # Process past episodes if available
        if self.use_n_past and past_episodes is not None:
            batch_size = past_episodes.size(0)
            
            # Initialize character embedding from past episodes
            e_char_past = torch.zeros(batch_size, self.N_echar).to(past_episodes.device)
            
            # Process each sample in the batch
            for i in range(batch_size):
                if n_past is not None and n_past[i] > 0:
                    # Get past episodes for this sample
                    sample_past_episodes = past_episodes[i, :n_past[i]]  # (n_past[i], 6)
                    
                    # Encode each past episode
                    encoded_episodes = self.past_episode_encoder(sample_past_episodes)  # (n_past[i], N_echar)
                    
                    # Sum contributions according to paper: e_char,i = sum(e_char,ij)
                    e_char_past[i] = torch.sum(encoded_episodes, dim=0)
            
            # Combine trajectory and past episode embeddings
            e_char_combined = e_char_traj + e_char_past
            
            return e_char_combined
        else:
            return e_char_traj


class PredNet(nn.Module):
    def __init__(
        self,
        Batch: int,
        ResidualBlocks: int,
        E_char: int,
        out_channels: int,
        time_step: int,
    ):
        super(PredNet, self).__init__()
        self.n = ResidualBlocks
        self.B = Batch
        self.e_char_shape = E_char  # 8
        self.current_state_shape = (self.B, 7, 13, 13)  # batch, channel, height, width
        self.softmax = nn.Softmax(dim=1)
        self.out_channels = out_channels
        self.time_sequence = time_step

        # Shared torso
        self.conv_1 = nn.Conv2d(
            in_channels=self.current_state_shape[1],
            out_channels=self.out_channels,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
        )
        self.res_blocks = nn.ModuleList()

        for i in range(self.n):
            self.res_blocks.append(
                ResidualBlock(
                    in_channels=self.out_channels,
                    out_channels=self.out_channels,
                    kernel_size=(3, 3),
                    padding=1,
                    stride=1,
                )
            )

        self.conv_2 = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
        )

        # Shared feature extraction
        self.fc1 = nn.Linear(out_channels, 64)
        self.fc2 = nn.Linear(64, 128)

        # Action prediction head
        self.fc3_action = nn.Linear(128, 4)

        # Consumption prediction head (4 goals: A, B, C, D)
        # Each output represents p(c_k) for object k being consumed
        self.fc3_consumption = nn.Linear(128, 4)

        # SR prediction heads for different discount factors
        # Output 13x13 grids for 3 different gammas
        self.conv_sr = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=32,
            kernel_size=(1, 1),
        )
        self.conv_sr_out = nn.Conv2d(
            in_channels=32,
            out_channels=3,  # 3 discount factors
            kernel_size=(1, 1),
        )

    def forward(self, x):
        # Shared torso
        x = self.conv_1(x)

        for i in range(self.n):
            x = self.res_blocks[i](x)

        x = self.conv_2(x)
        x = F.relu(x)

        # Store spatial features for SR prediction
        spatial_features = x

        # Global pooling for action and consumption predictions
        x_pooled = torch.mean(x, [2, 3])

        x_pooled = self.fc1(x_pooled)
        x_pooled = F.relu(x_pooled)

        x_pooled = self.fc2(x_pooled)
        x_pooled = F.relu(x_pooled)

        # Action prediction
        action_pred = self.fc3_action(x_pooled)

        # Consumption prediction (raw logits - sigmoid applied in loss function)
        consumption_pred = self.fc3_consumption(x_pooled)

        # SR prediction (using spatial features)
        sr_features = self.conv_sr(spatial_features)
        sr_features = F.relu(sr_features)
        sr_pred = self.conv_sr_out(sr_features)

        # Apply softmax to each SR channel independently
        batch_size, channels, height, width = sr_pred.shape
        sr_pred = sr_pred.view(batch_size, channels, -1)
        sr_pred = F.softmax(sr_pred, dim=2)
        sr_pred = sr_pred.view(batch_size, channels, height, width)

        return action_pred, consumption_pred, sr_pred


class ToMnet(nn.Module):
    def __init__(
        self,
        Batch: int,
        ResidualBlocks: int,
        N_echar: int,
        out_channels: int,
        time_step: int,
        Width: int,
        Height: int,
        Depth: int,
        max_n_past: int = 10,
        use_n_past: bool = True,
    ):
        super(ToMnet, self).__init__()

        self.time_step = time_step
        self.W = Width
        self.H = Height
        self.C = Depth
        self.B = Batch
        self.resN = ResidualBlocks
        self.Length_E = N_echar
        self.out_channels = out_channels
        self.max_n_past = max_n_past
        self.use_n_past = use_n_past

        self.char_net = CharNet(
            Batch=self.B,
            ResidualBlocks=self.resN,
            N_echar=self.Length_E,
            channels_in=self.C,
            out_channels=self.out_channels,
            time_step=self.time_step,
            max_n_past=self.max_n_past,
            use_n_past=self.use_n_past,
        )

        self.pred_net = PredNet(
            Batch=self.B,
            ResidualBlocks=self.resN,
            E_char=self.Length_E,
            out_channels=self.out_channels,
            time_step=self.time_step,
        )

    def SaveModel(self, destination):
        torch.save(self.state_dict(), destination)

    def forward(self, data):
        input_trajectory = data[0]  # input_traj
        input_current_state = data[1]  #  input_current
        
        # Check if past episodes data is provided
        if len(data) > 2 and self.use_n_past:
            past_episodes = data[2]  # past episodes tensor
            n_past = data[3] if len(data) > 3 else None  # number of past episodes
            e_char = self.char_net(input_trajectory, past_episodes, n_past)
        else:
            e_char = self.char_net(input_trajectory)

        e_char_new = torch.concat([e_char, e_char], dim=1)
        e_char_new = e_char_new[..., 0:13]

        e_char_new = torch.unsqueeze(e_char_new, dim=-1)

        e_char_new = torch.repeat_interleave(e_char_new, repeats=13, dim=-1)

        e_char_new = torch.unsqueeze(e_char_new, dim=1)

        mixed_data = torch.cat((e_char_new, input_current_state), dim=1)

        action_pred, consumption_pred, sr_pred = self.pred_net(mixed_data)

        return action_pred, consumption_pred, sr_pred
