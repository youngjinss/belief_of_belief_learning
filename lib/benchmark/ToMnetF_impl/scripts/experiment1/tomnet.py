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
    def __init__(self, time_frame: int):
        super(TimeDistributedConv2d, self).__init__()
        self.time_frame = time_frame
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
        time_frame: int,
    ):
        super(CharNet, self).__init__()

        # current_state_shape = (B, 10, 13, 13, 12) # batch, channel, height, width, time_frame
        self.n = ResidualBlocks
        self.N_echar = N_echar
        self.out_channels = out_channels
        self.channels_in = channels_in
        self.B = Batch  # Batch size
        self.time_frame = time_frame  # sequence length = time frame
        self.hidden_size_lstm = 64  # 128 # 64

        self.conv_1 = TimeDistributedConv2d(
            time_frame=self.time_frame
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

        self.e_char = nn.Linear(self.hidden_size_lstm, N_echar)

    def forward(self, x):
        x = self.conv_1(x)  # (Batch x channels x Width x Height x time frame)

        for i in range(self.n):
            x = self.res_blocks[i](x)

        x = torch.mean(x, [2, 3])

        x = x.reshape([x.size(0), self.time_frame, self.out_channels])
        x = self.lstm(x)

        x = self.e_char(x)

        return x


class PredNet(nn.Module):
    def __init__(
        self,
        Batch: int,
        ResidualBlocks: int,
        E_char: int,
        out_channels: int,
        time_frame: int,
    ):
        super(PredNet, self).__init__()
        self.n = ResidualBlocks
        self.B = Batch
        self.e_char_shape = E_char  # 8
        self.current_state_shape = (self.B, 7, 13, 13)  # batch, channel, height, width
        self.softmax = nn.Softmax(dim=1)
        self.out_channels = out_channels
        self.time_sequence = time_frame

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

        self.fc1 = nn.Linear(out_channels, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 4)

    def forward(self, x):

        x = self.conv_1(x)

        for i in range(self.n):
            x = self.res_blocks[i](x)

        x = self.conv_2(x)
        x = F.relu(x)

        x = torch.mean(x, [2, 3])

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        # Remove ReLU and softmax - CrossEntropyLoss handles this

        return x


class ToMnet(nn.Module):
    def __init__(
        self,
        Batch: int,
        ResidualBlocks: int,
        N_echar: int,
        out_channels: int,
        Max_trajectory_size: int,
        Width: int,
        Height: int,
        Depth: int,
    ):
        super(ToMnet, self).__init__()

        self.ts = Max_trajectory_size
        self.W = Width
        self.H = Height
        self.C = Depth
        self.B = Batch
        self.resN = ResidualBlocks
        self.Length_E = N_echar
        self.out_channels = out_channels

        self.char_net = CharNet(
            Batch=self.B,
            ResidualBlocks=self.resN,
            N_echar=self.Length_E,
            channels_in=self.C,
            out_channels=self.out_channels,
            time_frame=self.ts,
        )

        self.pred_net = PredNet(
            Batch=self.B,
            ResidualBlocks=self.resN,
            E_char=self.Length_E,
            out_channels=self.out_channels,
            time_frame=self.ts,
        )

    def SaveModel(self, destination):
        torch.save(self.state_dict(), destination)

    def forward(self, data):
        input_trajectory = data[0]  # input_traj
        input_current_state = data[1]  #  input_current

        e_char = self.char_net(input_trajectory)

        e_char_new = torch.concat([e_char, e_char], dim=1)
        e_char_new = e_char_new[..., 0:13]

        e_char_new = torch.unsqueeze(e_char_new, dim=-1)

        e_char_new = torch.repeat_interleave(e_char_new, repeats=13, dim=-1)

        e_char_new = torch.unsqueeze(e_char_new, dim=1)

        mixed_data = torch.cat((e_char_new, input_current_state), dim=1)

        pred = self.pred_net(mixed_data)

        return pred
