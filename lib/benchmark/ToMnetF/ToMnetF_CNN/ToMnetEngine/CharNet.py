import torch
from torch import nn
from ToMnetEngine.Blocks.TimeDistirbutedCnn import TimeDistributedConv2d as Conv2dT
from ToMnetEngine.Blocks.LSTM import LSTM as lstm
from ToMnetEngine.Blocks.ResBlock import ResidualBlock

"""
@Author Filip Borowiak 
"""


class CharNet(nn.Module):
 def __init__(self, Batch: int, ResidualBlocks: int,  N_echar: int, out_channels: int, channels_in: int, time_frame:int):
    super(CharNet, self).__init__()

    #current_state_shape = (B, 10, 13, 13, 12) # batch, channel, height, width, time_frame
    self.n = ResidualBlocks
    self.N_echar = N_echar
    self.out_channels = out_channels
    self.channels_in = channels_in
    self.B = Batch # Batch size
    self.time_frame = time_frame # sequence length = time frame
    self.hidden_size_lstm = 64 #128 # 64
    

    self.conv_1 = Conv2dT(time_frame=self.time_frame) # Use time frame conv2d to process different lengths of sequence
    self.res_blocks = [None] * self.n

    for i in range(self.n):
      self.res_blocks[i] = ResidualBlock(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=(3,3), padding=1, stride=1)

    self.lstm = lstm(self.out_channels, self.hidden_size_lstm)

    self.e_char = nn.Linear(self.hidden_size_lstm, N_echar)

 def forward(self, x):
    x = self.conv_1(x) # (Batch x channels x Width x Height x time frame)
    

    for i in range(self.n):
      self.res_blocks[i] = ResidualBlock(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=(3,3), padding=1, stride=1)
    
    x = torch.mean(x, [2, 3])
    
    
    x = x.reshape([x.size(0), self.time_frame, self.out_channels])
    x = self.lstm(x) 
    

    x = self.e_char(x)
    
    return x
