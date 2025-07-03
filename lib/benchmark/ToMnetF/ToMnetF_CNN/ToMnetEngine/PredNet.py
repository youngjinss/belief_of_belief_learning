import torch
from torch import nn
import torch.nn.functional as F
from ToMnetEngine.Blocks.ResBlock import ResidualBlock

"""
@Author Filip Borowiak 
"""


class PredNet(nn.Module):
  def __init__(self, Batch: int, ResidualBlocks: int,  E_char: int, out_channels: int, time_frame: int):
    super(PredNet, self).__init__()
    self.n = ResidualBlocks
    self.B = Batch
    self.e_char_shape = E_char # 8
    self.current_state_shape = (self.B, 7, 13, 13) # batch, channel, height, width
    self.softmax = nn.Softmax(dim=1)
    self.out_channels = out_channels
    self.time_sequence = time_frame

    self.conv_1 = nn.Conv2d(in_channels=self.current_state_shape[1], out_channels=self.out_channels, kernel_size=(3,3), stride=1, padding=1)
    self.res_blocks = [None] * self.n

    for i in range(self.n):
      self.res_blocks[i] = ResidualBlock(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=(3,3), padding=1, stride=1)

    self.conv_2 = (nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=(3,3), stride=1, padding=1))

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
    x = F.relu(x)
   


    x = self.softmax(x)


    return x
