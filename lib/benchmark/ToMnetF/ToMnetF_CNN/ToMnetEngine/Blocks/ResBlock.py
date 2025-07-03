from torch import nn

"""
@Author Filip Borowiak 
"""

class ResidualBlock(nn.Module):
  def __init__(self, in_channels: int, out_channels: int, stride: int, kernel_size: int, padding: int, downsample = None):
    super(ResidualBlock, self).__init__()

    # Block structure
    self.convBlock_1 = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
        nn.BatchNorm2d(out_channels),
        nn.ReLU())
    self.convBlock_2 = nn.Sequential(
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
        nn.BatchNorm2d(out_channels))

    self.relu = nn.ReLU()
    self.out_channels = out_channels


  def forward(self, x):
    residual = x
    x = self.convBlock_1(x)
    x = self.convBlock_2(x)

    return self.relu(x + residual) # skipping connection