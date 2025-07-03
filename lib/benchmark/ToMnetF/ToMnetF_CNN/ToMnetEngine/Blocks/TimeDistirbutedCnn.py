import torch.nn as nn

"""
@Author Filip Borowiak 
"""

class TimeDistributedConv2d(nn.Module):
    def __init__(self, time_frame):
        super(TimeDistributedConv2d, self).__init__()

        self.current_state_shape = (16, 10, 13, 13, time_frame) # batch, channel (depth), height, width, time_frame (ts)
        self.module = nn.Conv2d(in_channels=10, out_channels=32, kernel_size=(3,3), stride=1, padding=1)


    def forward(self, x):
        B, C, H, W, T =  x.size()
        # Combine batch size and time frames
        x_re = x.reshape(B * T, C, H, W)

        # pass through conv2d
        x = self.module(x_re)
        x = x.reshape(B, x.size(1), x.size(2), x.size(3), T) # reshape to original format

        return x
    



