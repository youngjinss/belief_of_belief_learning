import torch
import torch.nn as nn
from .ResBlock import ResidualBlock

"""
Time-distributed wrapper for ResidualBlock to handle 5D tensors
"""


class TimeDistributedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(TimeDistributedResidualBlock, self).__init__()
        
        self.res_block = ResidualBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride
        )
    
    def forward(self, x):
        # Input shape: (B, C, H, W, T)
        B, C, H, W, T = x.size()
        
        # Reshape to combine batch and time dimensions
        x = x.permute(0, 4, 1, 2, 3)  # (B, T, C, H, W)
        x = x.reshape(B * T, C, H, W)  # (B*T, C, H, W)
        
        # Apply residual block
        x = self.res_block(x)
        
        # Reshape back to 5D
        _, C_out, H_out, W_out = x.size()
        x = x.reshape(B, T, C_out, H_out, W_out)  # (B, T, C_out, H_out, W_out)
        x = x.permute(0, 2, 3, 4, 1)  # (B, C_out, H_out, W_out, T)
        
        return x