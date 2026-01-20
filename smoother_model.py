#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint


class ResidualBlock(nn.Module):
    """
    Residual Block: x + Conv(Relu(Conv(x)))
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.conv(x)

class DeflickerCNN(nn.Module):
    """
    Multi-frame Convolutional Network for Video Deflickering.
    
    Input: n RGB frames concatenated channel-wise (..., t-2, t-1, t, t+1, t+2, ...).
           Shape: [Batch, input_frames * 3, Height, Width] (Designed for 608x608 inputs)
    Output: 1 RGB frame (stabilized frame t).
            Shape: [Batch, 3, Height, Width]

    Args:
        input_frames: Size of the window. Must be odd.
        num_res_blocks: How many residual blocks to use
        hidden_channels: How many convolution filters per res block
        save_memory: False -> faster compute, but more VRAM; True -> slower compute, less VRAM
            use False for testing, and True for training
    """
    def __init__(self, input_frames=5, num_res_blocks=8, hidden_channels=64, save_memory=False):
        if input_frames % 2 == 0:
            print(f"Error: input_frames must be odd, currently {input_frames}")
            exit(-1)
        super(DeflickerCNN, self).__init__()

        self.save_memory = save_memory
        self.input_frames = input_frames
        
        in_channels = input_frames * 3  # frames * 3 channels (R, G, B)       

        # Initial Feature Extraction
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Backbone: Stack of Residual Blocks
        res_blocks = []
        for _ in range(num_res_blocks):
            res_blocks.append(ResidualBlock(hidden_channels))
        self.body = nn.Sequential(*res_blocks)
        
        # Final Reconstruction
        self.tail = nn.Conv2d(hidden_channels, 3, kernel_size=3, padding=1)

        # force init weights to zero for reconstruction loss during training
        nn.init.constant_(self.tail.weight, 0)
        nn.init.constant_(self.tail.bias, 0)
        
    def forward(self, x):
        features = self.head(x)

        if self.save_memory:
            for layer in self.body:
                features = checkpoint.checkpoint(layer, features, use_reentrant=False)
        else:
            features = self.body(features)
            
        out = self.tail(features)
        
        start_channel = (self.input_frames // 2) * 3 # middle frame * (3 channels R,G,B)
        input_t = x[:, start_channel : start_channel + 3, :, :]
        
        return torch.clamp(out + input_t, 0, 1)
