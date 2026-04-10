"""EDSR style image restoration model."""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels, res_scale=0.1):
        super().__init__()
        self.res_scale = res_scale
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.layers(x) * self.res_scale


class EDSR(nn.Module):
    def __init__(self, n_channels_in, n_classes_out, start_filters=224):
        super().__init__()
        channels = start_filters * 2
        self.head = nn.Conv2d(n_channels_in, channels, kernel_size=3, padding=1)
        self.body = nn.Sequential(*[ResidualBlock(channels) for _ in range(16)])
        self.body_tail = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.tail = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, n_classes_out, kernel_size=3, padding=1),
        )
        self.skip = nn.Conv2d(n_channels_in, n_classes_out, kernel_size=1)
        self.final_activation = nn.Tanh()

    def forward(self, x):
        features = self.head(x)
        residual = self.body_tail(self.body(features)) + features
        output = self.tail(residual) + self.skip(x)
        return self.final_activation(output)
