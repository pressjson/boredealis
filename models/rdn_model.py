"""Residual Dense Network for image restoration."""

import torch
import torch.nn as nn


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        new_features = self.layer(x)
        return torch.cat([x, new_features], dim=1)


class ResidualDenseBlock(nn.Module):
    def __init__(self, channels, growth_rate=32, num_layers=6):
        super().__init__()
        layers = []
        current_channels = channels
        for _ in range(num_layers):
            layers.append(DenseLayer(current_channels, growth_rate))
            current_channels += growth_rate
        self.layers = nn.Sequential(*layers)
        self.lff = nn.Conv2d(current_channels, channels, kernel_size=1)

    def forward(self, x):
        return self.lff(self.layers(x)) + x


class RDN(nn.Module):
    def __init__(self, n_channels_in, n_classes_out, start_filters=160):
        super().__init__()
        channels = start_filters * 2
        self.sfe1 = nn.Conv2d(n_channels_in, channels, kernel_size=3, padding=1)
        self.sfe2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.rdbs = nn.ModuleList([ResidualDenseBlock(channels) for _ in range(6)])
        self.gff = nn.Sequential(
            nn.Conv2d(channels * len(self.rdbs), channels, kernel_size=1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.reconstruction = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, n_classes_out, kernel_size=3, padding=1),
        )
        self.skip = nn.Conv2d(n_channels_in, n_classes_out, kernel_size=1)
        self.final_activation = nn.Tanh()

    def forward(self, x):
        skip = self.skip(x)
        sfe1 = self.sfe1(x)
        x = self.sfe2(sfe1)

        local_features = []
        for rdb in self.rdbs:
            x = rdb(x)
            local_features.append(x)

        x = self.gff(torch.cat(local_features, dim=1)) + sfe1
        x = self.reconstruction(x) + skip
        return self.final_activation(x)
