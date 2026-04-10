"""Attention U-Net architecture."""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.layers(x)


class AttentionGate(nn.Module):
    def __init__(self, gate_channels, skip_channels, inter_channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.skip = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, gate, skip):
        attention = self.psi(self.gate(gate) + self.skip(skip))
        return skip * attention


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.attention = AttentionGate(out_channels, skip_channels, max(out_channels // 2, 1))
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        skip = self.attention(x, skip)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class AttentionUNet(nn.Module):
    def __init__(self, n_channels_in, n_classes_out, start_filters=88):
        super().__init__()
        sf = start_filters
        self.inc = DoubleConv(n_channels_in, sf)
        self.down1 = Down(sf, sf * 2)
        self.down2 = Down(sf * 2, sf * 4)
        self.down3 = Down(sf * 4, sf * 8)
        self.down4 = Down(sf * 8, sf * 16)
        self.down5 = Down(sf * 16, sf * 32)

        self.up1 = Up(sf * 32, sf * 16, sf * 16)
        self.up2 = Up(sf * 16, sf * 8, sf * 8)
        self.up3 = Up(sf * 8, sf * 4, sf * 4)
        self.up4 = Up(sf * 4, sf * 2, sf * 2)
        self.up5 = Up(sf * 2, sf, sf)

        self.outc = nn.Conv2d(sf, n_classes_out, kernel_size=1)
        self.final_activation = nn.Tanh()

    def forward(self, x):
        s1 = self.inc(x)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        s5 = self.down4(s4)
        bottleneck = self.down5(s5)

        x = self.up1(bottleneck, s5)
        x = self.up2(x, s4)
        x = self.up3(x, s3)
        x = self.up4(x, s2)
        x = self.up5(x, s1)
        return self.final_activation(self.outc(x))
