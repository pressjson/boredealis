"""NAFNet style image restoration model."""

import torch
import torch.nn as nn


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight + self.bias


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, channels, dw_expand=2, ffn_expand=2):
        super().__init__()
        dw_channels = channels * dw_expand
        ffn_channels = channels * ffn_expand

        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, dw_channels, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(
            dw_channels,
            dw_channels,
            kernel_size=3,
            padding=1,
            groups=dw_channels,
            bias=True,
        )
        self.sg = SimpleGate()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channels // 2, dw_channels // 2, kernel_size=1, bias=True),
        )
        self.conv3 = nn.Conv2d(dw_channels // 2, channels, kernel_size=1, bias=True)

        self.norm2 = LayerNorm2d(channels)
        self.conv4 = nn.Conv2d(channels, ffn_channels * 2, kernel_size=1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channels, channels, kernel_size=1, bias=True)

        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = residual + x * self.beta

        residual = x
        x = self.norm2(x)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)
        return residual + x * self.gamma


class NAFNet(nn.Module):
    def __init__(self, n_channels_in, n_classes_out, start_filters=48):
        super().__init__()
        sf = start_filters
        channels = [sf, sf * 2, sf * 4, sf * 8, sf * 16]

        self.intro = nn.Conv2d(n_channels_in, channels[0], kernel_size=3, padding=1)
        self.encoders = nn.ModuleList(
            [
                nn.Sequential(NAFBlock(channels[0]), NAFBlock(channels[0])),
                nn.Sequential(NAFBlock(channels[1]), NAFBlock(channels[1])),
                nn.Sequential(NAFBlock(channels[2]), NAFBlock(channels[2])),
                nn.Sequential(NAFBlock(channels[3]), NAFBlock(channels[3])),
            ]
        )
        self.downs = nn.ModuleList(
            [
                nn.Conv2d(channels[0], channels[1], kernel_size=2, stride=2),
                nn.Conv2d(channels[1], channels[2], kernel_size=2, stride=2),
                nn.Conv2d(channels[2], channels[3], kernel_size=2, stride=2),
                nn.Conv2d(channels[3], channels[4], kernel_size=2, stride=2),
            ]
        )
        self.middle = nn.Sequential(NAFBlock(channels[4]), NAFBlock(channels[4]))
        self.ups = nn.ModuleList(
            [
                nn.ConvTranspose2d(channels[4], channels[3], kernel_size=2, stride=2),
                nn.ConvTranspose2d(channels[3], channels[2], kernel_size=2, stride=2),
                nn.ConvTranspose2d(channels[2], channels[1], kernel_size=2, stride=2),
                nn.ConvTranspose2d(channels[1], channels[0], kernel_size=2, stride=2),
            ]
        )
        self.decoders = nn.ModuleList(
            [
                nn.Sequential(NAFBlock(channels[3]), NAFBlock(channels[3])),
                nn.Sequential(NAFBlock(channels[2]), NAFBlock(channels[2])),
                nn.Sequential(NAFBlock(channels[1]), NAFBlock(channels[1])),
                nn.Sequential(NAFBlock(channels[0]), NAFBlock(channels[0])),
            ]
        )
        self.outro = nn.Conv2d(channels[0], n_classes_out, kernel_size=3, padding=1)
        self.skip = nn.Conv2d(n_channels_in, n_classes_out, kernel_size=1)
        self.final_activation = nn.Tanh()

    def forward(self, x):
        residual = self.skip(x)
        x = self.intro(x)

        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            skips.append(x)
            x = down(x)

        x = self.middle(x)

        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = up(x)
            x = x + skip
            x = decoder(x)

        x = self.outro(x) + residual
        return self.final_activation(x)
