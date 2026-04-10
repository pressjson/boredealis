"""DeepLabV3+ style image-to-image model."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class ASPPBranch(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        kernel_size = 1 if dilation == 1 else 3
        padding = 0 if dilation == 1 else dilation
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class ImagePooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        pooled = self.proj(self.pool(x))
        return F.interpolate(pooled, size=x.shape[-2:], mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                ASPPBranch(in_channels, out_channels, 1),
                ASPPBranch(in_channels, out_channels, 6),
                ASPPBranch(in_channels, out_channels, 12),
                ASPPBranch(in_channels, out_channels, 18),
                ImagePooling(in_channels, out_channels),
            ]
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * len(self.branches), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        features = [branch(x) for branch in self.branches]
        return self.project(torch.cat(features, dim=1))


class DeepLabV3Plus(nn.Module):
    def __init__(self, n_channels_in, n_classes_out, start_filters=160):
        super().__init__()
        sf = start_filters
        self.stem = ConvBlock(n_channels_in, sf, stride=2)
        self.stage1 = ConvBlock(sf, sf * 2, stride=2)
        self.stage2 = ConvBlock(sf * 2, sf * 4, stride=2)
        self.stage3 = ConvBlock(sf * 4, sf * 8, stride=2)
        self.stage4 = ConvBlock(sf * 8, sf * 16, stride=2)

        self.aspp = ASPP(sf * 16, sf * 4)
        self.low_level = nn.Sequential(
            nn.Conv2d(sf * 2, sf, kernel_size=1, bias=False),
            nn.BatchNorm2d(sf),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(sf * 5, sf * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(sf * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(sf * 2, sf * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(sf * 2),
            nn.ReLU(inplace=True),
        )
        self.outc = nn.Conv2d(sf * 2, n_classes_out, kernel_size=1)
        self.final_activation = nn.Tanh()

    def forward(self, x):
        input_size = x.shape[-2:]
        x = self.stem(x)
        low_level = self.stage1(x)
        x = self.stage2(low_level)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.aspp(x)
        x = F.interpolate(x, size=low_level.shape[-2:], mode="bilinear", align_corners=False)
        low_level = self.low_level(low_level)
        x = self.decoder(torch.cat([x, low_level], dim=1))
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return self.final_activation(self.outc(x))
