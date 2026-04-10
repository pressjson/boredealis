"""Compact U-Net++ architecture."""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
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


class UNetPlusPlus(nn.Module):
    def __init__(self, n_channels_in, n_classes_out, start_filters=56):
        super().__init__()
        f0 = start_filters
        f1 = f0 * 2
        f2 = f1 * 2
        f3 = f2 * 2
        f4 = f3 * 2
        f5 = f4 * 2

        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.x0_0 = ConvBlock(n_channels_in, f0)
        self.x1_0 = ConvBlock(f0, f1)
        self.x2_0 = ConvBlock(f1, f2)
        self.x3_0 = ConvBlock(f2, f3)
        self.x4_0 = ConvBlock(f3, f4)
        self.x5_0 = ConvBlock(f4, f5)

        self.x0_1 = ConvBlock(f0 + f1, f0)
        self.x1_1 = ConvBlock(f1 + f2, f1)
        self.x2_1 = ConvBlock(f2 + f3, f2)
        self.x3_1 = ConvBlock(f3 + f4, f3)
        self.x4_1 = ConvBlock(f4 + f5, f4)

        self.x0_2 = ConvBlock(f0 * 2 + f1, f0)
        self.x1_2 = ConvBlock(f1 * 2 + f2, f1)
        self.x2_2 = ConvBlock(f2 * 2 + f3, f2)
        self.x3_2 = ConvBlock(f3 * 2 + f4, f3)

        self.x0_3 = ConvBlock(f0 * 3 + f1, f0)
        self.x1_3 = ConvBlock(f1 * 3 + f2, f1)
        self.x2_3 = ConvBlock(f2 * 3 + f3, f2)

        self.x0_4 = ConvBlock(f0 * 4 + f1, f0)
        self.x1_4 = ConvBlock(f1 * 4 + f2, f1)

        self.x0_5 = ConvBlock(f0 * 5 + f1, f0)

        self.outc = nn.Conv2d(f0, n_classes_out, kernel_size=1)
        self.final_activation = nn.Tanh()

    def forward(self, x):
        x0_0 = self.x0_0(x)
        x1_0 = self.x1_0(self.pool(x0_0))
        x2_0 = self.x2_0(self.pool(x1_0))
        x3_0 = self.x3_0(self.pool(x2_0))
        x4_0 = self.x4_0(self.pool(x3_0))
        x5_0 = self.x5_0(self.pool(x4_0))

        x4_1 = self.x4_1(torch.cat([x4_0, self.up(x5_0)], dim=1))
        x3_1 = self.x3_1(torch.cat([x3_0, self.up(x4_0)], dim=1))
        x2_1 = self.x2_1(torch.cat([x2_0, self.up(x3_0)], dim=1))
        x1_1 = self.x1_1(torch.cat([x1_0, self.up(x2_0)], dim=1))
        x0_1 = self.x0_1(torch.cat([x0_0, self.up(x1_0)], dim=1))

        x3_2 = self.x3_2(torch.cat([x3_0, x3_1, self.up(x4_1)], dim=1))
        x2_2 = self.x2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], dim=1))
        x1_2 = self.x1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], dim=1))
        x0_2 = self.x0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], dim=1))

        x2_3 = self.x2_3(torch.cat([x2_0, x2_1, x2_2, self.up(x3_2)], dim=1))
        x1_3 = self.x1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], dim=1))
        x0_3 = self.x0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], dim=1))

        x1_4 = self.x1_4(torch.cat([x1_0, x1_1, x1_2, x1_3, self.up(x2_3)], dim=1))
        x0_4 = self.x0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], dim=1))

        x0_5 = self.x0_5(torch.cat([x0_0, x0_1, x0_2, x0_3, x0_4, self.up(x1_4)], dim=1))
        return self.final_activation(self.outc(x0_5))
