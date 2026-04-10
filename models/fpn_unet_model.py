"""FPN U-Net style image-to-image model."""

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class FPNUNet(nn.Module):
    def __init__(self, n_channels_in, n_classes_out, start_filters=160):
        super().__init__()
        sf = start_filters
        self.pool = nn.MaxPool2d(2)

        self.e1 = ConvBlock(n_channels_in, sf)
        self.e2 = ConvBlock(sf, sf * 2)
        self.e3 = ConvBlock(sf * 2, sf * 4)
        self.e4 = ConvBlock(sf * 4, sf * 8)
        self.e5 = ConvBlock(sf * 8, sf * 16)

        pyramid_channels = sf * 2
        self.l5 = nn.Conv2d(sf * 16, pyramid_channels, kernel_size=1)
        self.l4 = nn.Conv2d(sf * 8, pyramid_channels, kernel_size=1)
        self.l3 = nn.Conv2d(sf * 4, pyramid_channels, kernel_size=1)
        self.l2 = nn.Conv2d(sf * 2, pyramid_channels, kernel_size=1)
        self.l1 = nn.Conv2d(sf, pyramid_channels, kernel_size=1)

        self.s5 = ConvBlock(pyramid_channels, pyramid_channels)
        self.s4 = ConvBlock(pyramid_channels, pyramid_channels)
        self.s3 = ConvBlock(pyramid_channels, pyramid_channels)
        self.s2 = ConvBlock(pyramid_channels, pyramid_channels)
        self.s1 = ConvBlock(pyramid_channels, pyramid_channels)

        self.head = nn.Sequential(
            nn.Conv2d(pyramid_channels * 5, sf * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(sf * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(sf * 4, n_classes_out, kernel_size=1),
        )
        self.final_activation = nn.Tanh()

    def _upsample_to(self, x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        input_size = x.shape[-2:]
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        e5 = self.e5(self.pool(e4))

        p5 = self.s5(self.l5(e5))
        p4 = self.s4(self.l4(e4) + self._upsample_to(p5, e4))
        p3 = self.s3(self.l3(e3) + self._upsample_to(p4, e3))
        p2 = self.s2(self.l2(e2) + self._upsample_to(p3, e2))
        p1 = self.s1(self.l1(e1) + self._upsample_to(p2, e1))

        fused = torch.cat(
            [
                self._upsample_to(p5, e1),
                self._upsample_to(p4, e1),
                self._upsample_to(p3, e1),
                self._upsample_to(p2, e1),
                p1,
            ],
            dim=1,
        )
        output = self.head(fused)
        output = F.interpolate(output, size=input_size, mode="bilinear", align_corners=False)
        return self.final_activation(output)
