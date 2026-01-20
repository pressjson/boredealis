"""U-Net Architecture."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv."""

    def __init__(self, in_channels, out_channels, skip_channels):
        super().__init__()
        # for older models (ISVC and before), uncomment this portion and comment the other portion
        # if bilinear, use the normal convolutions to reduce the number of channels
        # self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # self.conv = DoubleConv(
        #     in_channels + skip_channels, out_channels, in_channels // 2
        # )
        # for newer models (randiv and after), uncomment this portion and comment the other portion
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x1, x2):
        # x1: from previous layer in decoder
        # x2: skip connection from encoder
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # Pad x1 to match x2's dimensions if necessary
        # (padding_left, padding_right, padding_top, padding_bottom)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class DeepUNet(nn.Module):
    def __init__(self, n_channels_in, n_classes_out, start_filters=64):
        super(DeepUNet, self).__init__()
        self.n_channels_in = n_channels_in
        self.n_classes_out = n_classes_out
        self.start_filters = start_filters

        # Encoder
        self.inc = DoubleConv(n_channels_in, start_filters)
        self.down1 = Down(start_filters, start_filters * 2)  # 608->304
        self.down2 = Down(start_filters * 2, start_filters * 4)  # 304->152
        self.down3 = Down(start_filters * 4, start_filters * 8)  # 152->76
        self.down4 = Down(start_filters * 8, start_filters * 16)  # 76->38
        self.down5 = Down(
            start_filters * 16, start_filters * 32
        )  # 38->19 (Bottleneck input)

        # Decoder
        self.up1 = Up(
            start_filters * 32, start_filters * 16, skip_channels=start_filters * 16
        )  # 19->38
        self.up2 = Up(
            start_filters * 16, start_filters * 8, skip_channels=start_filters * 8
        )  # 38->76
        self.up3 = Up(
            start_filters * 8, start_filters * 4, skip_channels=start_filters * 4
        )  # 76->152
        self.up4 = Up(
            start_filters * 4, start_filters * 2, skip_channels=start_filters * 2
        )  # 152->304
        self.up5 = Up(
            start_filters * 2, start_filters, skip_channels=start_filters
        )  # 304->608

        self.outc = OutConv(start_filters, n_classes_out)
        self.final_activation = nn.Tanh()

    def forward(self, x):
        # Encoder
        s1 = self.inc(x)  # 608x608, sf
        s2 = self.down1(s1)  # 304x304, sf*2
        s3 = self.down2(s2)  # 152x152, sf*4
        s4 = self.down3(s3)  # 76x76,   sf*8
        s5 = self.down4(s4)  # 38x38,   sf*16
        bottleneck = self.down5(s5)  # 19x19,   sf*32

        # Decoder
        d1 = self.up1(bottleneck, s5)  # 38x38, sf*16
        d2 = self.up2(d1, s4)  # 76x76, sf*8
        d3 = self.up3(d2, s3)  # 152x152, sf*4
        d4 = self.up4(d3, s2)  # 304x304, sf*2
        d5 = self.up5(d4, s1)  # 608x608, sf

        logits = self.outc(d5)

        if self.final_activation:
            return self.final_activation(logits)
        return logits
