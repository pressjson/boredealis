#!/usr/bin/env python3


import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F  # Needed for GELU in TransformerEncoderLayer
import torchvision.transforms.functional as FU
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import noise
import numpy
import time
import random
import math  # For isnan check

if not os.path.exists("local_settings.py"):
    print("Warning: local settings not found. Using default settings.")
    import settings
else:
    import local_settings as settings


class ImageDataset(Dataset):
    def __init__(
        self,
        root_dir=os.path.join("data", "images"),
        clear_images=None,
        cloud_images=None,
        clear_transform=None,
        cloud_transform=None,
    ):
        self.cloud_images = cloud_images
        self.clear_images = clear_images
        self.clear_transform = clear_transform
        self.cloud_transform = cloud_transform
        self.root_dir = root_dir

    def __len__(self):
        return len(self.clear_images)

    def __getitem__(self, i):
        lq_image = Image.open(self.cloud_images[i]).convert("RGB")
        hq_image = Image.open(self.clear_images[i]).convert("RGB")
        if self.clear_transform:
            lq_image = self.cloud_transform(lq_image)
            hq_image = self.clear_transform(hq_image)
        return lq_image, hq_image


# google gemini pro 2.5 advanced code
# edited by me
# originally from COSC200, adapted to fit this use case.


def generate_perlin_noise_image(
    size,
    lower_bound,
    upper_bound,
    scale=50.0,
    octaves=4,
    persistence=0.5,
    lacunarity=2.0,
):
    world = numpy.zeros((size, size))
    noise_base = random.randint(0, 10000)
    for x in range(size):
        for y in range(size):
            world[x][y] = noise.pnoise2(
                x / scale,
                y / scale,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
                repeatx=size / scale,
                repeaty=size / scale,
                base=noise_base,
            )

    min_val = numpy.min(world)
    max_val = numpy.max(world)

    normalized_world = (world - min_val) / (max_val - min_val)

    r_low, g_low, b_low = lower_bound
    r_up, g_up, b_up = upper_bound

    pixels = numpy.zeros((size, size, 3), dtype=numpy.uint8)

    for x in range(size):
        for y in range(size):
            noise_val_norm = normalized_world[x][y]

            # Linearly interpolate each color channel
            r = int(r_low + abs(r_up - r_low) * noise_val_norm)
            g = int(g_low + abs(g_up - g_low) * noise_val_norm)
            b = int(b_low + abs(b_up - b_low) * noise_val_norm)

            # Clamp values just in case of floating point inaccuracies, though not strictly necessary
            # with proper normalization and int conversion.
            r = max(0, min(255, r))
            g = max(0, min(255, g))
            b = max(0, min(255, b))

            pixels[x, y] = (r, g, b)

    return Image.fromarray(pixels)


class CombineWithClouds:
    def __init__(self, output_size):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.output_size = output_size

    def __call__(self, main_tensor):
        # Takes a pipeline tensor, turns it *back* into an image, combines it with a perlin noise, then back into a tensor
        # There has to be a better way, but this works, I guess
        main_image = FU.to_pil_image(main_tensor)

        # @TODO: when there is a file of cloud hex values, randomly pick two hex values and make them tuples
        lower_bound = (0, 0, 0)
        upper_bound = (255, 255, 255)

        alpha_lower_bound = 0.5
        alpha_upper_bound = 0.7

        fake_clouds = generate_perlin_noise_image(
            settings.IMAGE_SIZE[0], lower_bound=lower_bound, upper_bound=upper_bound
        )

        combined_image = Image.blend(
            main_image,
            fake_clouds,
            random.uniform(alpha_lower_bound, alpha_upper_bound),
        )

        return FU.pil_to_tensor(combined_image).float().div(255).to(self.device)


# google gemini


class RandomApplyTransforms:
    # His name is Randy
    def __init__(self, output_size, random_threshold, noise_weight):
        self.output_size = output_size
        self.random_threshold = random_threshold
        self.noise_weight = noise_weight
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __call__(self, sample):
        if random.uniform(0, 1) > self.random_threshold:
            # do nothing
            return sample

        if not torch.is_tensor(sample):
            print(
                f"Warning: sample is not a tensor. Things may go wrong\nSample type: {type(sample)}"
            )

        cloud = CombineWithClouds(settings.IMAGE_SIZE)
        sample = cloud(sample)
        sample = sample.add(
            torch.rand(settings.IMAGE_SIZE).mul(self.noise_weight).to(self.device)
        )

        return sample


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
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, skip_channels):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels, in_channels // 2)
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


class DeepUNetPyTorch(nn.Module):
    def __init__(self, n_channels_in, n_classes_out, start_filters=64):
        super(DeepUNetPyTorch, self).__init__()
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
        # The 'in_channels' for Up is the number of channels from the layer below (e.g., bottleneck)
        # The 'skip_channels' is the number of channels from the corresponding encoder layer
        # The 'out_channels' is the target number of channels for this decoder stage
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

        # Determine final activation
        if n_classes_out == 1:
            self.final_activation = nn.Sigmoid()
        elif n_classes_out > 1:
            self.final_activation = nn.Softmax(
                dim=1
            )  # Apply softmax over channel dimension
        else:  # Should not happen with positive n_classes_out
            self.final_activation = None  # Linear activation

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


def train_model(
    IMG_CHANNELS_IN=3,
    NUM_CLASSES_OUT=1,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = DeepUNetPyTorch(
        n_channels_in=IMG_CHANNELS_IN, n_classes_out=NUM_CLASSES_OUT, start_filters=64
    ).to(device)

    # dataloaders

    clear_dataloader = transforms.Compose(
        [
            transforms.Resize(settings.IMAGE_SIZE),
            transforms.ToTensor(),
        ]
    )
    cloud_dataloader = transforms.Compose(
        [
            transforms.Resize(settings.IMAGE_SIZE),
            RandomApplyTransforms(
                settings.IMAGE_SIZE,
                settings.RANDOM_APPLY_THRESHOLD,
                settings.NOISE_STRENGTH,
            ),
        ]
    )
