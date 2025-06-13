#!/usr/bin/env python3

import os
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F  # Needed for GELU in TransformerEncoderLayer
import torchvision.transforms.functional as TF
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFilter
import noise
import numpy
import time
import random

import cloud_colors

if not os.path.exists("local_settings.py"):
    print("Warning: local settings not found. Using default settings.")
    import settings
else:
    import local_settings as settings


class ImageDataset(Dataset):
    def __init__(
        self,
        images=None,
        data_dir=None,
        clear_transform=None,
        cloud_transform=None,
    ):
        self.images = images
        self.data_dir = data_dir
        self.clear_transform = clear_transform
        self.cloud_transform = cloud_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        image = Image.open(os.path.join(self.data_dir, self.images[i])).convert("RGB")
        if self.clear_transform:
            cloud_image = self.cloud_transform(image)
            clear_image = self.clear_transform(image)
        return cloud_image, clear_image


# google gemini pro 2.5 advanced code
# edited by me


def hex_to_rgb(hex):
    """Takes a hex string and returns the corresponding RGB tuple.

    From https://www.30secondsofcode.org/python/s/hex-to-rgb/

    Args:
        hex (str): a hex tuple, formatted AABBCC.

    Returns:
        int tuple, formatted (AA, BB, CC) but converted to the corresponding integer value.
    """
    return tuple(int(hex[i : i + 2], 16) for i in (0, 2, 4))


def generate_perlin_noise_image(
    size,
    lower_bound,
    upper_bound,
    scale=random.uniform(30.0, 500.0),
    octaves=random.randint(2, 8),
    persistence=0.5,
    lacunarity=2.0,
):
    world = numpy.zeros((size, size))
    noise_base = random.randint(1, 10000)
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


def crop_to_center_circle(pil_image: Image.Image) -> Image.Image:
    """
    Takes a PIL image of 608x608 and keeps only a center circle
    with a radius of 250 pixels. The area outside the circle
    will be made transparent.

    Args:
        pil_image (PIL.Image.Image): The input image, must be 608x608.

    Returns:
        PIL.Image.Image: A new image with the circular crop applied.
                         The image will be in RGBA format.
    """

    radius = 250
    vertical_offset = 15
    width, height = pil_image.size  # Should be 608, 608

    # Ensure the image has an alpha channel for transparency
    img_rgba = pil_image.convert("RGBA")

    # Create a mask:
    # Start with a completely black (transparent) mask
    mask = Image.new("L", (width, height), 0)  # 'L' mode for grayscale mask
    draw = ImageDraw.Draw(mask)

    # Calculate the bounding box for the circle
    # The center of the image is (width/2, height/2)
    center_x = width // 2
    center_y = height // 2

    # Bounding box coordinates (left, top, right, bottom)
    left = center_x - radius
    top = center_y - radius - vertical_offset
    right = center_x + radius
    bottom = center_y + radius - vertical_offset

    # Draw a white (opaque) circle on the black mask
    draw.ellipse((left, top, right, bottom), fill=255)  # 255 is white in 'L' mode

    # Apply the mask to the image
    # The 'mask' argument in putalpha uses the 'L' mode mask
    # to set the alpha channel of the RGBA image.
    img_rgba.putalpha(mask)

    return img_rgba


class CombineWithClouds:
    def __init__(self, output_size):
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.output_size = output_size

    def __call__(self, main_image):
        main_image = main_image.convert("RGBA")
        # Using static grayscale clouds
        # lower = random.randint(0, 130)
        # upper = random.randint(150, 255)

        # Using a file of cloud values
        # lower_bound = hex_to_rgb(random.choice(cloud_colors.LOWER_BOUNDS))
        # upper_bound = hex_to_rgb(random.choice(cloud_colors.UPPER_BOUNDS))
        # print(f"Lower bound = {lower_bound}")
        # print(f"Upper bound = {upper_bound}")
        # print(f"Main image: {main_image}")

        # Using the pixels in the image plus a grey
        # relies on randomness to eventually pick a valid pixel
        lower_bound = [0, 0, 0]
        upper_bound = [255, 255, 255]
        sampling_image = crop_to_center_circle(main_image)
        # Pick lower bound
        while True:
            random_coordinates = (
                random.randint(0, main_image.size[0] - 1),
                random.randint(0, main_image.size[0] - 1),
            )
            lower_bound = list(sampling_image.getpixel(random_coordinates))
            if not lower_bound[3] == 0:
                cloud_dimness = random.randint(20, 100)
                for i in range(len(lower_bound)):
                    lower_bound[i] = (
                        lower_bound[i] - cloud_dimness
                        if lower_bound[i] > cloud_dimness
                        else 0
                    )
                # print(lower_bound)
                break
        # Pick upper bound
        while True:
            random_coordinates = (
                random.randint(0, main_image.size[0] - 1),
                random.randint(0, main_image.size[0] - 1),
            )
            upper_bound = list(sampling_image.getpixel(random_coordinates))
            if not upper_bound[3] == 0:
                upper_bound[3] = 0
                # print(f"Input list: {upper_bound}")
                cloud_brightness = random.randint(50, 150)
                for i in range(len(upper_bound)):
                    # print(upper_bound[i])
                    upper_bound[i] = (
                        upper_bound[i] + cloud_brightness
                        if upper_bound[i] < (255 - cloud_brightness)
                        else 255
                    )
                    # print(upper_bound[i])
                break

        lower_bound = tuple(lower_bound[:3])
        # print(upper_bound)
        upper_bound = tuple(upper_bound[:3])
        # print(f"Output tuple: {upper_bound}")

        alpha_lower_bound = settings.ALPHA_LOWER_BOUND
        alpha_upper_bound = settings.ALPHA_UPPER_BOUND

        fake_clouds = generate_perlin_noise_image(
            settings.IMAGE_SIZE[0], lower_bound=lower_bound, upper_bound=upper_bound
        )
        # print(f"Fake clouds: {fake_clouds}")

        cropped_clouds = crop_to_center_circle(fake_clouds)
        # print(f"Cropped clouds: {cropped_clouds}")

        # combined_image = Image.blend(
        #     main_image,
        #     cropped_clouds,
        #     random.uniform(alpha_lower_bound, alpha_upper_bound),
        # )
        r, g, b, alpha = cropped_clouds.split()
        blend_strength = random.uniform(alpha_lower_bound, alpha_upper_bound)

        final_alpha = alpha.point(lambda p: int(p * blend_strength))
        cloud_layer = Image.merge("RGBA", (r, g, b, final_alpha))

        combined_image = Image.alpha_composite(main_image, cloud_layer)
        blurred_image = combined_image.filter(
            ImageFilter.GaussianBlur(radius=random.randint(0, 4))
        )
        blurred_center = crop_to_center_circle(blurred_image)

        combined_image = Image.alpha_composite(main_image, blurred_center)

        return combined_image.convert("RGB")


# google gemini


class RandomApplyTransforms:
    # His name is Randy
    def __init__(self, output_size, random_threshold, noise_weight):
        self.output_size = output_size
        self.random_threshold = random_threshold
        self.noise_weight = noise_weight
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __call__(self, sample):
        # for debugging why my computer crashes
        # return FU.to_tensor(sample)

        if random.uniform(0, 1) > self.random_threshold:
            # do nothing
            return TF.to_tensor(sample)

        cloud = CombineWithClouds(self.output_size)

        sample = cloud(sample)
        sample = TF.to_tensor(sample)
        noise = torch.rand_like(sample) * self.noise_weight
        sample = sample + noise
        sample = torch.clamp(sample, 0.0, 1.0)

        return sample


# VGG loss, implemented on https://github.com/crowsonkb/vgg_loss?tab=readme-ov-file
# @TODO: refactor this to a new file


class Lambda(nn.Module):
    """Wraps a callable in an :class:`nn.Module` without registering it."""

    def __init__(self, func):
        super().__init__()
        object.__setattr__(self, "forward", func)

    def extra_repr(self):
        return getattr(self.forward, "__name__", type(self.forward).__name__) + "()"


class WeightedLoss(nn.ModuleList):
    """A weighted combination of multiple loss functions."""

    def __init__(self, losses, weights, verbose=False):
        super().__init__()
        for loss in losses:
            self.append(loss if isinstance(loss, nn.Module) else Lambda(loss))
        self.weights = weights
        self.verbose = verbose

    def _print_losses(self, losses):
        for i, loss in enumerate(losses):
            print(f"({i}) {type(self[i]).__name__}: {loss.item()}")

    def forward(self, *args, **kwargs):
        losses = []
        for loss, weight in zip(self, self.weights):
            losses.append(loss(*args, **kwargs) * weight)
        if self.verbose:
            self._print_losses(losses)
        return sum(losses)


class TVLoss(nn.Module):
    """Total variation loss (Lp penalty on image gradient magnitude).

    The input must be 4D. If a target (second parameter) is passed in, it is
    ignored.

    ``p=1`` yields the vectorial total variation norm. It is a generalization
    of the originally proposed (isotropic) 2D total variation norm (see
    (see https://en.wikipedia.org/wiki/Total_variation_denoising) for color
    images. On images with a single channel it is equal to the 2D TV norm.

    ``p=2`` yields a variant that is often used for smoothing out noise in
    reconstructions of images from neural network feature maps (see Mahendran
    and Vevaldi, "Understanding Deep Image Representations by Inverting
    Them", https://arxiv.org/abs/1412.0035)

    :attr:`reduction` can be set to ``'mean'``, ``'sum'``, or ``'none'``
    similarly to the loss functions in :mod:`torch.nn`. The default is
    ``'mean'``.
    """

    def __init__(self, p, reduction="mean", eps=1e-8):
        super().__init__()
        if p not in {1, 2}:
            raise ValueError("p must be 1 or 2")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        self.p = p
        self.reduction = reduction
        self.eps = eps

    def forward(self, input, target=None):
        input = F.pad(input, (0, 1, 0, 1), "replicate")
        x_diff = input[..., :-1, :-1] - input[..., :-1, 1:]
        y_diff = input[..., :-1, :-1] - input[..., 1:, :-1]
        diff = x_diff**2 + y_diff**2
        if self.p == 1:
            diff = (diff + self.eps).mean(dim=1, keepdims=True).sqrt()
        if self.reduction == "mean":
            return diff.mean()
        if self.reduction == "sum":
            return diff.sum()
        return diff


class VGGLoss(nn.Module):
    """Computes the VGG perceptual loss between two batches of images.

    The input and target must be 4D tensors with three channels
    ``(B, 3, H, W)`` and must have equivalent shapes. Pixel values should be
    normalized to the range 0–1.

    The VGG perceptual loss is the mean squared difference between the features
    computed for the input and target at layer :attr:`layer` (default 8, or
    ``relu2_2``) of the pretrained model specified by :attr:`model` (either
    ``'vgg16'`` (default) or ``'vgg19'``).

    If :attr:`shift` is nonzero, a random shift of at most :attr:`shift`
    pixels in both height and width will be applied to all images in the input
    and target. The shift will only be applied when the loss function is in
    training mode, and will not be applied if a precomputed feature map is
    supplied as the target.

    :attr:`reduction` can be set to ``'mean'``, ``'sum'``, or ``'none'``
    similarly to the loss functions in :mod:`torch.nn`. The default is
    ``'mean'``.

    :meth:`get_features()` may be used to precompute the features for the
    target, to speed up the case where inputs are compared against the same
    target over and over. To use the precomputed features, pass them in as
    :attr:`target` and set :attr:`target_is_features` to :code:`True`.

    Instances of :class:`VGGLoss` must be manually converted to the same
    device and dtype as their inputs.
    """

    models = {"vgg16": models.vgg16, "vgg19": models.vgg19}

    def __init__(self, model="vgg16", layer=8, shift=0, reduction="mean"):
        super().__init__()
        self.shift = shift
        self.reduction = reduction
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.model = self.models[model](pretrained=True).features[: layer + 1]
        self.model.eval()
        self.model.requires_grad_(False)

    def get_features(self, input):
        return self.model(self.normalize(input))

    def train(self, mode=True):
        self.training = mode

    def forward(self, input, target, target_is_features=False):
        if target_is_features:
            input_feats = self.get_features(input)
            target_feats = target
        else:
            sep = input.shape[0]
            batch = torch.cat([input, target])
            if self.shift and self.training:
                padded = F.pad(batch, [self.shift] * 4, mode="replicate")
                batch = transforms.RandomCrop(batch.shape[2:])(padded)
            feats = self.get_features(batch)
            input_feats, target_feats = feats[:sep], feats[sep:]
        return F.mse_loss(input_feats, target_feats, reduction=self.reduction)


# Model definition


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
        self.final_activation = nn.Tanh()

        # # Determine final activation
        # if n_classes_out == 1:
        #     self.final_activation = nn.Sigmoid()
        # elif n_classes_out > 1:
        #     self.final_activation = nn.Softmax(
        #         dim=1
        #     )  # Apply softmax over channel dimension
        # else:  # Should not happen with positive n_classes_out
        #     self.final_activation = None  # Linear activation

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
    NUM_CLASSES_OUT=3,
    START_FILTERS=settings.START_FILTERS,
    DATA_DIR=os.path.join("data", "images"),
    num_epochs=settings.NUM_EPOCHS,
    previous_model_path=None,
    debug=False,
):
    """Training loop for training a model.

    Just run python network.py. Please.
    Config this with settings.py or local_settings.py.
    All of the args should be self explanatory.

    Args:
        IMG_CHANNELS_IN (int)
        NUM_CLASSES_OUT (int)
        START_FILTERS (int): configured in settings
        DATA_DIR (str)
        num_epochs (int): configured in settings
        previous_model_path (str)
        debug (bool): exits the loop early for displaying a sample of target and training data
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"HIP version (ROCm): {torch.version.hip}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"Using device: {device}")

    # return 1

    # Dataloaders

    clear_transform = transforms.Compose(
        [
            transforms.Resize(settings.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    cloud_transform = transforms.Compose(
        [
            transforms.Resize(settings.IMAGE_SIZE),
            RandomApplyTransforms(
                settings.IMAGE_SIZE,
                settings.RANDOM_APPLY_THRESHOLD,
                settings.NOISE_STRENGTH,
            ),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    # Making the datasets from data/images by default

    images = []

    i = settings.NUM_IMAGES
    for file in os.listdir(DATA_DIR):
        images.append(file)
        if i != -1:
            i = i - 1
            if i == 0:
                break

    print(f"Found {len(images)} images")

    if len(images) == 0:
        raise ValueError(
            f"No images found in {images}. Check file naming and structure."
        )
    print(
        f"Using {int(settings.VALUE_SPLIT * len(images))} for training and {len(images) - int(settings.VALUE_SPLIT * len(images))} for validation"
    )
    random.shuffle(images)
    train = images[: int(settings.VALUE_SPLIT * len(images))]
    valid = images[int(settings.VALUE_SPLIT * len(images)) :]

    train_dataset = ImageDataset(
        train,
        data_dir=DATA_DIR,
        clear_transform=clear_transform,
        cloud_transform=cloud_transform,
    )

    valid_dataset = ImageDataset(
        valid,
        data_dir=DATA_DIR,
        clear_transform=clear_transform,
        cloud_transform=cloud_transform,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=settings.BATCH_SIZE,
        num_workers=settings.NUM_WORKERS,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=settings.BATCH_SIZE,
        num_workers=settings.NUM_WORKERS,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
    )

    if debug == True:
        print("Visualizing a sample from training data...")
        sample_inputs, sample_targets = next(iter(train_dataloader))
        # output_tensor * 0.5 + 0.5
        show_tensor_image(
            (sample_inputs[0] * 0.5 + 0.5).cpu()
        )  # Show first cloudy image in batch
        show_tensor_image(
            (sample_targets[0] * 0.5 + 0.5).cpu()
        )  # Show first clear image in batch
        return -1

    if not os.path.exists(settings.MODEL_SAVE_PATH):
        os.mkdir(settings.MODEL_SAVE_PATH)

    scaler = None
    if settings.USE_AMP and device.type == "cuda":
        scaler = torch.amp.GradScaler()
        print("Using Automatic Mixed Precision (AMP).")

    start_epoch = 0

    model = DeepUNet(
        n_channels_in=IMG_CHANNELS_IN,
        n_classes_out=NUM_CLASSES_OUT,
        start_filters=START_FILTERS,
    )
    if previous_model_path == None:
        print(
            f"Initialized DeepUNet with {IMG_CHANNELS_IN} channels in, {NUM_CLASSES_OUT} classes out, and {START_FILTERS} start filters."
        )
    else:
        if not os.path.exists(previous_model_path):
            raise ValueError(
                f"Error: {previous_model_path} is not a valid path to a previous model."
            )
            return -1
        checkpoint = torch.load(previous_model_path)
        start_filters = checkpoint["start_filters"]
        in_channels = checkpoint["in_channels"]
        out_channels = checkpoint["out_channels"]
        start_epoch = checkpoint["epoch"]
        model = DeepUNet(
            n_channels_in=in_channels,
            n_classes_out=out_channels,
            start_filters=start_filters,
        )
        loaded_state_dict = checkpoint["model_state_dict"]
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        is_data_parallel = False
        for k, v in loaded_state_dict.items():
            if k.startswith("module."):
                is_data_parallel = True
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            else:
                new_state_dict[k] = v  # Non-DataParallel checkpoint or already stripped

        if is_data_parallel:
            print(
                "Checkpoint was saved from a DataParallel model. Stripping 'module.' prefix."
            )

        model.load_state_dict(new_state_dict)

        # model.load_state_dict(checkpoint["model_state_dict"])
        print(
            f"Loading model from {previous_model_path} with {in_channels} channels in, {out_channels} classes out, and {start_filters} start filters."
        )

    if torch.cuda.is_available() and torch.cuda.device_count() > 2:
        print(
            f"Wrapping model with nn.DataParallel for {torch.cuda.device_count()} GPUs."
        )
        if settings.USE_DEVICE_IDS:
            print(f"Using only devices {settings.DEVICE_IDS}")
            model = model.to(f"cuda:{int(settings.DEVICE_IDS[0])}")
            model = nn.DataParallel(
                model,
                device_ids=settings.DEVICE_IDS,
                output_device=settings.DEVICE_IDS[0],
            )
        else:
            model = model.to(device)
            model = nn.DataParallel(model)
    else:
        model = model.to(device)

    criterion = nn.L1Loss()
    vgg_loss_crit = VGGLoss().to(
        f"cuda:{settings.VGG_DEVICE_ID}" if settings.USE_VGG_DEVICE else device
    )
    L1_WEIGHT = 1.0
    VGG_WEIGHT = 0.1

    optimizer = optim.Adam(model.parameters(), lr=settings.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=settings.STEP_SIZE,
        factor=settings.GAMMA,
        mode="min",
    )
    best_val_loss = float("inf")

    # training loop
    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        print(f"\n--- Epoch {epoch}/{num_epochs} [Train] ---")
        batch_start_time = time.time()

        for i, (inputs, targets) in enumerate(train_dataloader):
            inputs, targets = inputs.to(
                f"cuda:{settings.DEVICE_IDS[0]}" if settings.USE_DEVICE_IDS else device
            ), targets.to(
                f"cuda:{settings.DEVICE_IDS[0]}" if settings.USE_DEVICE_IDS else device
            )

            # show_tensor_image(inputs[0])
            # show_tensor_image(targets[0])
            # break
            optimizer.zero_grad()

            if scaler:  # AMP
                with torch.amp.autocast(
                    device_type="cuda" if torch.cuda.is_available() else "cpu"
                ):
                    outputs = model(inputs)
                    l1_loss = criterion(outputs, targets)
                    outputs = (outputs + 1.0) / 2.0
                    targets = (targets + 1.0) / 2.0
                    vgg_loss = vgg_loss_crit(
                        outputs,
                        vgg_loss_crit.get_features(targets),
                        target_is_features=True,
                    )
                loss = l1_loss * L1_WEIGHT + vgg_loss * VGG_WEIGHT
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:  # No AMP
                outputs = model(inputs)
                l1_loss = criterion(outputs, targets)
                outputs = (outputs + 1.0) / 2.0
                targets = (targets + 1.0) / 2.0
                vgg_loss = vgg_loss_crit(
                    outputs,
                    vgg_loss_crit.get_features(targets),
                    target_is_features=True,
                )
                loss = l1_loss * L1_WEIGHT + vgg_loss * VGG_WEIGHT
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 20 == 0 or (i + 1) == len(train_dataloader):
                batch_time = time.time() - batch_start_time
                print(
                    f"  Batch {i+1}/{len(train_dataloader)} | Train Loss: {loss.item():.4f} | Time: {batch_time:.2f}s"
                )
            if (
                (i + 1) >= settings.MAX_EPOCH_TRAIN_SIZE
                and settings.MAX_EPOCH_TRAIN_SIZE != -1
            ):
                break

        epoch_train_loss = running_loss / (
            len(train_dataset)
            if settings.MAX_EPOCH_TRAIN_SIZE == -1
            else settings.MAX_EPOCH_TRAIN_SIZE
        )
        print(f"Epoch {epoch+1} [Train] Avg Loss: {epoch_train_loss:.4f}")

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in valid_dataloader:
                inputs, targets = inputs.to(
                    f"cuda:{settings.DEVICE_IDS[0]}"
                    if settings.USE_DEVICE_IDS
                    else device
                ), targets.to(
                    f"cuda:{settings.DEVICE_IDS[0]}"
                    if settings.USE_DEVICE_IDS
                    else device
                )
                # show_tensor_image(inputs[0])
                # show_tensor_image(targets[0])
                # break

                if scaler:  # AMP for validation
                    with torch.amp.autocast(
                        device_type="cuda" if torch.cuda.is_available() else "cpu"
                    ):
                        outputs = model(inputs)
                        l1_loss = criterion(outputs, targets)
                        outputs = (outputs + 1.0) / 2.0
                        targets = (targets + 1.0) / 2.0
                        vgg_loss = vgg_loss_crit(
                            outputs,
                            vgg_loss_crit.get_features(targets),
                            target_is_features=True,
                        )
                        loss = l1_loss * L1_WEIGHT + vgg_loss * VGG_WEIGHT
                else:
                    outputs = model(inputs)
                    l1_loss = criterion(outputs, targets)
                    outputs = (outputs + 1.0) / 2.0
                    targets = (targets + 1.0) / 2.0
                    vgg_loss = vgg_loss_crit(
                        outputs,
                        vgg_loss_crit.get_features(targets),
                        target_is_features=True,
                    )
                    loss = l1_loss * L1_WEIGHT + vgg_loss * VGG_WEIGHT

                running_val_loss += loss.item()

        epoch_val_loss = running_val_loss / len(valid_dataset)
        print(f"Epoch {epoch+1} [Val]   Avg Loss: {epoch_val_loss:.4f}")

        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch Duration: {epoch_duration:.2f}s")

        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(epoch_val_loss)
        elif scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Current Learning Rate: {current_lr}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            model_name = "checkpoint_best.pth"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "start_filters": START_FILTERS,
                    "in_channels": IMG_CHANNELS_IN,
                    "out_channels": NUM_CLASSES_OUT,
                    "epoch": epoch,
                },
                os.path.join(settings.MODEL_SAVE_PATH, model_name),
            )
            print(
                f"Model improved. Saved to {settings.MODEL_SAVE_PATH} (Val Loss: {best_val_loss:.4f})"
            )

        if epoch % settings.EPOCH_SAVE_INTERVAL == 0:

            model_name = f"checkpoint_epoch_{epoch}.pth"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "start_filters": START_FILTERS,
                    "in_channels": IMG_CHANNELS_IN,
                    "out_channels": NUM_CLASSES_OUT,
                    "epoch": epoch,
                },
                os.path.join(settings.MODEL_SAVE_PATH, model_name),
            )
            print(
                f"Reached a checkpoint. Saved to {settings.MODEL_SAVE_PATH} (Val Loss: {best_val_loss:.4f})"
            )

        # This actually worked on my RX 6800 XT, although it was because it cooled down the GPU,
        # not because of VRAM usage
        # print("Sleeping, hopefully to prevent vram overusage")
        # time.sleep(1)

    print("\n--- Training Finished ---")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Best model saved at: {settings.MODEL_SAVE_PATH}")


def show_tensor_image(tensor):
    image = tensor.detach().cpu()
    image = TF.to_pil_image(image)
    image.show()


if __name__ == "__main__":
    train_model(
        DATA_DIR=os.path.join("data", "images"),
        # previous_model_path=os.path.join("models", "64_checkpoint_best.pth"),
        debug=False,
    )
