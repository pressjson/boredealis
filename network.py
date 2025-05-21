#!/usr/bin/env python3

import os
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F  # Needed for GELU in TransformerEncoderLayer
import torchvision.transforms.functional as TF
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import noise
import numpy
import time
import random

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

    def __call__(self, main_image):
        # Takes a pipeline tensor, turns it *back* into an image, combines it with a perlin noise, then back into a tensor
        # There has to be a better way, but this works, I guess
        # if type(main_image) != Image:
        #     main_image = FU.to_pil_image(main_image)

        # @TODO: when there is a file of cloud hex values, randomly pick two hex values and make them tuples
        lower_bound = (0, 0, 0)
        upper_bound = (255, 255, 255)

        alpha_lower_bound = settings.ALPHA_LOWER_BOUND
        alpha_upper_bound = settings.ALPHA_UPPER_BOUND

        fake_clouds = generate_perlin_noise_image(
            settings.IMAGE_SIZE[0], lower_bound=lower_bound, upper_bound=upper_bound
        )

        combined_image = Image.blend(
            main_image,
            fake_clouds,
            random.uniform(alpha_lower_bound, alpha_upper_bound),
        )

        return combined_image


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
    NUM_CLASSES_OUT=3,
    START_FILTERS=settings.START_FILTERS,
    DATA_DIR=os.path.join("data", "images"),
    num_epochs=settings.NUM_EPOCHS,
    previous_model_path=None,
):
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
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
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
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    # Making the datasets from data/images by default

    images = []

    for file in os.listdir(DATA_DIR):
        images.append(file)

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

    if not os.path.exists(settings.MODEL_SAVE_PATH):
        os.mkdir(settings.MODEL_SAVE_PATH)

    scaler = None
    if settings.USE_AMP and device.type == "cuda":
        scaler = torch.amp.GradScaler()
        print("Using Automatic Mixed Precision (AMP).")

    start_epoch = 0
    # training loop

    model = DeepUNet(
        n_channels_in=IMG_CHANNELS_IN,
        n_classes_out=NUM_CLASSES_OUT,
        start_filters=START_FILTERS,
    ).to(device)
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
        model.load_state_dict(checkpoint["model_state_dict"])
        print(
            f"Loading model from {previous_model_path} with {in_channels} channels in, {out_channels} classes out, and {start_filters} start filters."
        )
        model = model.to(device)

    if torch.cuda.is_available() and torch.cuda.device_count() > 2:
        print(
            f"Wrapping model with nn.DataParallel for {torch.cuda.device_count()} GPUs."
        )
        model = nn.DataParallel(model)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=settings.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=settings.STEP_SIZE,
        factor=settings.GAMMA,
        mode="min",
    )
    best_val_loss = float("inf")

    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        print(f"\n--- Epoch {epoch}/{num_epochs} [Train] ---")
        batch_start_time = time.time()

        for i, (inputs, targets) in enumerate(train_dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            # show_tensor_image(inputs[0])
            # show_tensor_image(targets[0])
            # break
            optimizer.zero_grad()

            if scaler:  # AMP
                with torch.amp.autocast(
                    device_type="cuda" if torch.cuda.is_available() else "cpu"
                ):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:  # No AMP
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            if (i + 1) % 20 == 0 or (i + 1) == len(train_dataloader):
                batch_time = time.time() - batch_start_time
                print(
                    f"  Batch {i+1}/{len(train_dataloader)} | Train Loss: {loss.item():.4f} | Time: {batch_time:.2f}s"
                )
                # for debugging
                # break

        epoch_train_loss = running_loss / len(train_dataset)
        print(f"Epoch {epoch+1} [Train] Avg Loss: {epoch_train_loss:.4f}")

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in valid_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                # show_tensor_image(inputs[0])
                # show_tensor_image(targets[0])
                # break

                if scaler:  # AMP for validation
                    with torch.amp.autocast(
                        device_type="cuda" if torch.cuda.is_available() else "cpu"
                    ):
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                running_val_loss += loss.item() * inputs.size(0)

        epoch_val_loss = running_val_loss / len(valid_dataset)
        print(f"Epoch {epoch+1} [Val]   Avg Loss: {epoch_val_loss:.4f}")

        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch Duration: {epoch_duration:.2f}s")

        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(epoch_val_loss)
        elif scheduler is not None:
            scheduler.step()

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
        DATA_DIR="test/images",
        # previous_model_path="models/checkpoint_best.pth"
    )
