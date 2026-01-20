import os
from collections import OrderedDict
import time
import random
import numpy
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision import transforms
import torch.optim as optim
from PIL import Image, ImageDraw, ImageFilter
import noise
from unet_model import DeepUNet
from vgg_loss import VGGLoss

if not os.path.exists("local_settings.py"):
    print("Warning: local settings not found. Using default settings.")
    import settings
else:
    import local_settings as settings

# @TODO: separate all of this crap into many individual files
#        but i don't want to


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


def make_video_datasets(data_dir):
    videos = []
    for video_path in os.listdir(data_dir):
        videos.append(video_path)

    random.shuffle(videos)

    train_videos = videos[: int(settings.VALUE_SPLIT * len(videos))]
    valid_videos = videos[int(settings.VALUE_SPLIT * len(videos)) :]

    train = []
    valid = []
    for video in train_videos:
        video_path = os.path.join(data_dir, video)
        # if debug:
        #     print(video)
        for image in os.listdir(video_path):
            image = os.path.join(video, image)
            train.append(image)

    for video in valid_videos:
        video_path = os.path.join(data_dir, video)
        for image in os.listdir(video_path):
            image = os.path.join(video, image)
            # if debug:
            #     print(image)
            valid.append(image)

    random.shuffle(train)
    random.shuffle(valid)

    if settings.NUM_IMAGES != -1:
        train = train[: int(settings.NUM_IMAGES * settings.VALUE_SPLIT)]
        valid = valid[: int(settings.NUM_IMAGES * (1 - settings.VALUE_SPLIT))]

    return train, valid


def make_dataloaders(train, valid, data_dir, clear_transform, cloud_transform):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = ImageDataset(
        train,
        data_dir=data_dir,
        clear_transform=clear_transform,
        cloud_transform=cloud_transform,
    )
    valid_dataset = ImageDataset(
        valid,
        data_dir=data_dir,
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

    return train_dataloader, valid_dataloader


# perlin noise shenanigains

def hex_to_rgb(hex):
    """Takes a hex string and returns the corresponding RGB tuple.

    From https://www.30secondsofcode.org/python/s/hex-to-rgb/

    Args:
        hex (str): a hex tuple, formatted AABBCC.

    Returns:
        int tuple, formatted (AA, BB, CC) but converted to the corresponding integer value.
    """
    return tuple(int(hex[i : i + 2], 16) for i in (0, 2, 4))


def generate_perlin_noise_map(
    size,
    scale=None,
    octaves=None,
    persistence=None,
    lacunarity=None,
    iterations=1,
    weight=1.0,
):
    """Generate a Perlin noise map of size (size, size).
    Args:
        size (int): The size of the final array.
        iterations (int): The number of iterations to do. 1 = Perlin noise, more = fBm
        weight (float): The initial weight of layer 1. Each subsequent layer = weight / 2
        everything else: Perlin noise hyperparameters.

    Returns:
        A 2d array of the world between 0 and 1.
    """
    world = numpy.zeros((size, size))
    if iterations < 1:
        return world
    if scale is None:
        scale = random.uniform(100.0, 400.0)
    if octaves is None:
        octaves = random.randint(2, 5)
    if persistence is None:
        persistence = random.uniform(0.3, 0.6)
    if lacunarity is None:
        lacunarity = random.uniform(1.8, 4)

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

    normalized_world = weight * (world - min_val) / (max_val - min_val)
    # some simple recursive programming. sorry nasa
    normalized_world = weight * normalized_world + generate_perlin_noise_map(
        size,
        iterations=(iterations - 1),
        weight=(weight / 2),
    )

    return numpy.clip(normalized_world, 0, 1)


def colorize_array(normalized_world, lower_bound, upper_bound):
    """Makes a color image with a world, lower_bound, and upper_bound

    Args:
        normalized_world (arr): A world between 0 and 1.
        lower_bound (tuple): The lower bound.
        upper_bound (tuple): The upper bound.

    Returns:
        PIL.Image.Image from the interpolated world
    """
    size = normalized_world.shape[0]
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

            r = max(0, min(255, r))
            g = max(0, min(255, g))
            b = max(0, min(255, b))

            pixels[x, y] = (r, g, b)

    return Image.fromarray(pixels)

def draw_center_circle(
    radius=250,
    vertical_offset=15,
    size=settings.IMAGE_SIZE
):
    """Draw the center circle."""
    width, height = settings.IMAGE_SIZE
    radius = 250
    vertical_offset = 15
    # Create a mask:
    # Start with a completely black (transparent) mask
    mask = Image.new("L", size, 0)  # 'L' mode for grayscale mask
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

    return mask

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
    # Ensure the image has an alpha channel for transparency
    img_rgba = pil_image.convert("RGBA")

    mask = draw_center_circle(size=pil_image.size)
    img_rgba.putalpha(mask)

    return img_rgba


def sum_of_vals(arr):
    """Return the sum of all values in arr."""
    sum = 0
    for i in arr:
        sum += i
    return sum


def get_random_valid_coords(sampling_image, boost=0):
    """Get random valid coordinates in an image. Valid is in the center circle.

    Args:
        main_image (PIL.Image.Image): The image to get a bound from. Should be cropped before.
        boost (int): The amount of boost to add. Can be positive or negative.

    Returns:
        tuple of len 3 representing (R, G, B) for the bound.
    """
    while True:
        random_coordinates = (
            random.randint(0, sampling_image.size[0] - 1),
            random.randint(0, sampling_image.size[0] - 1),
        )
        bound = list(sampling_image.getpixel(random_coordinates))
        if not bound[3] == 0:
            bound[3] = 0
            cloud_brightness = random.randint(min(0, boost), max(0, boost))
            for i in range(len(bound)):
                # print(upper_bound[i])
                bound[i] = (
                    numpy.clip(0, 255, bound[i] + cloud_brightness)
                )
            bound = tuple(bound[:3])
            return bound


def make_alpha_image(
    blend_strength=0.0,
    scale=random.uniform(150, 300),
    octaves=random.randint(3, 5),
    persistence=random.uniform(0.4, 0.5),
    lacunarity=random.uniform(2.0, 2.2),
    iterations=random.randint(2, 4),
):
    alpha_world = generate_perlin_noise_map(
        settings.IMAGE_SIZE[0],
        scale=scale,
        octaves=octaves,
        persistence=persistence,
        lacunarity=lacunarity,
        iterations=iterations,
    )
    circle_mask = numpy.array(draw_center_circle())
    final_alpha = ((alpha_world * blend_strength) * (circle_mask / 255.0) * 255).astype(numpy.uint8)
    return Image.fromarray(final_alpha)


class CombineWithClouds:
    def __init__(self, output_size, noise_strength=None):
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.output_size = output_size
        self.alpha_strength = noise_strength

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
        # sampling_image = crop_to_center_circle(main_image)
        # Pick lower bound
        upper_bound = get_random_valid_coords(main_image, boost=150)
        lower_bound = get_random_valid_coords(main_image, boost=-10)

        if sum_of_vals(lower_bound) > sum_of_vals(upper_bound):
            # crude check
            lower_bound, upper_bound = upper_bound, lower_bound

        alpha_lower_bound = settings.ALPHA_LOWER_BOUND
        alpha_upper_bound = settings.ALPHA_UPPER_BOUND

        fake_clouds = generate_perlin_noise_map(
            settings.IMAGE_SIZE[0], iterations=random.randint(3, 5)
        )
        # for _ in range(0, random.randint(3, 5)):
        #     more_fake_clouds = generate_perlin_noise_map(settings.IMAGE_SIZE[0])
        #     fake_clouds = numpy.maximum(fake_clouds, more_fake_clouds)
        # print(f"Fake clouds: {fake_clouds}")

        fake_clouds = colorize_array(
            fake_clouds, lower_bound=lower_bound, upper_bound=upper_bound
        )

        # print(f"Cropped clouds: {cropped_clouds}")

        # combined_image = Image.blend(
        #     main_image,
        #     cropped_clouds,
        #     random.uniform(alpha_lower_bound, alpha_upper_bound),
        # )
        blend_strength = random.uniform(alpha_lower_bound, alpha_upper_bound)
        if self.alpha_strength:
            blend_strength = self.alpha_strength
            print(f"blend_strength: {blend_strength}")

        alpha_image = make_alpha_image(blend_strength=blend_strength)

        r, g, b = fake_clouds.split()

        fake_clouds = Image.merge("RGBA", (r, g, b, alpha_image))

        combined_image = Image.alpha_composite(main_image, fake_clouds) 
        final_image = combined_image.copy()

        blurred_image = combined_image.filter(

            ImageFilter.GaussianBlur(radius=random.randint(0, 1) if self.alpha_strength else 0)
        )
        # blurred_center = crop_to_center_circle(blurred_image)

        blur_mask = draw_center_circle()
        final_image.paste(blurred_image, (0, 0), blur_mask)

        return final_image.convert("RGB")


class RandomApplyTransforms:
    # His name is Randy
    def __init__(self, output_size, random_threshold, noise_weight, noise_strength=None):
        self.output_size = output_size
        self.random_threshold = random_threshold
        self.noise_weight = noise_weight
        self.alpha_strength = noise_strength
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __call__(self, sample):
        # for debugging why my computer crashes
        # return FU.to_tensor(sample)

        if random.uniform(0, 1) > self.random_threshold:
            # do nothing
            return TF.to_tensor(sample)

        # apply multiple clouds
        cloud = CombineWithClouds(self.output_size, self.alpha_strength)

        sample = cloud(sample)
        sample = TF.to_tensor(sample)
        noise = torch.rand_like(sample) * self.noise_weight
        sample = sample + noise
        sample = torch.clamp(sample, 0.0, 1.0)

        return sample


def train_model(
    IMG_CHANNELS_IN=3,
    NUM_CLASSES_OUT=3,
    START_FILTERS=settings.START_FILTERS,
    data_dir=os.path.join("data", "images"),
    num_epochs=settings.NUM_EPOCHS,
    previous_model_path=None,
    levels=5,
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
    # This approach combines all of the images into one dataset, which is bad because it can cause
    # similar looking frames to be in both the training and validation set

    # images = []

    # i = settings.NUM_IMAGES
    # for file in os.listdir(data_dir):
    #     images.append(file)
    #     if i != -1:
    #         i = i - 1
    #         if i == 0:
    #             break

    # print(f"Found {len(images)} images")

    # if len(images) == 0:
    #     raise ValueError(
    #         f"No images found in {images}. Check file naming and structure."
    #     )
    # print(
    #     f"Using {int(settings.VALUE_SPLIT * len(images))} for training and {len(images) - int(settings.VALUE_SPLIT * len(images))} for validation"
    # )
    # random.shuffle(images)
    # train = images[: int(settings.VALUE_SPLIT * len(images))]
    # valid = images[int(settings.VALUE_SPLIT * len(images)) :]

    # This approach uses videos decomposed into folders
    # DATA
    #   VIDEO_1
    #     frame_0001.png
    #     frame_0002.png
    #     ...
    #   VIDEO_2
    #     frame_001.png
    #     ...
    #   ...

    # videos = []
    # for video_path in os.listdir(data_dir):
    #     videos.append(video_path)

    # look, ma, i'm refactoring!

    # random.shuffle(videos)

    # train_videos = videos[: int(settings.VALUE_SPLIT * len(videos))]
    # valid_videos = videos[int(settings.VALUE_SPLIT * len(videos)) :]

    # train = []
    # valid = []
    # for video in train_videos:
    #     video_path = os.path.join(data_dir, video)
    #     # if debug:
    #     #     print(video)
    #     for image in os.listdir(video_path):
    #         image = os.path.join(video, image)
    #         train.append(image)

    # for video in valid_videos:
    #     video_path = os.path.join(data_dir, video)
    #     for image in os.listdir(video_path):
    #         image = os.path.join(video, image)
    #         # if debug:
    #         #     print(image)
    #         valid.append(image)

    # random.shuffle(train)
    # random.shuffle(valid)

    # if settings.NUM_IMAGES != -1:
    #     train = train[: int(settings.NUM_IMAGES * settings.VALUE_SPLIT)]
    #     valid = valid[: int(settings.NUM_IMAGES * (1 - settings.VALUE_SPLIT))]

    train, valid = make_video_datasets(data_dir)

    print(
        f"Using {len(train)} images for training and {len(valid)} images for validation"
    )

    # train_dataset = ImageDataset(
    #     train,
    #     data_dir=data_dir,
    #     clear_transform=clear_transform,
    #     cloud_transform=cloud_transform,
    # )
    # valid_dataset = ImageDataset(
    #     valid,
    #     data_dir=data_dir,
    #     clear_transform=clear_transform,
    #     cloud_transform=cloud_transform,
    # )

    # train_dataloader = DataLoader(
    #     train_dataset,
    #     batch_size=settings.BATCH_SIZE,
    #     num_workers=settings.NUM_WORKERS,
    #     shuffle=True,
    #     pin_memory=(device.type == "cuda"),
    # )
    # valid_dataloader = DataLoader(
    #     valid_dataset,
    #     batch_size=settings.BATCH_SIZE,
    #     num_workers=settings.NUM_WORKERS,
    #     shuffle=True,
    #     pin_memory=(device.type == "cuda"),
    # )

    train_dataloader, valid_dataloader = make_dataloaders(
        train, valid, data_dir, clear_transform, cloud_transform
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
    if previous_model_path is None:
        print(
            f"Initialized DeepUNet with {levels} layers, {IMG_CHANNELS_IN} channels in, {NUM_CLASSES_OUT} classes out, and {START_FILTERS} start filters."
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
    L1_WEIGHT = 0
    VGG_WEIGHT = 1

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
        num_batches_processed = 0
        print(f"\n--- Epoch {epoch}/{num_epochs} [Train] ---")
        batch_start_time = time.time()

        for i, (inputs, targets) in enumerate(train_dataloader):
            inputs, targets = inputs.to(
                f"cuda:{settings.DEVICE_IDS[0]}" if settings.USE_DEVICE_IDS else device
            ), targets.to(
                f"cuda:{settings.DEVICE_IDS[0]}" if settings.USE_DEVICE_IDS else device
            )
            optimizer.zero_grad()

            if scaler:  # AMP
                with torch.amp.autocast(
                    device_type="cuda" if torch.cuda.is_available() else "cpu"
                ):
                    outputs = model(inputs)
                    l1_loss = criterion(outputs, targets)
                    outputs = (outputs + 1.0) / 2.0
                    targets = (targets + 1.0) / 2.0
                    if settings.USE_VGG_DEVICE:
                        vgg_loss = vgg_loss_crit(
                            outputs.to(f"cuda:{settings.VGG_DEVICE_ID}"),
                            vgg_loss_crit.get_features(
                                targets.to(f"cuda:{settings.VGG_DEVICE_ID}")
                            ),
                            target_is_features=True,
                        )
                    else:
                        vgg_loss = vgg_loss_crit(
                            outputs,
                            vgg_loss_crit.get_features(targets),
                            target_is_features=True,
                        )
                loss = l1_loss * L1_WEIGHT + vgg_loss.to(l1_loss.device) * VGG_WEIGHT
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:  # No AMP
                outputs = model(inputs)
                l1_loss = criterion(outputs, targets)
                outputs = (outputs + 1.0) / 2.0
                targets = (targets + 1.0) / 2.0
                if settings.USE_VGG_DEVICE:
                    vgg_loss = vgg_loss_crit(
                        outputs.to(f"cuda:{settings.VGG_DEVICE_ID}"),
                        vgg_loss_crit.get_features(
                            targets.to(f"cuda:{settings.VGG_DEVICE_ID}")
                        ),
                        target_is_features=True,
                    )
                else:
                    vgg_loss = vgg_loss_crit(
                        outputs,
                        vgg_loss_crit.get_features(targets),
                        target_is_features=True,
                    )
                loss = l1_loss * L1_WEIGHT + vgg_loss.to(l1_loss.device) * VGG_WEIGHT
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            running_loss += loss.item()
            num_batches_processed = i + 1

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

        epoch_train_loss = running_loss / num_batches_processed
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
                if scaler:  # AMP for validation
                    with torch.amp.autocast(
                        device_type="cuda" if torch.cuda.is_available() else "cpu"
                    ):
                        outputs = model(inputs)
                        l1_loss = criterion(outputs, targets)
                        outputs = (outputs + 1.0) / 2.0
                        targets = (targets + 1.0) / 2.0
                        if settings.USE_VGG_DEVICE:
                            vgg_loss = vgg_loss_crit(
                                outputs.to(f"cuda:{settings.VGG_DEVICE_ID}"),
                                vgg_loss_crit.get_features(
                                    targets.to(f"cuda:{settings.VGG_DEVICE_ID}")
                                ),
                                target_is_features=True,
                            )
                        else:
                            vgg_loss = vgg_loss_crit(
                                outputs,
                                vgg_loss_crit.get_features(targets),
                                target_is_features=True,
                            )
                        loss = (
                            l1_loss * L1_WEIGHT
                            + vgg_loss.to(l1_loss.device) * VGG_WEIGHT
                        )
                else:
                    outputs = model(inputs)
                    l1_loss = criterion(outputs, targets)
                    outputs = (outputs + 1.0) / 2.0
                    targets = (targets + 1.0) / 2.0
                    if settings.USE_VGG_DEVICE:
                        vgg_loss = vgg_loss_crit(
                            outputs.to(f"cuda:{settings.VGG_DEVICE_ID}"),
                            vgg_loss_crit.get_features(
                                targets.to(f"cuda:{settings.VGG_DEVICE_ID}")
                            ),
                            target_is_features=True,
                        )
                    else:
                        vgg_loss = vgg_loss_crit(
                            outputs,
                            vgg_loss_crit.get_features(targets),
                            target_is_features=True,
                        )
                    loss = (
                        l1_loss * L1_WEIGHT + vgg_loss.to(l1_loss.device) * VGG_WEIGHT
                    )

                running_val_loss += loss.item()

        epoch_val_loss = running_val_loss / len(valid_dataloader)
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

    print("\n--- Training Finished ---")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Best model saved at: {settings.MODEL_SAVE_PATH}")


def show_tensor_image(tensor):
    image = tensor.detach().cpu()
    image = TF.to_pil_image(image)
    image.show()


if __name__ == "__main__":
    train_model(
        # data_dir=os.path.join("data", "images"),
        data_dir=os.path.join("png_split_training_images"),
        # previous_model_path=os.path.join("models", "64_checkpoint_best.pth"),
        debug=True,
        levels=5,
    )

#  LocalWords:  ROCm
