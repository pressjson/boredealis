"""Cloud augmentation transforms for training."""

import os
import random

import noise
import numpy
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFilter

if not os.path.exists("local_settings.py"):
    import settings
else:
    import local_settings as settings


def generate_perlin_noise_map(
    size,
    scale=None,
    octaves=None,
    persistence=None,
    lacunarity=None,
    iterations=1,
    weight=1.0,
):
    """Generate a normalized Perlin noise map."""
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
    normalized_world = weight * normalized_world + generate_perlin_noise_map(
        size,
        iterations=iterations - 1,
        weight=weight / 2,
    )

    return numpy.clip(normalized_world, 0, 1)


def colorize_array(normalized_world, lower_bound, upper_bound):
    """Map a normalized array into an RGB image."""
    size = normalized_world.shape[0]
    r_low, g_low, b_low = lower_bound
    r_up, g_up, b_up = upper_bound

    pixels = numpy.zeros((size, size, 3), dtype=numpy.uint8)

    for x in range(size):
        for y in range(size):
            noise_val_norm = normalized_world[x][y]
            r = int(r_low + abs(r_up - r_low) * noise_val_norm)
            g = int(g_low + abs(g_up - g_low) * noise_val_norm)
            b = int(b_low + abs(b_up - b_low) * noise_val_norm)

            r = max(0, min(255, r))
            g = max(0, min(255, g))
            b = max(0, min(255, b))

            pixels[x, y] = (r, g, b)

    return Image.fromarray(pixels)


def draw_center_circle(radius=250, vertical_offset=15, size=settings.IMAGE_SIZE):
    """Draw the center circle mask."""
    width, height = settings.IMAGE_SIZE
    radius = 250
    vertical_offset = 15
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)

    center_x = width // 2
    center_y = height // 2

    left = center_x - radius
    top = center_y - radius - vertical_offset
    right = center_x + radius
    bottom = center_y + radius - vertical_offset

    draw.ellipse((left, top, right, bottom), fill=255)

    return mask


def crop_to_center_circle(pil_image: Image.Image) -> Image.Image:
    """Keep only the center circle of an image."""
    img_rgba = pil_image.convert("RGBA")
    mask = draw_center_circle(size=pil_image.size)
    img_rgba.putalpha(mask)
    return img_rgba


def sum_of_vals(arr):
    """Return the sum of all values in arr."""
    total = 0
    for value in arr:
        total += value
    return total


def get_random_valid_coords(sampling_image, boost=0):
    """Get random valid coordinates in an image."""
    while True:
        random_coordinates = (
            random.randint(0, sampling_image.size[0] - 1),
            random.randint(0, sampling_image.size[0] - 1),
        )
        bound = list(sampling_image.getpixel(random_coordinates))
        if bound[3] == 0:
            continue

        bound[3] = 0
        cloud_brightness = random.randint(min(0, boost), max(0, boost))
        for i in range(len(bound)):
            bound[i] = numpy.clip(0, 255, bound[i] + cloud_brightness)
        return tuple(bound[:3])


def make_alpha_image(
    blend_strength=0.0,
    scale=random.uniform(150, 300),
    octaves=random.randint(3, 5),
    persistence=random.uniform(0.4, 0.5),
    lacunarity=random.uniform(2.0, 2.2),
    iterations=random.randint(2, 4),
):
    """Create the alpha mask used for synthetic clouds."""
    alpha_world = generate_perlin_noise_map(
        settings.IMAGE_SIZE[0],
        scale=scale,
        octaves=octaves,
        persistence=persistence,
        lacunarity=lacunarity,
        iterations=iterations,
    )
    circle_mask = numpy.array(draw_center_circle())
    final_alpha = (
        (alpha_world * blend_strength) * (circle_mask / 255.0) * 255
    ).astype(numpy.uint8)
    return Image.fromarray(final_alpha)


class CombineWithClouds:
    def __init__(self, output_size, noise_strength=None):
        self.output_size = output_size
        self.alpha_strength = noise_strength

    def __call__(self, main_image):
        main_image = main_image.convert("RGBA")

        upper_bound = get_random_valid_coords(main_image, boost=150)
        lower_bound = get_random_valid_coords(main_image, boost=-10)

        if sum_of_vals(lower_bound) > sum_of_vals(upper_bound):
            lower_bound, upper_bound = upper_bound, lower_bound

        alpha_lower_bound = settings.ALPHA_LOWER_BOUND
        alpha_upper_bound = settings.ALPHA_UPPER_BOUND

        fake_clouds = generate_perlin_noise_map(
            settings.IMAGE_SIZE[0], iterations=random.randint(3, 5)
        )
        fake_clouds = colorize_array(
            fake_clouds, lower_bound=lower_bound, upper_bound=upper_bound
        )
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
            ImageFilter.GaussianBlur(
                radius=random.randint(0, 1) if self.alpha_strength else 0
            )
        )

        blur_mask = draw_center_circle()
        final_image.paste(blurred_image, (0, 0), blur_mask)

        return final_image.convert("RGB")


class RandomApplyTransforms:
    def __init__(self, output_size, random_threshold, noise_weight, noise_strength=None):
        self.output_size = output_size
        self.random_threshold = random_threshold
        self.noise_weight = noise_weight
        self.alpha_strength = noise_strength

    def __call__(self, sample):
        if random.uniform(0, 1) > self.random_threshold:
            return TF.to_tensor(sample)

        cloud = CombineWithClouds(self.output_size, self.alpha_strength)
        sample = cloud(sample)
        sample = TF.to_tensor(sample)
        noise_tensor = torch.rand_like(sample) * self.noise_weight
        sample = sample + noise_tensor
        sample = torch.clamp(sample, 0.0, 1.0)

        return sample
