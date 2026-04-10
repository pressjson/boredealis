"""Dataset helpers for image-to-image training."""

import os
import random

from PIL import Image
from torch.utils.data import Dataset

if not os.path.exists("local_settings.py"):
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
        self.images = images or []
        self.data_dir = data_dir or ""
        self.clear_transform = clear_transform
        self.cloud_transform = cloud_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        image = Image.open(os.path.join(self.data_dir, self.images[i])).convert("RGB")
        if self.clear_transform is None or self.cloud_transform is None:
            raise ValueError("Both clear_transform and cloud_transform must be provided.")

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
        for image in os.listdir(video_path):
            image = os.path.join(video, image)
            train.append(image)

    for video in valid_videos:
        video_path = os.path.join(data_dir, video)
        for image in os.listdir(video_path):
            image = os.path.join(video, image)
            valid.append(image)

    random.shuffle(train)
    random.shuffle(valid)

    if settings.NUM_IMAGES != -1:
        train = train[: int(settings.NUM_IMAGES * settings.VALUE_SPLIT)]
        valid = valid[: int(settings.NUM_IMAGES * (1 - settings.VALUE_SPLIT))]

    return train, valid
