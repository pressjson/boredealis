"""Dataset helpers for smoother training."""

import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

if not os.path.exists("local_settings.py"):
    import settings
else:
    import local_settings as settings


class VideoDataset(Dataset):
    """Dataset for loading frame windows from videos."""

    def __init__(self, files, height=608, width=608, window=5):
        self.height = height
        self.width = width
        self.samples = []
        self.video_cache = []
        self.window = window

        print("Pre-loading videos into RAM (This might take a moment)...")

        for vid_path in files:
            cap = cv2.VideoCapture(vid_path)
            if not cap.isOpened():
                print(f"Failed to open {vid_path}")
                continue

            video_frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_frames.append(frame)
            cap.release()

            if len(video_frames) < window + 1:
                continue

            video_tensor = torch.from_numpy(np.stack(video_frames)).share_memory_()
            self.video_cache.append(video_tensor)

            total_frames = len(video_frames)
            start_t = window // 2 + 1
            end_t = total_frames - window // 2 - 1
            cache_idx = len(self.video_cache) - 1

            if end_t > start_t:
                for t in range(start_t, end_t):
                    self.samples.append((cache_idx, t))

        print(f"Loaded {len(self.video_cache)} videos. Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cache_idx, t = self.samples[idx]
        video_tensor = self.video_cache[cache_idx]
        padding = self.window // 2 + 1
        frames_uint8 = video_tensor[t - padding : t + padding]
        frames = frames_uint8.permute(0, 3, 1, 2).float() / 255.0
        window_prev = frames[0 : self.window].reshape(-1, self.height, self.width)
        window_curr = frames[1 : self.window + 1].reshape(-1, self.height, self.width)
        return window_curr, window_prev


def split_video_files(data_dir):
    data_arr = [os.path.join(data_dir, video) for video in os.listdir(data_dir)]
    if settings.NUM_IMAGES > 0:
        data_arr = data_arr[: settings.NUM_IMAGES]
    split_idx = int(len(data_arr) * settings.VALUE_SPLIT)
    return data_arr[:split_idx], data_arr[split_idx:]


def make_smoother_dataloaders(train_files, valid_files, input_frames, batch_size, num_workers):
    train_dataset = VideoDataset(train_files, window=input_frames)
    valid_dataset = VideoDataset(valid_files, window=input_frames)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_dataset, valid_dataset, train_loader, valid_loader
