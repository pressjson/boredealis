#!/usr/bin/env python3

IMAGE_SIZE = (608, 608)

NUM_EPOCHS = 100

# Hyperparams

# Probablity that the random transforms *do* get applied
RANDOM_APPLY_THRESHOLD = 0.5
NOISE_STRENGTH = 0
# n * data_size for training, (1 - n) * data_size for validation
VALUE_SPLIT = 0.8
LEARNING_RATE = 1e-3
MODEL_SAVE_PATH = "models"
EPOCH_SAVE_INTERVAL = 5
STEP_SIZE = 10
GAMMA = 0.5
# Transparency upper and lower bounds for the cloud transform
ALPHA_LOWER_BOUND = 0.1
ALPHA_UPPER_BOUND = 0.8
# Tweaking START_FILTERS can crash the GPU, since increasing this dramatically increases VRAM usage
# This also makes the model more deep, so find a balance
START_FILTERS = 32

# Default settings, optimized for my RX 6800 XT and 7800X3D
# The batch size is so small because the images are large. It quickly overwhelms my GPU

NUM_WORKERS = 16
BATCH_SIZE = 2

PIN_MEMORY = True
USE_AMP = True
