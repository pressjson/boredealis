#!/usr/bin/env python3

IMAGE_SIZE = (608, 608)

NUM_EPOCHS = 100

# Hyperparams

# Probablity that the random transforms *do* get applied
RANDOM_APPLY_THRESHOLD = 0.5
NOISE_STRENGTH = 0.05
# n * data_size for training, (1 - n) * data_size for validation
VALUE_SPLIT = 0.8
LEARNING_RATE = 1e-2
MODEL_SAVE_PATH = "models"
EPOCH_SAVE_INTERVAL = 5
STEP_SIZE = 10
GAMMA = 0.5
# Transparency upper and lower bounds for the cloud transform
ALPHA_LOWER_BOUND = 0.3
ALPHA_UPPER_BOUND = 0.7
# Tweaking START_FILTERS can crash the GPU, since increasing this dramatically increases VRAM usage
# This also makes the model more deep, so find a balance
START_FILTERS = 32

# Default settings, optimized for my rx 6800 xt and 7800X3D

NUM_WORKERS = 8
BATCH_SIZE = 1

PIN_MEMORY = False
USE_AMP = False
