#!/usr/bin/env python3

IMAGE_SIZE = (608, 608)

NUM_EPOCHS = 50

# Hyperparams

# Probablity that the random transforms *do* get applied
RANDOM_APPLY_THRESHOLD = 0.95
NOISE_STRENGTH = 0.1
# n * data_size for training, (1 - n) * data_size for validation
VALUE_SPLIT = 0.85
LEARNING_RATE = 1e-3
MODEL_SAVE_PATH = "models"
EPOCH_SAVE_INTERVAL = 1
STEP_SIZE = 10
GAMMA = 0.1
# Transparency upper and lower bounds for the cloud transform
ALPHA_LOWER_BOUND = 0.3
ALPHA_UPPER_BOUND = 0.7
# Tweaking START_FILTERS can crash the GPU, since increasing this dramatically increases VRAM usage
# This also makes the model more deep, so find a balance
START_FILTERS = 16

# Default settings, optimized for my rx 6800 xt and 7800X3D

NUM_WORKERS = 16
BATCH_SIZE = 4

PIN_MEMORY = True
USE_AMP = True
