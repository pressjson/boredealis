#!/usr/bin/env python3

IMAGE_SIZE = (608, 608)
DATA_BASE_DIR = "../data"

NUM_EPOCHS = 500

# transformer hyperparams

EMBED_DIM = 1024
PATCH_SIZE = 16
DEPTH = 24
NUM_HEADS = 16
MLP_RATIO = 4.0
DROPOUT = 0.1
IN_CHANS = 3

# local settings, for my rx 6800 xt

NUM_WORKERS = 8
USE_DATA_PARALLEL = True
BATCH_SIZE = 32
