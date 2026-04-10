"""Model helpers for smoother training."""

from collections import OrderedDict

import torch

from smoother_model import DeflickerCNN


def build_smoother_model(
    input_frames=5,
    num_res_blocks=12,
    hidden_channels=128,
    save_memory=True,
):
    return DeflickerCNN(
        input_frames=input_frames,
        num_res_blocks=num_res_blocks,
        hidden_channels=hidden_channels,
        save_memory=save_memory,
    )


def peek_checkpoint_architecture(checkpoint_path, default_input_frames, default_num_res_blocks, default_hidden_channels):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    input_frames = checkpoint.get("input_frames", default_input_frames)
    num_res_blocks = checkpoint.get("num_res_blocks", default_num_res_blocks)
    hidden_channels = checkpoint.get("hidden_channels", default_hidden_channels)
    del checkpoint
    return input_frames, num_res_blocks, hidden_channels


def load_smoother_checkpoint(model, optimizer, checkpoint_path, map_location):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model_state = checkpoint["model_state_dict"]
    optimizer_state = checkpoint["optimizer_state_dict"]

    new_state_dict = OrderedDict()
    for key, value in model_state.items():
        if key.startswith("module."):
            new_state_dict[key[7:]] = value
        else:
            new_state_dict[key] = value

    model.load_state_dict(new_state_dict)
    optimizer.load_state_dict(optimizer_state)
    return checkpoint.get("epoch", 0) + 1
