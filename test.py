#!/usr/bin/env python3

import os
import network
import torch
import torchvision.transforms.functional as FU
from PIL import Image

if not os.path.exists("local_settings.py"):
    print("Warning: local settings not found. Using default settings.")
    import settings
else:
    import local_settings as settings


def test(
    image_path=os.path.join("test", "images", "02032021_221508_0001.png"),
    model_load_path=os.path.join("models", "checkpoint_best.pth"),
):
    model = network.DeepUNet(
        3,
        3,
        128,
        # settings.START_FILTERS,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # checkpoint = torch.load(model_load_path)

    # model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    model.eval()

    sample = Image.open(image_path).convert("RGB")
    input_tensor = FU.to_tensor(sample)
    input_tensor = input_tensor.unsqueeze(0)
    input_tensor = input_tensor.to(device)
    print(input_tensor.shape)

    with torch.no_grad():
        print("Testing model")
        output_tensor = model(input_tensor)

    output_tensor = output_tensor.squeeze(0)
    # without denormalizing
    output_image = FU.to_pil_image(output_tensor)
    # with denormalizing
    # output_image = FU.to_pil_image(output_tensor * 0.5 + 0.5)

    return output_image


if __name__ == "__main__":
    test(
        # model_load_path="32_filters_models/checkpoint_epoch_20.pth"
        model_load_path="models/checkpoint_best.pth"
    ).show()
