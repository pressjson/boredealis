#!/usr/bin/env python3

import test
import network
import torch
from PIL import Image
import torchvision.transforms.functional as TF


def calculate_loss(ground_truth_path, clouded_image_path, model_path):
    ground_truth = Image.open(ground_truth_path).convert("RGB")
    clouded_image = Image.open(clouded_image_path).convert("RGB")

    # ground_truth = TF.to_tensor(ground_truth).unsqueeze(0)
    # clouded_image = TF.to_tensor(clouded_image).unsqueeze(0)
    # ground_truth = (ground_truth + 1.0) / 2
    # clouded_image = (clouded_image + 1.0) / 2
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = test.load_model(model_path, device)
    vgg_loss = network.VGGLoss().to(device)

    output = test.test_with_ram(
        image=clouded_image, preloaded_model=model, device=device, verbose=True
    )

    print(output)
    output = TF.to_tensor(output).unsqueeze(0)
    # output = (output + 1.0) / 2

    ground_truth = TF.to_tensor(ground_truth).unsqueeze(0)
    # clouded_image = TF.to_tensor(clouded_image).unsqueeze(0)

    loss = vgg_loss(
        output.to(device),
        ground_truth.to(device),
        # target_is_features=True,
    )

    print(f"VGG Loss for model {model_path}: {loss}")


def compare_with_itself(ground_truth):

    ground_truth = Image.open(ground_truth).convert("RGB")
    ground_truth = TF.to_tensor(ground_truth).unsqueeze(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vgg_loss = network.VGGLoss().to(device)

    loss = vgg_loss(
        ground_truth.to(device),
        ground_truth.to(device),
        # target_is_features=True,
    )
    print(f"VGG Loss for comparing an image to itself: {loss}")


if __name__ == "__main__":
    # calculate_loss(
    #     ground_truth_path="readme_images/improved_synthetic_base.png",
    #     clouded_image_path="readme_images/improved_synthetic_clouds.png",
    #     model_path="testing_models/160_checkpoint_epoch_23.pth",
    # )
    compare_with_itself("readme_images/improved_synthetic_base.png")
