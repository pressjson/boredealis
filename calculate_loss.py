#!/usr/bin/env python3

import test
import network
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import torch.nn as nn
from ignite.engine import *
from ignite.metrics import *
import os
import time

from model_utils import load_model


def calculate_loss(ground_truth_path, clouded_image_path, model_path):
    ground_truth = Image.open(ground_truth_path).convert("RGB")
    clouded_image = Image.open(clouded_image_path).convert("RGB")

    # ground_truth = TF.to_tensor(ground_truth).unsqueeze(0)
    # clouded_image = TF.to_tensor(clouded_image).unsqueeze(0)
    # ground_truth = (ground_truth + 1.0) / 2
    # clouded_image = (clouded_image + 1.0) / 2
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model(model_path, device=device)
    vgg_loss = network.VGGLoss().to(device)

    start_time = time.time()
    output = test.test_with_ram(
        image=clouded_image, preloaded_model=model, device=device, verbose=False
    )
    end_time = time.time()

    # print(output)
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

    l1_loss = nn.L1Loss()
    loss = l1_loss(output.to(device), ground_truth.to(device))
    print(f"L1 Loss for model {model_path}: {loss}")

    mse_loss = nn.MSELoss()
    loss = mse_loss(output.to(device), ground_truth.to(device))
    print(f"MSE Loss for model {model_path}: {loss}")

    default_evaluator = Engine(eval_step)
    psnr = PSNR(data_range=1.0)
    psnr.attach(default_evaluator, "psnr")
    state = default_evaluator.run([[output, ground_truth]])
    print(f"PSNR for model {model_path}: {state.metrics['psnr']}")

    metric = SSIM(data_range=1.0)
    metric.attach(default_evaluator, "ssim")
    state = default_evaluator.run([[output, ground_truth]])
    print(f"SSIM for model {model_path}: {state.metrics['ssim']}")

    print(f"Time for model {model_path}: {end_time - start_time}")


def eval_step(engine, batch):
    return batch


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
    model_directory = "testing_models"
    for model in os.listdir(model_directory):
        model_path = os.path.join(model_directory, model)
        # calculate_loss(
        #     ground_truth_path="readme_images/improved_synthetic_base.png",
        #     clouded_image_path="readme_images/improved_synthetic_clouds.png",
        #     model_path=model_path,
        # )
        calculate_loss(
            ground_truth_path="readme_images/improved_synthetic_base.png",
            clouded_image_path="readme_images/improved_synthetic_base.png",
            model_path=model_path,
        )
    # compare_with_itself("readme_images/improved_synthetic_base.png")
