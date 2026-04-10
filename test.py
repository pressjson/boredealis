#!/usr/bin/env python3

import argparse
import os
import time

import torch
import torchvision.transforms.functional as TF
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.v2 as T2

from cloud_transform import RandomApplyTransforms
from model_utils import get_model_names, load_model

if not os.path.exists("local_settings.py"):
    # print("Warning: local settings not found. Using default settings.")
    # Not needed, because the message already displayed with ~import network~
    import settings
else:
    import local_settings as settings


# def test(
#     image_path=os.path.join("test", "images", "02032021_221508_0001.png"),
#     model_load_path=os.path.join("models", "checkpoint_best.pth"),
# ):
#     model = network.DeepUNet(
#         3,
#         3,
#         64,
#         # settings.START_FILTERS,
#     )
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     checkpoint = torch.load(model_load_path)

#     model.load_state_dict(checkpoint["model_state_dict"])
#     model = model.to(device)

#     model.eval()

#     sample = Image.open(image_path).convert("RGB")
#     input_tensor = FU.to_tensor(sample)
#     input_tensor = input_tensor.unsqueeze(0)
#     input_tensor = input_tensor.to(device)
#     print(input_tensor.shape)

#     with torch.no_grad():
#         print("Testing model")
#         output_tensor = model(input_tensor)

#     output_tensor = output_tensor.squeeze(0)
#     # without denormalizing
#     output_image = FU.to_pil_image(output_tensor)
#     # with denormalizing
#     output_image = FU.to_pil_image(output_tensor * 0.5 + 0.5)

#     return output_image


# generated from google gemini 2.5 pro based off my code
# that llm is actually smart


# @TODO: refactor this crap

def test(
    image_path=os.path.join("readme_images", "Randii.png"),
    model_load_path=os.path.join("models", "checkpoint_best.pth"),
    image_size_trained=settings.IMAGE_SIZE,
    preloaded_model=None,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    verbose=True,
    blend_strength=0.3,
    iterations=1,
    model_name=None,
) -> Image.Image:
    """Tests the model and returns a PIL Image.

    Args:
        image_path (str): Path to the image to be converted.
        model_load_path (str): Path to the model to be loaded.
        image_size_trained (str): Should not be touched.

    Returns:
        PIL.Image.Image if successful.
        -1 if something went wrong.
    """

    if not preloaded_model:
        model = load_model(
            model_load_path=model_load_path,
            device=device,
            verbose=verbose,
            model_name=model_name,
        )
    else:
        model = preloaded_model
    # --- Image Preprocessing ---
    # This should match the transformations applied to your training input (cloudy_image or clear_image before normalization)
    # Specifically Resize and Normalize.
    # Your training loop normalizes inputs to [-1, 1]
    # if verbose:
    #     print(f"Model: {model}")
    preprocess = T.Compose(
        [

            T.Resize(image_size_trained),  # Use the size the model was trained on
            # T.ToTensor(),  # Converts PIL image [0,255] to tensor [0,1]
            RandomApplyTransforms(
                settings.IMAGE_SIZE,
                settings.RANDOM_APPLY_THRESHOLD,
                0,
                noise_strength=blend_strength,
            ),
            T.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
            ),  # Normalizes [0,1] to [-1,1]
        ]
    )

    if verbose:
        print(f"Loading and preprocessing image: {image_path}")
    # try:
    image = Image.open(image_path).convert("RGB")
    if image.size != settings.IMAGE_SIZE:
        if verbose:
            print(
                f"Warning: input size {image.size} is not {settings.IMAGE_SIZE}. Resizing . . ."
            )
            image = image.resize(settings.IMAGE_SIZE, resample=Image.BILINEAR)
            return image

    if iterations < 1:
        print(f"Warning: iterations should be greater than or equal to 1, currently {iterations}")
    for _ in range(iterations): 
        input_tensor = preprocess(image)

        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension (B, C, H, W)
        input_tensor = input_tensor.to(device)
        if verbose:
            print(f"Input tensor shape: {input_tensor.shape}")
        # except Exception as e:
        #     print(f"Error processing image {image_path}: {e}")
        #     return None

        # --- Inference ---
        with torch.no_grad():  # Disable gradient calculations for inference
            if verbose:
                print("Running inference...")
            output_tensor = model(input_tensor)
            if verbose:
                print(
                    f"Output tensor shape: {output_tensor.shape}, min: {output_tensor.min():.2f}, max: {output_tensor.max():.2f}"
                )

        # --- Postprocessing ---
        output_tensor = output_tensor.squeeze(0)
        # Denormalize: model outputs are in [-1, 1] (due to Tanh)
        # We need to map them back to [0, 1] for to_pil_image
        output_tensor_denorm = output_tensor * 0.5 + 0.5
        # Clamp to ensure values are strictly in [0, 1] range after denormalization
        output_tensor_denorm = torch.clamp(output_tensor_denorm, 0, 1)

        # Convert tensor to PIL Image
        image = TF.to_pil_image(output_tensor_denorm)
    if verbose:
        print("Inference complete. Output image generated.")

    return image


def test_with_ram(
    image=None,
    image_size_trained=settings.IMAGE_SIZE,
    preloaded_model=None,
    model_load_path="",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    verbose=True,
    fake_clouds=None,
    # noise_strength=0.3,
    # alpha_image=None,
) -> Image.Image:
    """tests the model and returns a pil image.

    Rrgs:
        image_path (str): path to the image to be converted.
        model_load_path (str): path to the model to be loaded.
        image_size_trained (str): should not be touched.

    Returns:
        PIL.Image.Image if successful.
        -1 if something went wrong.
    """
    start_time = time.time()
    if not preloaded_model:
        model = load_model(
            model_load_path=model_load_path, device=device, verbose=verbose
        )
    else:
        model = preloaded_model
    # --- image preprocessing ---
    # this should match the transformations applied to your training input (cloudy_image or clear_image before normalization)
    # specifically resize and normalize.
    # your training loop normalizes inputs to [-1, 1]
    # if verbose:
    #     print(f"model: {model}")
    if verbose:
        print(f"Time to load model: {time.time() - start_time}")

    # cropped_image = crop_to_center_circle(image)
    # upper_bound = get_random_valid_coords(cropped_image, boost=150)
    # lower_bound = get_random_valid_coords(cropped_image, boost=-10)
    # fake_clouds = colorize_array(
    #     fake_clouds, lower_bound=lower_bound, upper_bound=upper_bound
    # )
    # r, g, b = fake_clouds.split()

    # fake_clouds = Image.merge("RGBA", (r, g, b, alpha_image))

    image = image.convert("RGBA")
    image = Image.alpha_composite(image, fake_clouds) 
    image = image.convert("RGB")
    if verbose:
        print(f"Time to overlay clouds: {time.time() - start_time}")
        image.show()

    preprocess = T.Compose(
        [
            T.Resize(image_size_trained),  # Use the size the model was trained on
            T.ToTensor(),  # Converts PIL image [0,255] to tensor [0,1]
            # RandomApplyTransforms(
            #     settings.IMAGE_SIZE,
            #     settings.RANDOM_APPLY_THRESHOLD,
            #     settings.NOISE_STRENGTH,
            #     noise_strength,
            # ),
            T.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
            ),  # Normalizes [0,1] to [-1,1]
        ]
    )

    if verbose:
        print(f"Loading and preprocessing image: {image}")
    try:
        if image.size != settings.IMAGE_SIZE:
            if verbose:
                print(
                    f"Warning: input size {image.size} is not {settings.IMAGE_SIZE}. Resizing . . ."
                )
                image = image.resize(settings.IMAGE_SIZE, resample=image.bilinear)
        input_tensor = preprocess(image)
        # noise = torch.rand_like(input_tensor) * (settings.NOISE_STRENGTH ** 2)
        # input_tensor = input_tensor + noise
        # input_tensor = torch.clamp(input_tensor, 0.0, 1.0)
        input_tensor = input_tensor.unsqueeze(0)  # add batch dimension (b, c, h, w)
        input_tensor = input_tensor.to(device)
        if verbose:
            print(f"Input tensor shape: {input_tensor.shape}")
    except Exception as e:
        print(f"Error processing image {image}: {e}")
        return None

    if verbose:
        print(f"Time to preprocess: {time.time() - start_time}")

    # --- inference ---
    with torch.no_grad():  # disable gradient calculations for inference
        if verbose:
            print("Running inference...")
        output_tensor = model(input_tensor)
        if verbose:
            print(
                f"output tensor shape: {output_tensor.shape}, min: {output_tensor.min():.2f}, max: {output_tensor.max():.2f}"
            )
    if verbose:
        print(f"Time to infer: {time.time() - start_time}")

    # --- postprocessing ---
    output_tensor = output_tensor.squeeze(0)

    # denormalize: model outputs are in [-1, 1] (due to tanh)
    # we need to map them back to [0, 1] for to_pil_image
    output_tensor_denorm = output_tensor * 0.5 + 0.5
    # clamp to ensure values are strictly in [0, 1] range after denormalization
    output_tensor_denorm = torch.clamp(output_tensor_denorm, 0, 1)

    # convert tensor to pil image
    output_image = TF.to_pil_image(output_tensor_denorm)
    if verbose:
        print("Inference complete. Output image generated.")
        print(f"Time to finish: {time.time() - start_time}")

    return output_image


def save_test(
    model_load_path=None,
    image_path=None,
    save_path=None,
    blend_strength=None,
    iterations=1,
    device=None,
    model_name=None,
):
    """Save testing model_load_path with image_path to save_path."""
    if not save_path:
        print("Error: no save path specified.")
        return -1
    test(
        model_load_path=model_load_path,
        image_path=image_path,
        blend_strength=blend_strength,
        iterations=iterations,
        device=device,
        model_name=model_name,
    ).save(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.path.join("testing_models", "randiv_96_checkpoint_best.pth"))
    parser.add_argument("--image", default=os.path.join("readme_images", "clouds.png"))
    parser.add_argument("--output", default="")
    parser.add_argument("--model-name", choices=get_model_names())
    parser.add_argument("--blend-strength", type=float, default=0.3)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    output_image = test(
        model_load_path=args.model,
        image_path=args.image,
        device=args.device,
        verbose=not args.quiet,
        blend_strength=args.blend_strength,
        iterations=args.iterations,
        model_name=args.model_name,
    )
    if args.output:
        output_image.save(args.output)
    else:
        output_image.show()
