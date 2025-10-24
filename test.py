#!/usr/bin/env python3

import os
import time
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import torchvision.transforms as T
from network import get_random_valid_coords, crop_to_center_circle, colorize_array

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

from network import DeepUNet, RandomApplyTransforms


def load_model(
    model_load_path=os.path.join("models", "64_checkpoint_best.pth"),
    device="cuda" if torch.cuda.is_available() else "cpu",
    verbose=True,
):
    """Loads a model, and returns it.

    Args:
        model_load_path (str): Path to the models to be loaded.
        device (str): Device to load the model on to, defaults to default device.
        verbose (bool): Prints helpful information if True.

    Returns:
        network.DeepUNet().to(device) in evaluation mode.
    """

    if verbose:
        print(f"Loading checkpoint from: {model_load_path}")
    # map_location ensures model loads correctly even if saved on GPU and loading on CPU
    checkpoint = torch.load(model_load_path, map_location=device)

    # Extract model hyperparameters from the checkpoint
    start_filters = checkpoint.get("start_filters")
    n_channels_in = checkpoint.get("in_channels")
    n_classes_out = checkpoint.get("out_channels")

    if None in [start_filters, n_channels_in, n_classes_out]:
        print(
            "Error: Checkpoint does not contain necessary model hyperparameters (start_filters, in_channels, out_channels)."
        )
        print(
            "Attempting to use defaults (3, 3, 64) but this may fail or be incorrect."
        )
        # Fallback, but ideally the checkpoint should always have these
        n_channels_in = n_channels_in if n_channels_in is not None else 3
        n_classes_out = n_classes_out if n_classes_out is not None else 3
        start_filters = start_filters if start_filters is not None else 64

    # Instantiate the model with loaded hyperparameters
    # Ensure DeepUNet is defined/imported correctly
    # from network import DeepUNet # Or however you access your model class
    model = DeepUNet(  # Make sure DeepUNet is correctly imported/defined
        n_channels_in=n_channels_in,
        n_classes_out=n_classes_out,
        start_filters=start_filters,
    )
    if verbose:
        print(
            f"Instantiated model with: in_channels={n_channels_in}, out_channels={n_classes_out}, start_filters={start_filters}"
        )

    # Handle DataParallel state_dict keys
    # If the model was saved with nn.DataParallel, keys will have 'module.' prefix.
    # We need to remove this prefix if we are not using nn.DataParallel during inference.
    state_dict = checkpoint["model_state_dict"]
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    is_data_parallel = any(key.startswith("module.") for key in state_dict.keys())

    if is_data_parallel:
        if verbose:
            print("Adjusting keys from DataParallel model.")
        for k, v in state_dict.items():
            name = k[7:] if k.startswith("module.") else k  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)
    if verbose:
        print(f"Model was trained to epoch {checkpoint['epoch']}.")

    model = model.to(device)
    model.eval()  # Set model to evaluation mode
    return model


def test(
    image_path=os.path.join("readme_images", "Randii.png"),
    model_load_path=os.path.join("models", "checkpoint_best.pth"),
    image_size_trained=settings.IMAGE_SIZE,
    preloaded_model=None,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    verbose=True,
    noise_strength=0.3,
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

    model = DeepUNet(  # Make sure DeepUNet is correctly imported/defined
        n_channels_in=3,
        n_classes_out=3,
        start_filters=32,
    )

    if not preloaded_model:
        model = load_model(
            model_load_path=model_load_path, device=device, verbose=verbose
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
                settings.NOISE_STRENGTH,
                noise_strength,
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
    output_image = TF.to_pil_image(output_tensor_denorm)
    if verbose:
        print("Inference complete. Output image generated.")

    return output_image


def test_with_ram(
    image=None,
    image_size_trained=settings.IMAGE_SIZE,
    preloaded_model=None,
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

    # model = DeepUNet(  # make sure deepunet is correctly imported/defined
    #     n_channels_in=3,
    #     n_classes_out=3,
    #     start_filters=32,
    # )

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

def save_test(model_load_path=None, image_path=None, save_path=None):
    if not save_path:
        print("Error: no save path specified.")
        return -1
    test(model_load_path=model_load_path, image_path=image_path).save(save_path)


if __name__ == "__main__":
    test(
        model_load_path=os.path.join("testing_models", "randiv_96_checkpoint_best.pth"),
        image_path=os.path.join("readme_images", "clouds.png"),
    ).show()
