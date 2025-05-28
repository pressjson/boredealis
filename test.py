#!/usr/bin/env python3

import os
import network
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import torchvision.transforms as T

if not os.path.exists("local_settings.py"):
    print("Warning: local settings not found. Using default settings.")
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
# that llm is actually genius


def test(
    # image_path=os.path.join("test", "images", "02032021_221508_0001.png"),
    image_path="Randii.png",
    # image_path=os.path.join("test", "images", "02032021_221508_0001.png"),

    model_load_path=os.path.join("models", "checkpoint_best.pth"),
    image_size_trained=settings.IMAGE_SIZE,
    # image_size_trained=(608, 608),
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(model_load_path):
        print(f"Error: Model checkpoint not found at {model_load_path}")
        return -1
    if not os.path.exists(image_path):
        print(f"Error: Test image not found at {image_path}")
        return -1

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
    model = network.DeepUNet(  # Make sure DeepUNet is correctly imported/defined
        n_channels_in=n_channels_in,
        n_classes_out=n_classes_out,
        start_filters=start_filters,
    )
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
        print("Adjusting keys from DataParallel model.")
        for k, v in state_dict.items():
            name = k[7:] if k.startswith("module.") else k  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()  # Set model to evaluation mode

    # --- Image Preprocessing ---
    # This should match the transformations applied to your training input (cloudy_image or clear_image before normalization)
    # Specifically Resize and Normalize.
    # Your training loop normalizes inputs to [-1, 1]
    preprocess = T.Compose(
        [
            T.Resize(image_size_trained),  # Use the size the model was trained on
            T.ToTensor(),  # Converts PIL image [0,255] to tensor [0,1]
            T.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
            ),  # Normalizes [0,1] to [-1,1]
        ]
    )

    print(f"Loading and preprocessing image: {image_path}")
    try:
        image = Image.open(image_path).convert("RGB")
        input_tensor = preprocess(image)
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension (B, C, H, W)
        input_tensor = input_tensor.to(device)
        print(f"Input tensor shape: {input_tensor.shape}")
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

    # --- Inference ---
    with torch.no_grad():  # Disable gradient calculations for inference
        print("Running inference...")
        output_tensor = model(input_tensor)
        print(
            f"Output tensor shape: {output_tensor.shape}, min: {output_tensor.min():.2f}, max: {output_tensor.max():.2f}"
        )

    # --- Postprocessing ---
    output_tensor = output_tensor.squeeze(
        0
    ).cpu()  # Remove batch dimension and move to CPU

    # Denormalize: model outputs are in [-1, 1] (due to Tanh)
    # We need to map them back to [0, 1] for to_pil_image
    output_tensor_denorm = output_tensor * 0.5 + 0.5
    # Clamp to ensure values are strictly in [0, 1] range after denormalization
    output_tensor_denorm = torch.clamp(output_tensor_denorm, 0, 1)

    # Convert tensor to PIL Image
    output_image = TF.to_pil_image(output_tensor_denorm)
    print("Inference complete. Output image generated.")

    return output_image


if __name__ == "__main__":
    test(
        # model_load_path="32_filters_models/checkpoint_epoch_20.pth"
        model_load_path="models/checkpoint_best.pth"
    ).show()
