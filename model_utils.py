"""Shared model discovery and construction helpers."""

from collections import OrderedDict
import inspect
from pathlib import Path

import models
import torch


MODEL_ALIASES = {
    "attentionunet": "AttentionUNet",
    "attention_unet": "AttentionUNet",
    "deeplabv3plus": "DeepLabV3Plus",
    "deeplab_v3_plus": "DeepLabV3Plus",
    "deepunet": "DeepUNet",
    "deep_unet": "DeepUNet",
    "unet": "DeepUNet",
    "edsr": "EDSR",
    "fpnunet": "FPNUNet",
    "fpn_unet": "FPNUNet",
    "nafnet": "NAFNet",
    "rdn": "RDN",
    "resunet": "ResUNet",
    "res_unet": "ResUNet",
    "restormer": "Restormer",
    "unetpp": "UNetPlusPlus",
    "unetplusplus": "UNetPlusPlus",
    "unet_plus_plus": "UNetPlusPlus",
}


def get_model_builders():
    builders = {}
    for path in Path("models").glob("*_model.py"):
        model_key = path.name.removesuffix("_model.py").replace("_", "")
        export_name = MODEL_ALIASES.get(model_key)
        if export_name is None:
            matching_exports = [
                candidate
                for candidate in models.__all__
                if candidate.lower() == model_key
            ]
            if not matching_exports:
                continue
            export_name = matching_exports[0]
        builders[model_key] = getattr(models, export_name)

    for alias, export_name in MODEL_ALIASES.items():
        builders[alias] = getattr(models, export_name)

    return builders


def get_model_names():
    return sorted(get_model_builders())


def get_model_default_start_filters(model_name):
    model_builders = get_model_builders()
    model_key = model_name.lower()
    if model_key not in model_builders:
        raise ValueError(
            f"Unsupported model '{model_name}'. Choose from: {', '.join(sorted(model_builders))}."
        )

    signature = inspect.signature(model_builders[model_key].__init__)
    return signature.parameters["start_filters"].default


def build_model(model_name, n_channels_in, n_classes_out, start_filters=None):
    model_builders = get_model_builders()
    model_key = model_name.lower()
    if model_key not in model_builders:
        raise ValueError(
            f"Unsupported model '{model_name}'. Choose from: {', '.join(sorted(model_builders))}."
        )

    kwargs = {
        "n_channels_in": n_channels_in,
        "n_classes_out": n_classes_out,
    }
    if start_filters is not None:
        kwargs["start_filters"] = start_filters

    return model_builders[model_key](**kwargs)


def resolve_model_name(checkpoint, model_name=None):
    if model_name:
        return model_name
    return checkpoint.get("model_name", "unet")


def load_model(
    model_load_path,
    device="cpu",
    verbose=True,
    model_name=None,
):
    if verbose:
        print(f"Loading checkpoint from: {model_load_path}")

    checkpoint = torch.load(model_load_path, map_location=device)

    start_filters = checkpoint.get("start_filters")
    n_channels_in = checkpoint.get("in_channels")
    n_classes_out = checkpoint.get("out_channels")
    resolved_model_name = resolve_model_name(checkpoint, model_name=model_name)

    if None in [start_filters, n_channels_in, n_classes_out]:
        print(
            "Error: Checkpoint does not contain necessary model hyperparameters (start_filters, in_channels, out_channels)."
        )
        print(
            "Attempting to use saved values plus model constructor defaults where available."
        )
        n_channels_in = n_channels_in if n_channels_in is not None else 3
        n_classes_out = n_classes_out if n_classes_out is not None else 3
        start_filters = (
            start_filters
            if start_filters is not None
            else get_model_default_start_filters(resolved_model_name)
        )

    model = build_model(
        resolved_model_name,
        n_channels_in=n_channels_in,
        n_classes_out=n_classes_out,
        start_filters=start_filters,
    )
    if verbose:
        print(
            f"Instantiated {model.__class__.__name__} from model_name={resolved_model_name} with: in_channels={n_channels_in}, out_channels={n_classes_out}, start_filters={start_filters}"
        )

    state_dict = checkpoint["model_state_dict"]
    new_state_dict = OrderedDict()
    is_data_parallel = any(key.startswith("module.") for key in state_dict.keys())

    if is_data_parallel:
        if verbose:
            print("Adjusting keys from DataParallel model.")
        for key, value in state_dict.items():
            name = key[7:] if key.startswith("module.") else key
            new_state_dict[name] = value
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)

    if verbose and "epoch" in checkpoint:
        print(f"Model was trained to epoch {checkpoint['epoch']}.")

    model = model.to(device)
    model.eval()
    return model


def update_checkpoint_model_name(checkpoint_path, model_name, output_path=None, force=False):
    model_key = model_name.lower()
    if model_key not in get_model_builders():
        raise ValueError(
            f"Unsupported model '{model_name}'. Choose from: {', '.join(get_model_names())}."
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    existing_model_name = checkpoint.get("model_name")
    if existing_model_name and existing_model_name.lower() != model_key and not force:
        raise ValueError(
            f"Checkpoint already has model_name='{existing_model_name}'. Pass force=True to replace it."
        )

    checkpoint["model_name"] = model_key
    save_path = output_path or checkpoint_path
    torch.save(checkpoint, save_path)
    return save_path


def update_checkpoint_model_name_in_directory(
    directory_path,
    model_name,
    recursive=False,
    output_suffix=None,
    force=False,
):
    directory = Path(directory_path)
    if not directory.is_dir():
        raise ValueError(f"Directory does not exist: {directory_path}")

    pattern = "**/*.pth" if recursive else "*.pth"
    results = []
    for checkpoint_path in sorted(directory.glob(pattern)):
        output_path = None
        if output_suffix:
            output_path = checkpoint_path.with_name(
                f"{checkpoint_path.stem}{output_suffix}{checkpoint_path.suffix}"
            )

        saved_path = update_checkpoint_model_name(
            checkpoint_path,
            model_name,
            output_path=output_path,
            force=force,
        )
        results.append({
            "input_path": str(checkpoint_path),
            "output_path": str(saved_path),
        })

    return results
