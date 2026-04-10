"""Train the U-Net."""
import argparse
import os
from collections import OrderedDict
import time
import numpy
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import torchvision.transforms.functional as TF
from torchvision import transforms
import torch.optim as optim
from cloud_transform import RandomApplyTransforms
from image_datasets import ImageDataset, make_video_datasets
from model_utils import build_model, get_model_default_start_filters, get_model_names
from vgg_loss import VGGLoss

if not os.path.exists("local_settings.py"):
    print("Warning: local settings not found. Using default settings.")
    import settings
else:
    import local_settings as settings

# @TODO: separate all of this crap into many individual files
#        but i don't want to


def make_dataloaders(
    train,
    valid,
    data_dir,
    clear_transform,
    cloud_transform,
    batch_size=settings.BATCH_SIZE,
    num_workers=settings.NUM_WORKERS,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = ImageDataset(
        train,
        data_dir=data_dir,
        clear_transform=clear_transform,
        cloud_transform=cloud_transform,
    )
    valid_dataset = ImageDataset(
        valid,
        data_dir=data_dir,
        clear_transform=clear_transform,
        cloud_transform=cloud_transform,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
    )

    return train_dataloader, valid_dataloader


def hex_to_rgb(hex):
    """Take a hex string and returns the corresponding RGB tuple.

    From https://www.30secondsofcode.org/python/s/hex-to-rgb/

    Args:
        hex (str): a hex tuple, formatted AABBCC.

    Returns:
        int tuple, formatted (AA, BB, CC) but converted to the corresponding integer value.
    """
    return tuple(int(hex[i : i + 2], 16) for i in (0, 2, 4))


def parse_device_arg(device_arg):
    if device_arg is None:
        return None

    requested_device = device_arg.strip().lower()
    if requested_device in {"cpu", "cuda"}:
        return requested_device

    device_ids = [int(device_id.strip()) for device_id in requested_device.split(",") if device_id.strip()]
    if not device_ids:
        raise ValueError("--device must be 'cpu', 'cuda', or a comma-separated list of CUDA device ids.")

    return device_ids


def resolve_training_devices(device_arg):
    parsed_device = parse_device_arg(device_arg)
    if parsed_device is None:
        if settings.USE_DEVICE_IDS:
            device_ids = list(settings.DEVICE_IDS)
            return torch.device(f"cuda:{device_ids[0]}"), device_ids
        default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return default_device, None

    if parsed_device == "cpu":
        return torch.device("cpu"), None

    if parsed_device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but not available.")
        return torch.device("cuda"), None

    if not torch.cuda.is_available():
        raise ValueError("CUDA device ids were requested but CUDA is not available.")

    return torch.device(f"cuda:{parsed_device[0]}"), parsed_device


def warn_deprecated_vgg_device(vgg_device_arg):
    if vgg_device_arg is not None:
        print(
            "Warning: --vgg-device is deprecated and ignored. "
            "VGGLoss now runs on the local training device.",
            flush=True,
        )


def train_model(
    n_channels_in=3,
    n_classes_out=3,
    start_filters=None,
    data_dir=os.path.join("data", "images"),
    output_dir=settings.MODEL_SAVE_PATH,
    model_name="unetpp",
    device_arg=None,
    vgg_device_arg=None,
    batch_size=settings.BATCH_SIZE,
    num_workers=settings.NUM_WORKERS,
    num_epochs=settings.NUM_EPOCHS,
    previous_model_path=None,
    levels=5,
    debug=False,
):
    """Training loop for training a model.

    Just run python network.py. Please.
    Config this with settings.py or local_settings.py.
    All of the args should be self explanatory.

    Args:
        n_channels_in (int)
        n_classes_out (int)
        start_filters (int | None): defaults to the selected model constructor value
        data_dir (str)
        output_dir (str)
        model_name (str)
        device_arg (str | None)
        batch_size (int): configured in settings
        num_workers (int): configured in settings
        num_epochs (int): configured in settings
        previous_model_path (str)
        debug (bool): exits the loop early for displaying a sample of target and training data
    """
    device, device_ids = resolve_training_devices(device_arg)
    warn_deprecated_vgg_device(vgg_device_arg)
    vgg_device = device
    print(f"Using device: {device}")
    print(f"Using VGG device: {vgg_device}")

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"HIP version (ROCm): {torch.version.hip}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"Using device: {device}")

    # return 1

    # Dataloaders

    clear_transform = transforms.Compose(
        [
            transforms.Resize(settings.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    cloud_transform = transforms.Compose(
        [
            transforms.Resize(settings.IMAGE_SIZE),
            RandomApplyTransforms(
                settings.IMAGE_SIZE,
                settings.RANDOM_APPLY_THRESHOLD,
                settings.NOISE_STRENGTH,
            ),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    train, valid = make_video_datasets(data_dir)

    print(
        f"Using {len(train)} images for training and {len(valid)} images for validation"
    )

    train_dataloader, valid_dataloader = make_dataloaders(
        train,
        valid,
        data_dir,
        clear_transform,
        cloud_transform,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    if debug:
        print("Visualizing a sample from training data...")
        sample_inputs, sample_targets = next(iter(train_dataloader))
        # output_tensor * 0.5 + 0.5
        show_tensor_image(
            (sample_inputs[0] * 0.5 + 0.5).cpu()
        )  # Show first cloudy image in batch
        show_tensor_image(
            (sample_targets[0] * 0.5 + 0.5).cpu()
        )  # Show first clear image in batch
        return -1

    os.makedirs(output_dir, exist_ok=True)

    scaler = None
    if settings.USE_AMP and device.type == "cuda":
        scaler = torch.amp.GradScaler()
        print("Using Automatic Mixed Precision (AMP).")

    start_epoch = 0

    resolved_start_filters = start_filters
    if resolved_start_filters is None:
        resolved_start_filters = get_model_default_start_filters(model_name)

    model = build_model(model_name, n_channels_in, n_classes_out, start_filters)
    if previous_model_path is None:
        print(
            f"Initialized {model.__class__.__name__} with {levels} layers, {n_channels_in} channels in, {n_classes_out} classes out, and {resolved_start_filters} start filters."
        )
    else:
        if not os.path.exists(previous_model_path):
            raise ValueError(
                f"Error: {previous_model_path} is not a valid path to a previous model."
            )
            return -1
        checkpoint = torch.load(previous_model_path)
        resolved_start_filters = checkpoint["start_filters"]
        in_channels = checkpoint["in_channels"]
        out_channels = checkpoint["out_channels"]
        checkpoint_model_name = checkpoint.get("model_name", model_name)
        start_epoch = checkpoint["epoch"]
        model = build_model(
            checkpoint_model_name,
            in_channels,
            out_channels,
            resolved_start_filters,
        )
        loaded_state_dict = checkpoint["model_state_dict"]

        new_state_dict = OrderedDict()
        is_data_parallel = False
        for k, v in loaded_state_dict.items():
            if k.startswith("module."):
                is_data_parallel = True
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            else:
                new_state_dict[k] = v  # Non-DataParallel checkpoint or already stripped

        if is_data_parallel:
            print(
                "Checkpoint was saved from a DataParallel model. Stripping 'module.' prefix."
            )

        model.load_state_dict(new_state_dict)

        # model.load_state_dict(checkpoint["model_state_dict"])
        print(
            f"Loading {model.__class__.__name__} from {previous_model_path} with {in_channels} channels in, {out_channels} classes out, and {resolved_start_filters} start filters."
        )

    if device_ids and len(device_ids) > 1:
        print(
            f"Wrapping model with nn.DataParallel for devices {device_ids}."
        )
        model = model.to(device)
        model = nn.DataParallel(
            model,
            device_ids=device_ids,
            output_device=device_ids[0],
        )
    else:
        model = model.to(device)

    criterion = nn.L1Loss()
    vgg_loss_crit = VGGLoss().to(vgg_device)
    L1_WEIGHT = 0
    VGG_WEIGHT = 1

    optimizer = optim.Adam(model.parameters(), lr=settings.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=settings.STEP_SIZE,
        factor=settings.GAMMA,
        mode="min",
    )
    best_val_loss = float("inf")

    # training loop
    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        num_batches_processed = 0
        print(f"\n--- Epoch {epoch}/{num_epochs} [Train] ---")
        batch_start_time = time.time()

        for i, (inputs, targets) in enumerate(train_dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            if scaler:  # AMP
                with torch.autocast(device_type='cuda'):
                    outputs = model(inputs)
                    l1_loss = criterion(outputs, targets)
                    outputs = (outputs + 1.0) / 2.0
                    targets = (targets + 1.0) / 2.0
                    vgg_loss = vgg_loss_crit(
                        outputs,
                        vgg_loss_crit.get_features(targets),
                        target_is_features=True,
                    )
                loss = l1_loss * L1_WEIGHT + vgg_loss.to(l1_loss.device) * VGG_WEIGHT
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:  # No AMP
                outputs = model(inputs)
                l1_loss = criterion(outputs, targets)
                outputs = (outputs + 1.0) / 2.0
                targets = (targets + 1.0) / 2.0
                vgg_loss = vgg_loss_crit(
                    outputs,
                    vgg_loss_crit.get_features(targets),
                    target_is_features=True,
                )
                loss = l1_loss * L1_WEIGHT + vgg_loss.to(l1_loss.device) * VGG_WEIGHT
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            running_loss += loss.item()
            num_batches_processed = i + 1

            if (i + 1) % 20 == 0 or (i + 1) == len(train_dataloader):
                batch_time = time.time() - batch_start_time
                print(
                    f"  Batch {i+1}/{len(train_dataloader)} | Train Loss: {loss.item():.4f} | Time: {batch_time:.2f}s"
                )
            if (
                (i + 1) >= settings.MAX_EPOCH_TRAIN_SIZE
                and settings.MAX_EPOCH_TRAIN_SIZE != -1
            ):
                break

        epoch_train_loss = running_loss / num_batches_processed
        print(f"Epoch {epoch+1} [Train] Avg Loss: {epoch_train_loss:.4f}")

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in valid_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                if scaler:  # AMP for validation
                    with torch.autocast(device_type='cuda'):
                        outputs = model(inputs)
                        l1_loss = criterion(outputs, targets)
                        outputs = (outputs + 1.0) / 2.0
                        targets = (targets + 1.0) / 2.0
                        vgg_loss = vgg_loss_crit(
                            outputs,
                            vgg_loss_crit.get_features(targets),
                            target_is_features=True,
                        )
                        loss = (
                            l1_loss * L1_WEIGHT
                            + vgg_loss.to(l1_loss.device) * VGG_WEIGHT
                        )
                else:
                    outputs = model(inputs)
                    l1_loss = criterion(outputs, targets)
                    outputs = (outputs + 1.0) / 2.0
                    targets = (targets + 1.0) / 2.0
                    vgg_loss = vgg_loss_crit(
                        outputs,
                        vgg_loss_crit.get_features(targets),
                        target_is_features=True,
                    )
                    loss = (
                        l1_loss * L1_WEIGHT + vgg_loss.to(l1_loss.device) * VGG_WEIGHT
                    )

                running_val_loss += loss.item()

        epoch_val_loss = running_val_loss / len(valid_dataloader)
        print(f"Epoch {epoch+1} [Val]   Avg Loss: {epoch_val_loss:.4f}")

        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch Duration: {epoch_duration:.2f}s")

        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(epoch_val_loss)
        elif scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Current Learning Rate: {current_lr}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            checkpoint_name = "checkpoint_best.pth"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": model_name,
                    "start_filters": resolved_start_filters,
                    "in_channels": n_channels_in,
                    "out_channels": n_classes_out,
                    "epoch": epoch,
                },
                os.path.join(output_dir, checkpoint_name),
            )
            print(
                f"Model improved. Saved to {output_dir} (Val Loss: {best_val_loss:.4f})"
            )

        if epoch % settings.EPOCH_SAVE_INTERVAL == 0:

            checkpoint_name = f"checkpoint_epoch_{epoch}.pth"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": model_name,
                    "start_filters": resolved_start_filters,
                    "in_channels": n_channels_in,
                    "out_channels": n_classes_out,
                    "epoch": epoch,
                },
                os.path.join(output_dir, checkpoint_name),
            )
            print(
                f"Reached a checkpoint. Saved to {output_dir} (Val Loss: {best_val_loss:.4f})"
            )

    print("\n--- Training Finished ---")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Best model saved at: {output_dir}")


def show_tensor_image(tensor):
    image = tensor.detach().cpu()
    image = TF.to_pil_image(image)
    image.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.path.join("png_split_training_images"))
    parser.add_argument("--output-dir", default=settings.MODEL_SAVE_PATH)
    parser.add_argument("--model", default="unet", choices=get_model_names())
    parser.add_argument("--device")
    parser.add_argument(
        "--vgg-device",
        help="Deprecated and ignored. VGGLoss now runs on the local training device.",
    )
    parser.add_argument("--batch-size", type=int, default=settings.BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=settings.NUM_WORKERS)
    parser.add_argument("--epochs", type=int, default=settings.NUM_EPOCHS)
    parser.add_argument("--debug", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--n-channels-in", type=int, default=3)
    parser.add_argument("--n-classes-out", type=int, default=3)
    parser.add_argument("--previous-model-path")
    parser.add_argument("--start-filters", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_name=args.model,
        device_arg=args.device,
        vgg_device_arg=args.vgg_device,
        batch_size=args.batch_size,
        num_workers=args.workers,
        n_channels_in=args.n_channels_in,
        n_classes_out=args.n_classes_out,
        num_epochs=args.epochs,
        previous_model_path=args.previous_model_path,
        start_filters=args.start_filters,
        debug=args.debug,
        levels=5,
    )

#  LocalWords:  ROCm
