"""Train models with DistributedDataParallel."""

import argparse
import contextlib
import gc
import os
import socket
import sys
import time
from collections import OrderedDict


MASK_ENV_VAR = "BOREDEALIS_TRAIN_DEVICES_MASKED"


def _get_cli_arg_value(flag_name):
    for index, arg in enumerate(sys.argv[1:], start=1):
        if arg == flag_name:
            if index + 1 < len(sys.argv):
                return sys.argv[index + 1]
            return None
        if arg.startswith(f"{flag_name}="):
            return arg.split("=", 1)[1]
    return None


def _set_cli_arg_value(flag_name, value):
    value = str(value)
    for index, arg in enumerate(sys.argv[1:], start=1):
        if arg == flag_name:
            if index + 1 < len(sys.argv):
                sys.argv[index + 1] = value
            return
        if arg.startswith(f"{flag_name}="):
            sys.argv[index] = f"{flag_name}={value}"
            return


def _parse_global_cuda_id(device_arg):
    if device_arg is None:
        return None

    normalized = device_arg.strip().lower()
    if normalized in {"", "cpu", "cuda"}:
        return None
    if normalized.startswith("cuda:"):
        normalized = normalized.split(":", 1)[1]
    if "," in normalized:
        return None
    return int(normalized)


def _parse_global_cuda_ids(device_arg):
    if device_arg is None:
        return None

    normalized = device_arg.strip().lower()
    if normalized in {"", "cpu", "cuda"}:
        return None
    if normalized.startswith("cuda:"):
        return [int(normalized.split(":", 1)[1])]
    return [int(device_id.strip()) for device_id in normalized.split(",") if device_id.strip()]


def _configure_visible_devices():
    if os.environ.get(MASK_ENV_VAR) == "1":
        return

    train_device_arg = _get_cli_arg_value("--device")
    train_device_ids = _parse_global_cuda_ids(train_device_arg)
    if train_device_ids is None:
        return

    visible_device_ids = list(train_device_ids)

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(device_id) for device_id in visible_device_ids)

    remapped_train_ids = [visible_device_ids.index(device_id) for device_id in train_device_ids]
    _set_cli_arg_value("--device", ",".join(str(device_id) for device_id in remapped_train_ids))

    os.environ[MASK_ENV_VAR] = "1"


_configure_visible_devices()

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from cloud_transform import RandomApplyTransforms
import image_datasets
import image_datasets_ram
from model_utils import build_model, get_model_default_start_filters, get_model_names
from vgg_loss import VGGLoss

if not os.path.exists("local_settings.py"):
    print("Warning: local settings not found. Using default settings.")
    import settings
else:
    import local_settings as settings


def make_dataloaders(
    train,
    valid,
    data_dir,
    clear_transform,
    cloud_transform,
    cache_ram=False,
    batch_size=settings.BATCH_SIZE,
    num_workers=settings.NUM_WORKERS,
    rank=0,
    world_size=1,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_module = image_datasets_ram if cache_ram else image_datasets
    train_dataset = dataset_module.ImageDataset(
        train,
        data_dir=data_dir,
        clear_transform=clear_transform,
        cloud_transform=cloud_transform,
    )
    valid_dataset = dataset_module.ImageDataset(
        valid,
        data_dir=data_dir,
        clear_transform=clear_transform,
        cloud_transform=cloud_transform,
    )

    train_sampler = None
    valid_sampler = None
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
        valid_sampler = DistributedSampler(
            valid_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        pin_memory=(device.type == "cuda"),
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        sampler=valid_sampler,
        pin_memory=(device.type == "cuda"),
    )

    return train_dataloader, valid_dataloader, train_sampler, valid_sampler


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
            return list(settings.DEVICE_IDS)
        if torch.cuda.is_available():
            return [0]
        return ["cpu"]

    if parsed_device == "cpu":
        return ["cpu"]

    if parsed_device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but not available.")
        return [0]

    if not torch.cuda.is_available():
        raise ValueError("CUDA device ids were requested but CUDA is not available.")

    return parsed_device


def warn_deprecated_vgg_device(rank, vgg_device_arg):
    if vgg_device_arg is not None and is_rank_zero(rank):
        print(
            "Warning: --vgg-device is deprecated and ignored. "
            "Each training rank now keeps its own VGGLoss on the local training device.",
            flush=True,
        )


def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def is_rank_zero(rank):
    return rank == 0


def rank_print(rank, message):
    if is_rank_zero(rank):
        print(message, flush=True)


def rank_debug(rank, message, enabled=False):
    if enabled:
        print(f"[rank {rank}] {message}", file=sys.stderr, flush=True)


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def load_checkpoint_state_dict(model, loaded_state_dict, rank):
    new_state_dict = OrderedDict()
    had_parallel_prefix = False
    for key, value in loaded_state_dict.items():
        if key.startswith("module."):
            had_parallel_prefix = True
            new_state_dict[key[7:]] = value
        else:
            new_state_dict[key] = value

    if had_parallel_prefix:
        rank_print(rank, "Checkpoint was saved from a parallel model. Stripping 'module.' prefix.")

    model.load_state_dict(new_state_dict)


def reduce_mean(value, device, world_size):
    tensor = torch.tensor([value], device=device, dtype=torch.float64)
    if world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= world_size
    return tensor.item()


def reduce_sum(value, device, world_size):
    tensor = torch.tensor([value], device=device, dtype=torch.float64)
    if world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.item()


def is_oom_error(exc):
    if isinstance(exc, torch.OutOfMemoryError):
        return True

    message = str(exc).lower()
    return "out of memory" in message or "cuda error: out of memory" in message


def clear_device_memory(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)


def _try_probe_batch_size(batch_size, model, criterion, vgg_loss_crit, device, use_amp, n_channels_in, n_classes_out):
    inputs = torch.randn(
        batch_size,
        n_channels_in,
        settings.IMAGE_SIZE[0],
        settings.IMAGE_SIZE[1],
        device=device,
    )
    targets = torch.randn(
        batch_size,
        n_classes_out,
        settings.IMAGE_SIZE[0],
        settings.IMAGE_SIZE[1],
        device=device,
    )

    model.zero_grad(set_to_none=True)
    autocast_context = (
        torch.autocast(device_type="cuda")
        if use_amp and device.type == "cuda"
        else contextlib.nullcontext()
    )

    with autocast_context:
        outputs = model(inputs)
        l1_loss = criterion(outputs, targets)
        outputs = (outputs + 1.0) / 2.0
        targets = (targets + 1.0) / 2.0
        vgg_loss = vgg_loss_crit(
            outputs,
            vgg_loss_crit.get_features(targets),
            target_is_features=True,
        )
        loss = l1_loss + vgg_loss.to(l1_loss.device)

    loss.backward()


def auto_resolve_batch_size(rank, world_size, model, criterion, vgg_loss_crit, device, args):
    if args.batch_size != -1:
        return args.batch_size

    use_amp = settings.USE_AMP and device.type == "cuda"
    last_success = 0
    first_failure = None
    candidate = 1
    hard_cap = 4096

    while candidate <= hard_cap:
        clear_device_memory(device)
        try:
            _try_probe_batch_size(
                candidate,
                model,
                criterion,
                vgg_loss_crit,
                device,
                use_amp,
                args.n_channels_in,
                args.n_classes_out,
            )
        except RuntimeError as exc:
            if not is_oom_error(exc):
                raise
            first_failure = candidate
            clear_device_memory(device)
            break

        last_success = candidate
        candidate *= 2

    if first_failure is None:
        first_failure = candidate

    low = last_success + 1
    high = first_failure - 1
    while low <= high:
        mid = (low + high) // 2
        clear_device_memory(device)
        try:
            _try_probe_batch_size(
                mid,
                model,
                criterion,
                vgg_loss_crit,
                device,
                use_amp,
                args.n_channels_in,
                args.n_classes_out,
            )
        except RuntimeError as exc:
            if not is_oom_error(exc):
                raise
            high = mid - 1
            clear_device_memory(device)
            continue

        last_success = mid
        low = mid + 1

    if last_success < 1:
        raise RuntimeError("Unable to fit batch size 1 during auto batch-size probing.")

    safe_batch_size = max(1, int(last_success * 0.85))
    local_batch_size = torch.tensor([safe_batch_size], device=device, dtype=torch.int64)
    local_max_batch_size = torch.tensor([last_success], device=device, dtype=torch.int64)
    if world_size > 1:
        dist.all_reduce(local_batch_size, op=dist.ReduceOp.MIN)
        dist.all_reduce(local_max_batch_size, op=dist.ReduceOp.MIN)

    resolved_batch_size = int(local_batch_size.item())
    resolved_max_batch_size = int(local_max_batch_size.item())
    rank_print(
        rank,
        f"Auto batch size selected {resolved_batch_size} per rank (minimum probed max {resolved_max_batch_size}, 15% safety margin).",
    )
    clear_device_memory(device)
    return resolved_batch_size


def train_worker(rank, world_size, device_ids, args, master_port):
    using_cuda = device_ids != ["cpu"]
    rank_debug(rank, f"starting worker with device_ids={device_ids}", args.debug)
    if using_cuda:
        device = torch.device(f"cuda:{device_ids[rank]}")
        torch.cuda.set_device(device)
        rank_debug(rank, f"bound to device {device}", args.debug)
    else:
        device = torch.device("cpu")

    if world_size > 1:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(master_port)
        backend = args.backend
        init_kwargs = {
            "backend": backend,
            "rank": rank,
            "world_size": world_size,
        }
        if using_cuda and backend == "nccl":
            init_kwargs["device_id"] = device.index
        rank_debug(rank, f"initializing process group with {init_kwargs}", args.debug)
        dist.init_process_group(**init_kwargs)
        rank_debug(rank, "initialized process group", args.debug)

    warn_deprecated_vgg_device(rank, args.vgg_device)
    vgg_device = device
    rank_print(rank, f"Using device: {device}")
    rank_print(rank, f"Using VGG device: {vgg_device}")
    rank_print(rank, f"PyTorch version: {torch.__version__}")
    rank_print(rank, f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available() and is_rank_zero(rank):
        print(f"CUDA version: {torch.version.cuda}")
        print(f"HIP version (ROCm): {torch.version.hip}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

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

    train, valid = image_datasets.make_video_datasets(args.data_dir)
    if args.cache_ram and is_rank_zero(rank):
        print("Using RAM-cached image dataset.")
    rank_print(
        rank,
        f"Using {len(train)} images for training and {len(valid)} images for validation",
    )

    if is_rank_zero(rank):
        os.makedirs(args.output_dir, exist_ok=True)
    if world_size > 1:
        if using_cuda:
            rank_debug(rank, f"entering pre-train barrier on cuda:{device.index}", args.debug)
            dist.barrier(device_ids=[device.index])
        else:
            dist.barrier()
        rank_debug(rank, "passed pre-train barrier", args.debug)

    scaler = None
    if settings.USE_AMP and device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")
        rank_print(rank, "Using Automatic Mixed Precision (AMP).")

    start_epoch = 0
    resolved_start_filters = args.start_filters
    if resolved_start_filters is None:
        resolved_start_filters = get_model_default_start_filters(args.model)

    model = build_model(args.model, args.n_channels_in, args.n_classes_out, args.start_filters)
    rank_debug(rank, f"built model {model.__class__.__name__}", args.debug)
    if args.previous_model_path is None:
        rank_print(
            rank,
            f"Initialized {model.__class__.__name__} with {args.levels} layers, {args.n_channels_in} channels in, {args.n_classes_out} classes out, and {resolved_start_filters} start filters.",
        )
    else:
        if not os.path.exists(args.previous_model_path):
            raise ValueError(
                f"Error: {args.previous_model_path} is not a valid path to a previous model."
            )
        checkpoint = torch.load(args.previous_model_path, map_location="cpu")
        resolved_start_filters = checkpoint["start_filters"]
        in_channels = checkpoint["in_channels"]
        out_channels = checkpoint["out_channels"]
        checkpoint_model_name = checkpoint.get("model_name", args.model)
        start_epoch = checkpoint["epoch"]
        model = build_model(
            checkpoint_model_name,
            in_channels,
            out_channels,
            resolved_start_filters,
        )
        load_checkpoint_state_dict(model, checkpoint["model_state_dict"], rank)
        rank_print(
            rank,
            f"Loading {model.__class__.__name__} from {args.previous_model_path} with {in_channels} channels in, {out_channels} classes out, and {resolved_start_filters} start filters.",
        )

    model = model.to(device)
    rank_debug(rank, f"moved model to {device}", args.debug)

    criterion = nn.L1Loss()
    vgg_loss_crit = VGGLoss().to(vgg_device)
    l1_weight = 0
    vgg_weight = 1

    resolved_batch_size = auto_resolve_batch_size(
        rank,
        world_size,
        model,
        criterion,
        vgg_loss_crit,
        device,
        args,
    )

    if world_size > 1:
        ddp_device_ids = [device.index] if device.type == "cuda" else None
        rank_debug(rank, f"wrapping model with DDP device_ids={ddp_device_ids}", args.debug)
        model = DDP(model, device_ids=ddp_device_ids, output_device=device.index if device.type == "cuda" else None)
        rank_print(rank, f"Wrapping model with DistributedDataParallel for devices {device_ids}.")
        rank_debug(rank, "wrapped model with DDP", args.debug)

    train_dataloader, valid_dataloader, train_sampler, _ = make_dataloaders(
        train,
        valid,
        args.data_dir,
        clear_transform,
        cloud_transform,
        cache_ram=args.cache_ram,
        batch_size=resolved_batch_size,
        num_workers=args.workers,
        rank=rank,
        world_size=world_size,
    )

    if args.debug and is_rank_zero(rank):
        print("Visualizing a sample from training data...")
        sample_inputs, sample_targets = next(iter(train_dataloader))
        show_tensor_image((sample_inputs[0] * 0.5 + 0.5).cpu())
        show_tensor_image((sample_targets[0] * 0.5 + 0.5).cpu())
        if world_size > 1:
            dist.destroy_process_group()
        return

    optimizer = optim.Adam(model.parameters(), lr=settings.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=settings.STEP_SIZE,
        factor=settings.GAMMA,
        mode="min",
    )
    best_val_loss = float("inf")

    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        running_loss = 0.0
        num_batches_processed = 0
        rank_print(rank, f"\n--- Epoch {epoch}/{args.epochs} [Train] ---")
        batch_start_time = time.time()

        for i, (inputs, targets) in enumerate(train_dataloader):
            inputs = inputs.to(device, non_blocking=device.type == "cuda")
            targets = targets.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad()
            autocast_context = (
                torch.autocast(device_type="cuda")
                if scaler is not None
                else contextlib.nullcontext()
            )

            with autocast_context:
                outputs = model(inputs)
                l1_loss = criterion(outputs, targets)
                outputs = (outputs + 1.0) / 2.0
                targets = (targets + 1.0) / 2.0
                vgg_loss = vgg_loss_crit(
                    outputs,
                    vgg_loss_crit.get_features(targets),
                    target_is_features=True,
                )
                loss = l1_loss * l1_weight + vgg_loss.to(l1_loss.device) * vgg_weight

            if scaler is not None:
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            running_loss += loss.item()
            num_batches_processed = i + 1

            if is_rank_zero(rank) and ((i + 1) % 20 == 0 or (i + 1) == len(train_dataloader)):
                batch_time = time.time() - batch_start_time
                print(
                    f"  Batch {i+1}/{len(train_dataloader)} | Train Loss: {loss.item():.4f} | Time: {batch_time:.2f}s"
                )
            if (i + 1) >= settings.MAX_EPOCH_TRAIN_SIZE and settings.MAX_EPOCH_TRAIN_SIZE != -1:
                break

        train_loss_sum = reduce_sum(running_loss, device, world_size)
        train_batch_sum = reduce_sum(num_batches_processed, device, world_size)
        epoch_train_loss = train_loss_sum / max(train_batch_sum, 1.0)
        rank_print(rank, f"Epoch {epoch+1} [Train] Avg Loss: {epoch_train_loss:.4f}")

        model.eval()
        running_val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for inputs, targets in valid_dataloader:
                inputs = inputs.to(device, non_blocking=device.type == "cuda")
                targets = targets.to(device, non_blocking=device.type == "cuda")
                autocast_context = (
                    torch.autocast(device_type="cuda")
                    if scaler is not None
                    else contextlib.nullcontext()
                )
                with autocast_context:
                    outputs = model(inputs)
                    l1_loss = criterion(outputs, targets)
                    outputs = (outputs + 1.0) / 2.0
                    targets = (targets + 1.0) / 2.0
                    vgg_loss = vgg_loss_crit(
                        outputs,
                        vgg_loss_crit.get_features(targets),
                        target_is_features=True,
                    )
                    loss = l1_loss * l1_weight + vgg_loss.to(l1_loss.device) * vgg_weight

                running_val_loss += loss.item()
                val_batches += 1

        val_loss_sum = reduce_sum(running_val_loss, device, world_size)
        val_batch_sum = reduce_sum(val_batches, device, world_size)
        epoch_val_loss = val_loss_sum / max(val_batch_sum, 1.0)
        rank_print(rank, f"Epoch {epoch+1} [Val]   Avg Loss: {epoch_val_loss:.4f}")

        epoch_duration = time.time() - epoch_start_time
        rank_print(rank, f"Epoch Duration: {epoch_duration:.2f}s")

        scheduler.step(epoch_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        rank_print(rank, f"Current Learning Rate: {current_lr}")

        raw_model = unwrap_model(model)
        if is_rank_zero(rank) and epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            checkpoint_name = "checkpoint_best.pth"
            torch.save(
                {
                    "model_state_dict": raw_model.state_dict(),
                    "model_name": args.model,
                    "start_filters": resolved_start_filters,
                    "in_channels": args.n_channels_in,
                    "out_channels": args.n_classes_out,
                    "epoch": epoch,
                },
                os.path.join(args.output_dir, checkpoint_name),
            )
            print(f"Model improved. Saved to {args.output_dir} (Val Loss: {best_val_loss:.4f})")

        if is_rank_zero(rank) and epoch % settings.EPOCH_SAVE_INTERVAL == 0:
            checkpoint_name = f"checkpoint_epoch_{epoch}.pth"
            torch.save(
                {
                    "model_state_dict": raw_model.state_dict(),
                    "model_name": args.model,
                    "start_filters": resolved_start_filters,
                    "in_channels": args.n_channels_in,
                    "out_channels": args.n_classes_out,
                    "epoch": epoch,
                },
                os.path.join(args.output_dir, checkpoint_name),
            )
            print(f"Reached a checkpoint. Saved to {args.output_dir} (Val Loss: {best_val_loss:.4f})")

    rank_print(rank, "\n--- Training Finished ---")
    rank_print(rank, f"Best Validation Loss: {best_val_loss:.4f}")
    rank_print(rank, f"Best model saved at: {args.output_dir}")

    if world_size > 1:
        if using_cuda:
            dist.barrier(device_ids=[device.index])
        else:
            dist.barrier()
        dist.destroy_process_group()


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
    backend=None,
    cache_ram=False,
    debug=False,
):
    device_ids = resolve_training_devices(device_arg)
    resolved_backend = backend
    if resolved_backend is None:
        resolved_backend = "nccl" if device_ids != ["cpu"] else "gloo"
    args = argparse.Namespace(
        n_channels_in=n_channels_in,
        n_classes_out=n_classes_out,
        start_filters=start_filters,
        data_dir=data_dir,
        output_dir=output_dir,
        model=model_name,
        device=device_arg,
        vgg_device=vgg_device_arg,
        batch_size=batch_size,
        workers=num_workers,
        epochs=num_epochs,
        previous_model_path=previous_model_path,
        levels=levels,
        backend=resolved_backend,
        cache_ram=cache_ram,
        debug=debug,
    )

    if device_ids == ["cpu"] or len(device_ids) == 1:
        train_worker(0, 1, device_ids, args, None)
        return

    if batch_size != -1 and batch_size < len(device_ids):
        print(
            f"Warning: batch size {batch_size} is smaller than world size {len(device_ids)}. Some ranks may receive empty batches."
        )

    master_port = get_free_port()
    mp.spawn(
        train_worker,
        args=(len(device_ids), device_ids, args, master_port),
        nprocs=len(device_ids),
        join=True,
    )


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
        help="Deprecated and ignored. Each rank uses its local training device for VGGLoss.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=settings.BATCH_SIZE,
        help="Per-rank batch size. Use -1 to auto-probe the largest safe size with a 15%% safety margin.",
    )
    parser.add_argument("--workers", type=int, default=settings.NUM_WORKERS)
    parser.add_argument("--epochs", type=int, default=settings.NUM_EPOCHS)
    parser.add_argument("--debug", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--n-channels-in", type=int, default=3)
    parser.add_argument("--n-classes-out", type=int, default=3)
    parser.add_argument("--previous-model-path")
    parser.add_argument("--start-filters", type=int)
    parser.add_argument("--backend", choices=["nccl", "gloo"])
    parser.add_argument("--cache-ram", default=False, action=argparse.BooleanOptionalAction)
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
        backend=args.backend,
        cache_ram=args.cache_ram,
        debug=args.debug,
        levels=5,
    )
