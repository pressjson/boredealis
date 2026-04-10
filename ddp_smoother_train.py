#!/usr/bin/env python3

import argparse
import os
import socket
import sys
import time


MASK_ENV_VAR = "BOREDEALIS_SMOOTHER_DEVICES_MASKED"


def _get_cli_arg_value(flag_name):
    for index, arg in enumerate(sys.argv[1:], start=1):
        if arg == flag_name:
            return sys.argv[index + 1] if index + 1 < len(sys.argv) else None
        if arg.startswith(f"{flag_name}="):
            return arg.split("=", 1)[1]
    return None


def _set_cli_arg_value(flag_name, value):
    value = str(value)
    for index, arg in enumerate(sys.argv[1:], start=1):
        if arg == flag_name and index + 1 < len(sys.argv):
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
    vgg_device_arg = _get_cli_arg_value("--vgg-device")
    vgg_device_id = _parse_global_cuda_id(vgg_device_arg)
    visible_device_ids = list(train_device_ids)
    if vgg_device_id is not None and vgg_device_id not in visible_device_ids:
        visible_device_ids.append(vgg_device_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(device_id) for device_id in visible_device_ids)
    remapped_train_ids = [visible_device_ids.index(device_id) for device_id in train_device_ids]
    _set_cli_arg_value("--device", ",".join(str(device_id) for device_id in remapped_train_ids))
    if vgg_device_id is not None:
        _set_cli_arg_value("--vgg-device", str(visible_device_ids.index(vgg_device_id)))

    os.environ[MASK_ENV_VAR] = "1"


_configure_visible_devices()

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from raft import RAFT
from smoother_datasets import VideoDataset, split_video_files
from smoother_losses import DeflickerLoss, LAMBDAS, generate_circle_mask, resolve_vgg_device
from smoother_models import build_smoother_model, load_smoother_checkpoint, peek_checkpoint_architecture

torch.multiprocessing.set_sharing_strategy("file_system")

if not os.path.exists("local_settings.py"):
    print("Warning: local settings not found. Using default settings.")
    import settings
else:
    import local_settings as settings


def rank_print(rank, message):
    if rank == 0:
        print(message, flush=True)


def parse_device_arg(device_arg):
    if device_arg is None:
        return None
    requested_device = device_arg.strip().lower()
    if requested_device in {"cpu", "cuda"}:
        return requested_device
    return [int(device_id.strip()) for device_id in requested_device.split(",") if device_id.strip()]


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
        return [0]
    return parsed_device


def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def make_dataloaders(train_files, valid_files, input_frames, batch_size, num_workers, rank, world_size):
    train_dataset = VideoDataset(train_files, window=input_frames)
    valid_dataset = VideoDataset(valid_files, window=input_frames)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        sampler=valid_sampler,
        num_workers=num_workers,
    )
    return train_dataset, valid_dataset, train_loader, valid_loader, train_sampler


def reduce_sum(value, device, world_size):
    tensor = torch.tensor([value], device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.item()


def train_worker(rank, world_size, device_ids, args, master_port):
    using_cuda = device_ids != ["cpu"]
    device = torch.device(f"cuda:{device_ids[rank]}") if using_cuda else torch.device("cpu")
    if using_cuda:
        torch.cuda.set_device(device)

    if world_size > 1:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(master_port)
        init_kwargs = {"backend": args.backend, "rank": rank, "world_size": world_size}
        if using_cuda and args.backend == "nccl":
            init_kwargs["device_id"] = device.index
        dist.init_process_group(**init_kwargs)

    vgg_device = resolve_vgg_device(args.vgg_device, device)
    rank_print(rank, f"Using device: {device}")
    rank_print(rank, f"Using VGG device: {vgg_device}")

    input_frames = args.input_frames
    num_res_blocks = args.num_res_blocks
    hidden_channels = args.hidden_channels
    if args.previous_model_path and os.path.exists(args.previous_model_path):
        input_frames, num_res_blocks, hidden_channels = peek_checkpoint_architecture(
            args.previous_model_path,
            input_frames,
            num_res_blocks,
            hidden_channels,
        )

    model = build_smoother_model(
        input_frames=input_frames,
        num_res_blocks=num_res_blocks,
        hidden_channels=hidden_channels,
        save_memory=True,
    ).to(device)
    model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None, output_device=device.index if device.type == "cuda" else None)
    raft_model = RAFT(device).to(device)
    criterion = DeflickerLoss(lambda_values=args.lambdas, device=vgg_device).to(vgg_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.LEARNING_RATE)
    start_epoch = 0

    if args.previous_model_path and os.path.exists(args.previous_model_path):
        start_epoch = load_smoother_checkpoint(model.module, optimizer, args.previous_model_path, device)

    train_files, valid_files = split_video_files(args.data_dir)
    train_dataset, valid_dataset, train_loader, valid_loader, train_sampler = make_dataloaders(
        train_files,
        valid_files,
        input_frames,
        args.batch_size,
        args.workers,
        rank,
        world_size,
    )
    roi_mask = generate_circle_mask(height=train_dataset.height, width=train_dataset.width, device=vgg_device)
    mid_idx = input_frames // 2
    curr_start = mid_idx * 3
    curr_end = curr_start + 3
    prev_start = (mid_idx - 1) * 3
    prev_end = prev_start + 3

    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs_curr, inputs_prev) in enumerate(train_loader):
            inputs_curr = inputs_curr.to(device)
            inputs_prev = inputs_prev.to(device)
            optimizer.zero_grad()
            input_frame_t = inputs_curr[:, curr_start:curr_end, :, :]
            input_frame_prev = inputs_curr[:, prev_start:prev_end, :, :]
            input_frame_curr = inputs_curr[:, curr_start:curr_end, :, :]
            flow = raft_model(input_frame_curr, input_frame_prev)
            with torch.autocast(device_type="cuda") if device.type == "cuda" else torch.autocast(device_type="cpu"):
                output_t = model(inputs_curr)
                output_prev = model(inputs_prev)
                losses = criterion(output_t, input_frame_t, output_prev, flow, roi_mask)
            losses.total_loss.backward()
            optimizer.step()
            running_loss += losses.total_loss.item()
            if rank == 0 and batch_idx % 20 == 0:
                print(f"    Batch {batch_idx}/{len(train_loader)} | Total Loss: {losses.total_loss.item():.4f}")

        validation_loss = 0.0
        model.eval()
        with torch.no_grad():
            for inputs_curr, inputs_prev in valid_loader:
                inputs_curr = inputs_curr.to(device)
                inputs_prev = inputs_prev.to(device)
                input_frame_t = inputs_curr[:, curr_start:curr_end, :, :]
                input_frame_prev = inputs_curr[:, prev_start:prev_end, :, :]
                input_frame_curr = inputs_curr[:, curr_start:curr_end, :, :]
                flow = raft_model(input_frame_curr, input_frame_prev)
                with torch.autocast(device_type="cuda") if device.type == "cuda" else torch.autocast(device_type="cpu"):
                    output_t = model(inputs_curr)
                    output_prev = model(inputs_prev)
                losses = criterion(output_t, input_frame_t, output_prev, flow, roi_mask)
                validation_loss += losses.total_loss.item()

        avg_loss = reduce_sum(running_loss, device, world_size) / max(reduce_sum(len(train_loader), device, world_size), 1.0)
        avg_val = reduce_sum(validation_loss, device, world_size) / max(reduce_sum(len(valid_loader), device, world_size), 1.0)
        rank_print(rank, f"Epoch {epoch+1}/{args.epochs} | Train: {avg_loss:.4f} | Val: {avg_val:.4f} | Duration: {time.time() - epoch_start:.2f}s")

        if rank == 0 and epoch % settings.EPOCH_SAVE_INTERVAL == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "input_frames": input_frames,
                    "num_res_blocks": num_res_blocks,
                    "hidden_channels": hidden_channels,
                },
                os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pth"),
            )

    if world_size > 1:
        dist.barrier(device_ids=[device.index] if device.type == "cuda" and args.backend == "nccl" else None)
        dist.destroy_process_group()


def train_smoother_ddp(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device_ids = resolve_training_devices(args.device)
    if device_ids == ["cpu"] or len(device_ids) == 1:
        train_worker(0, 1, device_ids, args, None)
        return
    master_port = get_free_port()
    mp.spawn(train_worker, args=(len(device_ids), device_ids, args, master_port), nprocs=len(device_ids), join=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.path.join("media", "filtered_training_videos"))
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--device")
    parser.add_argument("--vgg-device")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--input-frames", type=int, default=3)
    parser.add_argument("--num-res-blocks", type=int, default=12)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--previous-model-path")
    parser.add_argument("--backend", choices=["nccl", "gloo"], default="nccl")
    parser.add_argument("--lambda-l1", type=float, default=5.0)
    parser.add_argument("--lambda-rec", type=float, default=1.0)
    parser.add_argument("--lambda-l1-perc", type=float, default=0.1)
    parser.add_argument("--lambda-rec-perc", type=float, default=0.5)
    args = parser.parse_args()
    args.lambdas = LAMBDAS(
        l1=args.lambda_l1,
        rec=args.lambda_rec,
        l1_perc=args.lambda_l1_perc,
        rec_perc=args.lambda_rec_perc,
    )
    return args


if __name__ == "__main__":
    train_smoother_ddp(parse_args())
