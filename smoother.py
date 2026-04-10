#!/usr/bin/env python3

import argparse
import os
import time

import torch
import torch.nn as nn
from raft import RAFT

from smoother_datasets import make_smoother_dataloaders, split_video_files
from smoother_losses import DeflickerLoss, LAMBDAS, generate_circle_mask, resolve_vgg_device
from smoother_models import build_smoother_model, load_smoother_checkpoint, peek_checkpoint_architecture

torch.multiprocessing.set_sharing_strategy("file_system")

if not os.path.exists("local_settings.py"):
    print("Warning: local settings not found. Using default settings.")
    import settings
else:
    import local_settings as settings


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
        return torch.device("cuda"), None
    return torch.device(f"cuda:{parsed_device[0]}"), parsed_device


def maybe_wrap_parallel(model, device, device_ids, name):
    if device_ids and len(device_ids) > 1:
        print(f"Wrapping {name} with nn.DataParallel for devices {device_ids}")
        model = model.to(device)
        model = nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[0])
        return model
    return model.to(device)


def train_smoother(
    data_dir=os.path.join("media", "filtered_training_videos"),
    output_dir=settings.MODEL_SAVE_PATH,
    input_frames=3,
    num_res_blocks=12,
    hidden_channels=128,
    device_arg=None,
    vgg_device_arg=None,
    batch_size=settings.BATCH_SIZE,
    num_workers=settings.NUM_WORKERS,
    num_epochs=settings.NUM_EPOCHS,
    previous_model_path=None,
    lambdas=None,
    debug=False,
):
    device, device_ids = resolve_training_devices(device_arg)
    vgg_device = resolve_vgg_device(vgg_device_arg, device)
    os.makedirs(output_dir, exist_ok=True)
    start_epoch = 0
    save_memory = True

    print(f"Using device: {device}")
    print(f"Using VGG device: {vgg_device}")

    if previous_model_path and os.path.exists(previous_model_path):
        print(f"Peeking at {previous_model_path} for architecture...")
        input_frames, num_res_blocks, hidden_channels = peek_checkpoint_architecture(
            previous_model_path,
            input_frames,
            num_res_blocks,
            hidden_channels,
        )
        print(f"Resuming with: Depth={num_res_blocks}, Width={hidden_channels}")

    print("Initializing model")
    model = build_smoother_model(
        input_frames=input_frames,
        num_res_blocks=num_res_blocks,
        hidden_channels=hidden_channels,
        save_memory=save_memory,
    )
    print("Model initialized with:")
    print(f"  Input frames = {input_frames}")
    print(f"  Num res blocks = {num_res_blocks}")
    print(f"  Hidden channels = {hidden_channels}")
    print(f"  Save memory = {save_memory}")

    print("Initializing loss and optimizer")
    criterion = DeflickerLoss(lambda_values=lambdas, device=vgg_device).to(vgg_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.LEARNING_RATE)

    if previous_model_path and os.path.exists(previous_model_path):
        print(f"Loading previous model from {previous_model_path}")
        start_epoch = load_smoother_checkpoint(model, optimizer, previous_model_path, device)

    print(f"Initializing RAFT on {device}")
    raft_model = RAFT(device).to(device)
    model = maybe_wrap_parallel(model, device, device_ids, "model")
    raft_model = maybe_wrap_parallel(raft_model, device, device_ids, "RAFT")

    print("Initializing datasets")
    train_files, valid_files = split_video_files(data_dir)
    train_dataset, valid_dataset, train_loader, valid_loader = make_smoother_dataloaders(
        train_files,
        valid_files,
        input_frames=input_frames,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    if not len(train_loader) > 0 or not len(valid_loader) > 0:
        raise ValueError("error: no training and/or validation samples found. womp womp.")

    print("Generating mask")
    roi_mask = generate_circle_mask(
        height=train_dataset.height,
        width=train_dataset.width,
        device=vgg_device,
    )

    mid_idx = input_frames // 2
    curr_start = mid_idx * 3
    curr_end = curr_start + 3
    prev_start = (mid_idx - 1) * 3
    prev_end = prev_start + 3

    print("Let's go!")
    for epoch in range(start_epoch, num_epochs):
        start_time = time.time()
        print(f"Epoch {epoch+1} / {num_epochs}")
        running_loss = 0.0

        if debug:
            print("Debug ending. Exiting . . .")
            return

        print("  Beginning training")
        model.train()
        for batch_idx, (inputs_curr, inputs_prev) in enumerate(train_loader):
            inputs_curr = inputs_curr.to(device)
            inputs_prev = inputs_prev.to(device)
            optimizer.zero_grad()

            input_frame_t = inputs_curr[:, curr_start:curr_end, :, :]
            input_frame_prev = inputs_curr[:, prev_start:prev_end, :, :]
            input_frame_curr = inputs_curr[:, curr_start:curr_end, :, :]
            flow = raft_model(input_frame_curr, input_frame_prev)

            autocast_context = torch.autocast(device_type="cuda") if device.type == "cuda" else torch.autocast(device_type="cpu")
            with autocast_context:
                output_t = model(inputs_curr)
                output_prev = model(inputs_prev)
                losses = criterion(
                    output_t=output_t,
                    input_t=input_frame_t,
                    output_prev=output_prev,
                    flow=flow,
                    occlusion_mask=roi_mask,
                )

            losses.total_loss.backward()
            optimizer.step()
            running_loss += losses.total_loss.item()

            if batch_idx % 20 == 0:
                print(
                    f"    Batch {batch_idx}/{len(train_loader)} | Total Loss: {losses.total_loss.item():.4f} | Temp: {losses.temp_loss.item():.4f} | Rec: {losses.rec_loss.item():.4f} | Temp Perc: {losses.temp_perc_loss.item():.4f} | Rec Perc: {losses.rec_perc_loss.item():.4f} | Time: {time.time() - start_time:.2f}s"
                )

        print(f"  Training finished in {(time.time() - start_time):.4f}s | Total Loss: {running_loss/len(train_loader):.4f}")

        print("  Beginning validation")
        validation_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch_idx, (inputs_curr, inputs_prev) in enumerate(valid_loader):
                inputs_curr = inputs_curr.to(device)
                inputs_prev = inputs_prev.to(device)
                input_frame_t = inputs_curr[:, curr_start:curr_end, :, :]
                input_frame_prev = inputs_curr[:, prev_start:prev_end, :, :]
                input_frame_curr = inputs_curr[:, curr_start:curr_end, :, :]
                flow = raft_model(input_frame_curr, input_frame_prev)

                autocast_context = torch.autocast(device_type="cuda") if device.type == "cuda" else torch.autocast(device_type="cpu")
                with autocast_context:
                    output_t = model(inputs_curr)
                    output_prev = model(inputs_prev)
                losses = criterion(
                    output_t=output_t,
                    input_t=input_frame_t,
                    output_prev=output_prev,
                    flow=flow,
                    occlusion_mask=roi_mask,
                )
                validation_loss += losses.total_loss.item()

                if batch_idx % 20 == 0:
                    print(
                        f"    Batch {batch_idx}/{len(valid_loader)} | Total Loss: {losses.total_loss.item():.4f} | Temp: {losses.temp_loss.item():.4f} | Rec: {losses.rec_loss.item():.4f} | Temp Perc: {losses.temp_perc_loss.item():.4f} | Rec Perc: {losses.rec_perc_loss.item():.4f} | Time: {time.time() - start_time:.2f}s"
                    )

        print(f"  Validation finished in {(time.time() - start_time):.4f}s | Total Loss: {validation_loss/len(valid_loader):.4f}")
        avg_loss = running_loss / len(train_loader)
        avg_val = validation_loss / len(valid_loader)
        print(f"Epoch {epoch+1} Complete.\nAverage Loss: {avg_loss:.4f} | Average Validation Loss {avg_val:.4f}")
        print(f"Epoch duration: {time.time() - start_time:.2f}s")

        if epoch % settings.EPOCH_SAVE_INTERVAL == 0:
            save_dict = {
                "epoch": epoch,
                "model_state_dict": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "input_frames": input_frames,
                "num_res_blocks": num_res_blocks,
                "hidden_channels": hidden_channels,
            }
            checkpoint_name = f"checkpoint_epoch_{epoch}.pth"
            torch.save(save_dict, os.path.join(output_dir, checkpoint_name))
            print(f"Model saved to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.path.join("media", "filtered_training_videos"))
    parser.add_argument("--output-dir", default=settings.MODEL_SAVE_PATH)
    parser.add_argument("--device")
    parser.add_argument("--vgg-device")
    parser.add_argument("--batch-size", type=int, default=settings.BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=settings.NUM_WORKERS)
    parser.add_argument("--epochs", type=int, default=settings.NUM_EPOCHS)
    parser.add_argument("--input-frames", type=int, default=3)
    parser.add_argument("--num-res-blocks", type=int, default=12)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--previous-model-path")
    parser.add_argument("--debug", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--lambda-l1", type=float, default=5.0)
    parser.add_argument("--lambda-rec", type=float, default=1.0)
    parser.add_argument("--lambda-l1-perc", type=float, default=0.1)
    parser.add_argument("--lambda-rec-perc", type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_smoother(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        input_frames=args.input_frames,
        num_res_blocks=args.num_res_blocks,
        hidden_channels=args.hidden_channels,
        device_arg=args.device,
        vgg_device_arg=args.vgg_device,
        batch_size=args.batch_size,
        num_workers=args.workers,
        num_epochs=args.epochs,
        previous_model_path=args.previous_model_path,
        lambdas=LAMBDAS(
            l1=args.lambda_l1,
            rec=args.lambda_rec,
            l1_perc=args.lambda_l1_perc,
            rec_perc=args.lambda_rec_perc,
        ),
        debug=args.debug,
    )
