#!/usr/bin/env python3

"""Batch-train model architectures across one or more devices."""

import argparse
import asyncio
import os
import re
import sys
from dataclasses import dataclass

from model_utils import get_model_names
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn


MODEL_NAMES = get_model_names()

RESERVED_FORWARD_FLAGS = {
    "--data-dir",
    "--output-dir",
    "--model",
    "--device",
}

EPOCH_PATTERN = re.compile(r"--- Epoch (\d+)/(\d+) \[Train\] ---")


@dataclass
class TrainJob:
    model_name: str
    input_dir: str
    output_dir: str


def parse_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", required=True, help="Comma-separated devices, e.g. 0,1,2,3 or cpu")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--models", default=",".join(MODEL_NAMES), help="Comma-separated model names")
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Per-rank batch size. Use -1 to auto-probe the largest safe size with a 15% safety margin.",
    )
    parser.add_argument("--workers", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--start-filters", type=int)
    parser.add_argument("--n-channels-in", type=int)
    parser.add_argument("--n-classes-out", type=int)
    parser.add_argument("--previous-model-path")
    parser.add_argument("--debug", default=False, action=argparse.BooleanOptionalAction)
    return parser.parse_known_args()


def validate_forward_args(extra_args):
    for arg in extra_args:
        if arg in RESERVED_FORWARD_FLAGS or any(arg.startswith(f"{flag}=") for flag in RESERVED_FORWARD_FLAGS):
            raise ValueError(f"{arg} is managed by batch_train_models.py and cannot be forwarded.")


def build_forward_args(args, extra_args):
    forward_args = []

    def append_flag(flag, value):
        if value is not None:
            forward_args.extend([flag, str(value)])

    append_flag("--batch-size", args.batch_size)
    append_flag("--workers", args.workers)
    append_flag("--epochs", args.epochs)
    append_flag("--start-filters", args.start_filters)
    append_flag("--n-channels-in", args.n_channels_in)
    append_flag("--n-classes-out", args.n_classes_out)
    append_flag("--previous-model-path", args.previous_model_path)
    forward_args.append("--debug" if args.debug else "--no-debug")
    forward_args.extend(extra_args)
    return forward_args


def create_jobs(models, input_dir, output_dir):
    return [
        TrainJob(
            model_name=model_name,
            input_dir=input_dir,
            output_dir=os.path.join(output_dir, model_name),
        )
        for model_name in models
    ]


async def stream_training_output(process, log_path, progress, task_id):
    total_epochs = None
    with open(log_path, "w", encoding="utf-8") as log_file:
        assert process.stdout is not None
        async for raw_line in process.stdout:
            line = raw_line.decode("utf-8", errors="replace").rstrip()
            log_file.write(line + "\n")
            log_file.flush()

            match = EPOCH_PATTERN.search(line)
            if match:
                current_epoch = int(match.group(1)) + 1
                total_epochs = int(match.group(2))
                progress.update(
                    task_id,
                    total=total_epochs,
                    completed=current_epoch,
                    status=f"epoch {current_epoch}/{total_epochs}",
                )
                continue

            if "Training Finished" in line and total_epochs is not None:
                progress.update(task_id, completed=total_epochs, status="finishing")

    return total_epochs


async def run_job(job, device, forward_args, progress, task_id):
    os.makedirs(job.output_dir, exist_ok=True)
    log_path = os.path.join(job.output_dir, "train.log")
    command = [
        sys.executable,
        "network.py",
        "--model",
        job.model_name,
        "--device",
        device,
        "--data-dir",
        job.input_dir,
        "--output-dir",
        job.output_dir,
        *forward_args,
    ]

    progress.update(task_id, status=f"starting on {device}")
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    total_epochs = await stream_training_output(process, log_path, progress, task_id)
    return_code = await process.wait()

    if return_code == 0:
        if total_epochs is not None:
            progress.update(task_id, total=total_epochs, completed=total_epochs)
        progress.update(task_id, status=f"done on {device}")
        return True, log_path

    progress.update(task_id, status=f"failed on {device}")
    return False, log_path


async def worker(device, queue, forward_args, progress, task_ids, results):
    while True:
        job = await queue.get()
        task_id = task_ids[job.model_name]
        progress.update(task_id, status=f"running on {device}")
        try:
            ok, log_path = await run_job(job, device, forward_args, progress, task_id)
            results.append((job.model_name, ok, log_path))
        finally:
            queue.task_done()


async def main_async(args, extra_args):
    console = Console()
    validate_forward_args(extra_args)

    devices = parse_csv(args.devices)
    models = parse_csv(args.models)
    invalid_models = [model for model in models if model not in MODEL_NAMES]
    if invalid_models:
        raise ValueError(f"Unsupported models: {', '.join(invalid_models)}")

    if not devices:
        raise ValueError("At least one device must be provided.")

    os.makedirs(args.output_dir, exist_ok=True)
    forward_args = build_forward_args(args, extra_args)
    jobs = create_jobs(models, args.input_dir, args.output_dir)

    queue = asyncio.Queue()
    for job in jobs:
        queue.put_nowait(job)

    results = []
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}[/bold]"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("{task.fields[status]}"),
        TimeElapsedColumn(),
        console=console,
    )

    with progress:
        task_ids = {
            job.model_name: progress.add_task(job.model_name, total=None, status="queued")
            for job in jobs
        }
        workers = [
            asyncio.create_task(worker(device, queue, forward_args, progress, task_ids, results))
            for device in devices
        ]
        await queue.join()
        for current_worker in workers:
            current_worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

    failed = [result for result in results if not result[1]]
    if failed:
        for model_name, _, log_path in failed:
            console.print(f"[red]failed[/red] {model_name} - see {log_path}")
        return 1

    console.print("[green]all models finished successfully[/green]")
    return 0


def main():
    args, extra_args = parse_args()
    try:
        raise SystemExit(asyncio.run(main_async(args, extra_args)))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
