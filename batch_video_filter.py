#!/usr/bin/env python3

"""Batch filter an entire directory of videos using all of the GPUs specified in SETTINGS.

I could do this in bash, but multithreading . . ."""

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model", "-m", help="path to model", default="")
parser.add_argument("--input", "-i", help="input path (include extension)", default="")
parser.add_argument("--output", "-o", help="output path (include extension)", default="")
parser.add_argument("--vids-per-device", "-V", default=1)
parser.add_argument("--ext", default=".mp4")
parser.add_argument("--filter", default=True, action=argparse.BooleanOptionalAction,
                    help="--filter for cloud removal, --no-filter for smoothing; default is filter")
parser.add_argument("--debug", action='store_true', default=False)
[args, OTHERS] = parser.parse_known_args()

INPUT_DIR = args.input
OUTPUT_DIR = args.output
MODEL_PATH = args.model
VIDS_PER_DEVICE = 1
FILTER = args.filter
EXT = args.ext
print(f"args: input = {INPUT_DIR} | output = {OUTPUT_DIR} | model = {MODEL_PATH} | addtl args: {max(0, len(OTHERS))}")

if OTHERS:
    print(f"    others: {OTHERS}")

import sys
if args.debug:
    print("Debug complete!")
    sys.exit(0)

import os
if not os.path.exists(INPUT_DIR) or INPUT_DIR == "":
    print(f"error: input dir {INPUT_DIR} does not exist")
    sys.exit(-1)

os.makedirs(OUTPUT_DIR, exist_ok=True)


if not os.path.exists("local_settings.py"):
    print("Warning: local settings not found. Using default settings.")
    import settings
else:
    import local_settings as settings

class ARGS:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.MODEL_PATH = MODEL_PATH,
        self.OTHERS = OTHERS
        # self.device = ""
        # if self.OTHERS:
        #     device_flag = [x for x in self.OTHERS if "--device" in x]
        #     if device_flag:
        #         self.device=device_flag[0][9:]

    def return_command(self):
        cmd = [
            sys.executable, "main.py",
            "--input", self.input_path,
            "--output", self.output_path,
            "--model", self.MODEL_PATH,
        ]
        if len(self.OTHERS) > 0:
            cmd = cmd + [arg for arg in self.OTHERS]
        # if self.device:
        #     cmd = cmd + [self.device]
        return cmd

AVAILABLE_DEVICES = [f"cuda:{x}" for x in settings.DEVICE_IDS] * VIDS_PER_DEVICE if settings.DEVICE_IDS else ["cpu"]
print(f"running on {AVAILABLE_DEVICES}")
MAX_WORKERS = len(AVAILABLE_DEVICES)

import time
import asyncio
from asyncio import Queue


def init_queue(folder):
    q = Queue()
    for root, _, files in os.walk(folder):
        for f in files:
            path = os.path.relpath(os.path.join(root, f), folder)
            q.put_nowait(path)
    return q

async def worker(device, queue):
    while True:
        item = await queue.get()
        try:
            input_path = os.path.join(INPUT_DIR, item)
            output_path = os.path.splitext(os.path.join(OUTPUT_DIR, item))[0] + EXT
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            args = ARGS(input_path, output_path)
            if FILTER:
                await filter_video(args, device)
            else:
                await smooth_video(args, device)
        finally:
            queue.task_done()

async def filter_video(args: ARGS, device):
    cmd = args.return_command() + [f"--device={device}"]
    print(f"running {cmd} on device {device}")
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    stdout, stderr = await proc.communicate()
    # await asyncio.sleep(1)

    # cpu-bound, no gpu parallelism
    # print(f"running filter on device {device}")
    # return await asyncio.to_thread(
    #     filter_video_in_a_pipeline,
    #     args.MODEL_PATH,
    #     args.input_path,
    #     args.output_path,
    #     device
    # )


async def smooth_video(args: ARGS, device):
    cmd = [
        sys.executable,
        "run_smoother.py",
        "--model", args.MODEL_PATH,
        "--input", args.input_path,
        "--output", args.output_path,
        "--device", device,
    ]
    print(f"running {cmd} on device {device}")
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()

async def main():
    print("initializing . . .")
    q = init_queue(INPUT_DIR)
    tasks = []
    print("filtering . . .")
    for device in AVAILABLE_DEVICES:
        t = asyncio.create_task(worker(device, q))
        tasks.append(t)

    await q.join()
    for t in tasks:
        t.cancel()

    await asyncio.gather(*tasks, return_exceptions=True)

if __name__ == "__main__":
    t = time.time()
    asyncio.run(main())
    print("done (:")
    print(f"it only took {time.time() - t:.2f} seconds")
