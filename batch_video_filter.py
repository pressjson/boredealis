#!/usr/bin/env python3

"""
Batch filter an entire directory of videos using all of the GPUs specified.

I could do this in bash, but multithreading . . .
"""

import os
if not os.path.exists("local_settings.py"):
    print("Warning: local settings not found. Using default settings.")
    import settings
else:
    import local_settings as settings

import argparse
from model_utils import get_model_names

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="path to model", default="")
parser.add_argument("--model-name", help="override checkpoint model architecture", choices=get_model_names())
parser.add_argument("--input", help="input path", default="")
parser.add_argument("--output", help="output path", default="")
parser.add_argument("--vids-per-device", "-V", default=1)
parser.add_argument("--ext", default=".mp4")
parser.add_argument("--filter", default=True, action=argparse.BooleanOptionalAction,
                    help="--filter for cloud removal, --no-filter for smoothing; default is filter")
parser.add_argument("--debug", action='store_true', default=False)
parser.add_argument("--device", nargs="*", default=settings.DEVICE_IDS if settings.USE_DEVICE_IDS else "cuda",
                    help="the numbers for device ids to use, e.g. <0 1 2 3>")
[args, OTHERS] = parser.parse_known_args()

INPUT_DIR = args.input
OUTPUT_DIR = args.output
MODEL_PATH = args.model
MODEL_NAME = args.model_name
VIDS_PER_DEVICE = int(args.vids_per_device)
FILTER = args.filter
EXT = args.ext
print(f"args: input = {INPUT_DIR} | output = {OUTPUT_DIR} | model = {MODEL_PATH} | addtl args: {max(0, len(OTHERS))}")
if OTHERS:
    print(f"    others: {OTHERS}")

AVAILABLE_DEVICES = [f"cuda:{int(x)}" for x in args.device] * VIDS_PER_DEVICE
print(f"running on {AVAILABLE_DEVICES}")
MAX_WORKERS = len(AVAILABLE_DEVICES)

import sys
if args.debug:
    print("Debug complete!")
    sys.exit(0)

if not os.path.exists(INPUT_DIR) or INPUT_DIR == "":
    print(f"error: input dir {INPUT_DIR} does not exist")
    sys.exit(-1)

os.makedirs(OUTPUT_DIR, exist_ok=True)

class ARGS:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.MODEL_PATH = MODEL_PATH
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
        if MODEL_NAME:
            cmd += ["--model-name", MODEL_NAME]
        if len(self.OTHERS) > 0:
            cmd = cmd + [arg for arg in self.OTHERS]
        # if self.device:
        #     cmd = cmd + [self.device]
        return cmd


import time
import asyncio
from asyncio import Queue

total_items = 0
completed_count = 0

def init_queue(folder) -> asyncio.Queue:
    q = Queue()
    for root, _, files in os.walk(folder):
        for f in files:
            path = os.path.relpath(os.path.join(root, f), folder)
            q.put_nowait(path)
    return q

async def worker(device, queue):
    while True:
        global completed_count
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
            completed_count += 1
        finally:
            queue.task_done()

async def filter_video(args: ARGS, device):
    cmd = args.return_command() + ["--device", device]
    print(f"running {cmd} on device {device}")
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    stdout, stderr = await proc.communicate()

async def smooth_video(these_args: ARGS, device):
    t = time.time()
    cmd = [
        sys.executable,
        "smoother_test.py",
        "--model", these_args.MODEL_PATH,
        "--input", these_args.input_path,
        "--output", these_args.output_path,
        "--device", device,
    ]
    item = completed_count
    print(f"running {cmd} | item {item} / {total_items} ")
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    print(f"finished item {item} in {time.time() - t:.2f}s")

async def main():
    global total_items
    print("initializing . . .")
    q = init_queue(INPUT_DIR)
    print(f"initialized with {q.qsize()} items.")
    total_items = q.qsize()
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
