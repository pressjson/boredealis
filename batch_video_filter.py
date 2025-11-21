#!/usr/bin/env python3

"""Batch filter an entire directory of videos using all of the GPUs specified in SETTINGS.

I could do this in bash, but multithreading . . ."""

import sys

def print_help():
    print("usage: python3 batch_video_filter.py <input_dir> <output_dir> <model_path> <other_flags>")
    print("<other_flags> should be compatible with main.py")
    print("<input_dir> and <output_dir> should just be the directories")

if len(sys.argv) < 4 or "-h" in sys.argv or "--help" in sys.argv:
    print_help()
    sys.exit(-1)

INPUT_DIR = sys.argv[1]
OUTPUT_DIR = sys.argv[2]
MODEL_PATH = sys.argv[3]
VIDS_PER_DEVICE = 1
EXT = ".mp4"

print(f"args: input = {INPUT_DIR} | output = {OUTPUT_DIR} | model = {MODEL_PATH} | addtl args: {max(0, len(sys.argv) - 4)}")

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
        self.MODEL_PATH = sys.argv[3]
        self.OTHERS = sys.argv[4:] if len(sys.argv) >= 5 else []

    def return_command(self):
        cmd = ["python3", "main.py", f"-i='{self.input_path}'", f"-o='{self.output_path}'", f"-c='{self.MODEL_PATH}'"] 
        if len(self.OTHERS) > 0:
            cmd = cmd + [arg for arg in self.OTHERS]
        return cmd

AVAILABLE_DEVICES = [f"cuda:{x}" for x in settings.DEVICE_IDS] * VIDS_PER_DEVICE
print(f"running on {AVAILABLE_DEVICES}")
MAX_WORKERS = len(AVAILABLE_DEVICES)

import time
import asyncio
from asyncio import Queue

def init_queue(folder):
    q = Queue()
    for video in os.listdir(folder):
        q.put_nowait(video)
    return q

async def worker(device, queue):
    while True:
        try:
            item = await queue.get()
            input_path = os.path.join(INPUT_DIR, item)
            output_path = os.path.splitext(os.path.join(OUTPUT_DIR, item))[0] + EXT
            args = ARGS(input_path, output_path)
            await filter_video(args, device)
        finally:
            queue.task_done()

async def filter_video(args: ARGS, device):
    cmd = args.return_command() + [f"--device={device}"]
    print(f"running {cmd}")
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    stdout, stderr = await proc.communicate()
    # await asyncio.sleep(1)
    print(f"device {device} finished {cmd}")


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
    print(f"it only took {time.time() - t} seconds")
    
