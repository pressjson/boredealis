#!/usr/bin/env python3

"""Batch filter an entire directory of videos using all of the GPUs specified in SETTINGS.

I could do this in bash, but multithreading . . ."""

import sys

INPUT_DIR = ""
OUTPUT_DIR = ""

for arg in sys.argv[1:]:
    if arg.startswith("--help") or arg == "-h" or len(sys.argv) != 4:
        print("usage: python3 batch_video_filter.py <input_dir> <output_dir> <model_path>")
        sys.exit(-1)
    else:
        INPUT_DIR = sys.argv[1]
        OUTPUT_DIR = sys.argv[2]
        MODEL_PATH = sys.argv[3]

import os
if not os.path.exists(INPUT_DIR) or INPUT_DIR == "":
    print(f"warning: input dir {INPUT_DIR} does not exist")
    sys.exit(-1)

print(f"args: input = {INPUT_DIR} | output = {OUTPUT_DIR} | model = {MODEL_PATH}")
os.makedirs(OUTPUT_DIR, exist_ok=True)
        

if not os.path.exists("local_settings.py"):
    print("Warning: local settings not found. Using default settings.")
    import settings
else:
    import local_settings as settings


AVAILABLE_DEVICES = [f"cuda:{x}" for x in settings.DEVICE_IDS]
print(f"running on {AVAILABLE_DEVICES}")
MAX_WORKERS = len(AVAILABLE_DEVICES)
import time
import asyncio
from asyncio import Queue
import random
from ffmpeg_wrapper import split_filename

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
            base, ext = split_filename(input_path)
            output_path = f"{base}.mp4"
            await filter_video(input_path, output_path, device)
        finally:
            queue.task_done()

async def filter_video(input_path, output_path, device):
    cmd = ["python3", "main.py", f"-i={input_path}", f"-o={output_path}", f"-c={MODEL_PATH}", f"--device={device}"]
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
    print("done (:")

if __name__ == "__main__":
    asyncio.run(main())
    
