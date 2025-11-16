#!/usr/bin/env python3

"""Batch filter an entire directory of videos using all of the GPUs specified in SETTINGS.

I could do this in bash, but multithreading . . ."""

import sys

INPUT_DIR = None
OUTPUT_DIR = None

for arg in sys.argv[1:]:
    if arg.startswith("--help") or arg == "-h" or len(argv) != 4:
        print("usage: python3 batch_video_filter.py <input_dir> <output_dir> <model_path>")
    else:
        INPUT_DIR = sys.argv[1]
        OUTPUT_DIR = sys.argv[2]
        MODEL_PATH = sys.argv[3]

import os
if not os.path.exists(INPUT_DIR):
    print(f"warning: input dir {INPUT_DIR} does not exist")
    return -1
os.makedirs(OUTPUT_DIR)
        

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
import main
import subprocess

def init_queue(folder):
    q = Queue()
    for video in os.listdir(folder):
        q.put_nowait(video)
    return q

async def worker(device, queue):
    while True:
        item = await queue.get()
        input_path = os.path.join(INPUT_DIR, item)
        output_path = os.path.join(OUTPUT_DIR, item)
        try:
            await filter(input_path, output_path, device)
        finally:
            queue.task_done()

async def filter(input_path, output_path, device):
    cmd = ["python3", "main.py", f"-i={input_path}", f"-o={output_path}", f"--device={device}"]
    print(cmd)
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout
    

def main():
    
    tasks = []
    for device in AVAILABLE_DEVICES:
        t = asyncio.create_task(worker(device, q))
        tasks.append(t)

    await q.join()
    for t in tasks():
        t.cancel()

    await asyncio.gather(*tasks, return_exceptions=True)

asyncio.run(main)
    
