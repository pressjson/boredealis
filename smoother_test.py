#!/usr/bin/env python3

from smoother import DeflickerCNN, load_checkpoint

import os
if not os.path.exists("local_settings.py"):
    print("Warning: local settings not found. Using default settings.")
    import settings
else:
    import local_settings as settings
    
import cv2
import sys
import numpy
import torch
from rich.progress import track
from collections import deque

def main(
    model_path="",
    input_video_path="",
    output_path="",
    device = "cuda" if torch.cuda.is_available() else "cpu",
    verbose=True,
    debug=False, # just in case i need it in the future
):

    if not os.path.exists(model_path):
        print(f"Error: model path does not exist: {model_path}")
        exit(-1)
    if not os.path.exists(input_video_path):
        print(f"Error: input video path does not exist: {input_video_path}")
        exit(-1)
    
    output_base_path = os.path.split(output_path)[0]
    if verbose:
        print(f"Making output dir {output_base_path}")
    os.makedirs(output_base_path, exist_ok=True)

    # load checkpoint's previous metadata
    if verbose:
        print(f"Peeking at {model_path} for architecture...")
    temp_ckpt = torch.load(model_path, map_location='cpu')
    num_res_blocks = temp_ckpt['num_res_blocks']
    hidden_channels = temp_ckpt['hidden_channels']
    input_frames = temp_ckpt['input_frames']
    del temp_ckpt

    # then init correctly sized model
    model = DeflickerCNN(input_frames=input_frames, num_res_blocks=num_res_blocks, hidden_channels=hidden_channels)
    model.to(device)

    if verbose:
        print(f"Instantiated with: Input frames={input_frames}, Depth={num_res_blocks}, Width={hidden_channels}")

    # oddity of recycling code
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.LEARNING_RATE)
    if verbose:
        print(f"Loading previous model from {model_path}")
    epoch = load_checkpoint(model, optimizer, model_path, device)
    del optimizer
    model.eval()
    if verbose:
        print(f"Model loaded. It was trained to {epoch} epochs.")

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {input_video_path}")
        return

    # some help from Gemini
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # make and pad the queue
    window = deque(maxlen=input_frames)
    
    # Read first frame (t=0)
    ret, frame0 = cap.read()
    if not ret:
        print("Error: Could not read first frame.")
        return
    
    frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
    last_valid_frame = frame0

    i = 0
    # Fill start padding, end with frame 0
    for _ in range(-1, input_frames // 2):
        window.append(frame0)
        i += 1

    if verbose:
        print(f"Added {i} padding frames in the beginning")
    
    # Fill lookahead frames 
    i = 0
    for _ in range(-1, input_frames // 2 - 1):
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            last_valid_frame = frame
            window.append(frame)
        else:
            window.append(last_valid_frame)
        i += 1

    if verbose:
        print(f"Added {i} frames to fill the queue")
        print(f"Current queue size: {len(window)}")

    if len(window) != input_frames:
        print(f"Error: the sliding window {len(window)} is not the same size as input frames {input_frames}")
        exit(-1)
        

    with torch.no_grad():
        for i in track(range(total_frames), description="[green]Processing video . . .[/green]"):

            # 1. Convert window to tensor
            # window is a deque of 5 numpy arrays [H, W, 3]
            window_np = numpy.stack(window)
            tensor = torch.from_numpy(window_np).permute(0, 3, 1, 2).float() / 255.0
            input_tensor = tensor.reshape(1, -1, height, width).to(device)

            # 2. Forward Pass
            output = model(input_tensor)

            # 3. Post-process & Write
            output_np = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
            output_np = numpy.clip(output_np * 255.0, 0, 255).astype(numpy.uint8)
            output_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)
            out_writer.write(output_bgr)

            # --- Sliding Step ---
            # Prepare window for the NEXT iteration (output i+1).
            # We need to shift left and append the frame at position (i+1) + 2 = i+3.
            # Only read if we haven't reached the end of the loop logic.
            if i < total_frames - 1:
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    last_valid_frame = frame
                    window.append(frame)
                else:
                    # End of stream, pad with last known frame
                    window.append(last_valid_frame)
 
    cap.release()
    out_writer.release()
    print(f"Done! Saved filtered video to: {output_path}")

     
if __name__ == "__main__":
    # crude usage
    if len(sys.argv) != 4:
        print("usage: python3 smoother_test.py input_video_path output_video_path model_path")
        exit(-1)
    main(
        input_video_path=sys.argv[1],
        output_path=sys.argv[2],
        model_path=sys.argv[3],
    )

    
