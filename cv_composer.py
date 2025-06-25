#!/usr/bin/env python3
import os
import cv2
import re

def decompose_video(video_path, output_dir, output_stem="output", verbose=False):
    """Turns a video into individual images in output_dir

    Args:
        video_path (str): Path to the location of the input video
        output_dir (str): Path to the directory to dump the images into
        output_stem (str): Base name of the video, formatted <output_name>_%04d.png

    Returns:
        int: frames per second of the input video
        -1 on fail
    """
    if not os.path.exists(video_path):
        print(f"Error: video path does not exist: {video_path}")
        return -1

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if os.listdir(output_dir):
        print("Warning: output dir is not empty. Deleting . . .")
        rmdir(output_dir)
        os.mkdir(output_dir)

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_count = 1

    if verbose:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"\nVideo Info:")
        print(f"  Path: {video_path}")
        print(f"  FPS: {fps:.2f}")
        print(f"  Total Frames: {total_frames if total_frames > 0 else 'N/A (could be a stream)'}")
        print(f"  Resolution: {width}x{height}")
        print(f"  Saving frames to: {output_dir}")
        print(f"  Image format: .png")

    while True:
        ret, frame = cap.read()

        # no next frame
        if not ret:
            return

        output_filename = f"{output_stem}_{str(frame_count).zfill(4)}.png"
        # if verbose:
        #     print(f"Writing {output_filename}")

        cv2.imwrite(os.path.join(output_dir, output_filename), frame)

        frame_count += 1

    cap.release()

    return fps


def compose_video(image_dir, output_name, fps=25, codec="mp4v", verbose=True):
    """Composes a video out of a directory of images.

    Args:
        image_dir (str): Path of the image directory to be converted
        output_name (str): Path of the video to be saved to, extension and all
        fps (int): Framerate of the video to to be saved. By default it is 25
        codec (str): Codec to be used by OpenCV, by default avc1
        verbose (Bool): Lord help you, it prints too much info.

    Returns:
        Nothing on success
        -1 on failure
    """
    if fps == None:
        print(f"Warning: FPS value of {fps} is not valid. Using default value of 25.")
        fps = 25

    if not os.path.exists(image_dir):
        print(f"Error: image_dir does not exist: {image_dir}")
        return -1

    if os.path.exists(output_name):
        print(f"Warning: {output_name} exists. Removing.")
        os.remove(output_name)

    images = []

    for image in os.listdir(image_dir):
        # if verbose:
        #     print(f"Matching {image} in {image_dir}")
        match = re.match(r"(\w+)_(\d+)\.png", image)
        if not match:
            print(f"Error: {image} did not match the regex.")
            return -1
        # if verbose:
        #     print(match.group(2))


        images.append(image)

    images.sort(key=lambda f: int(re.match(r"(\w+)_(\d+)\.png", f).group(2)))

    # if verbose:
    #     print(images)

    first_frame = cv2.imread(os.path.join(image_dir, images[0]))
    if verbose:
        print(f"First frame: {first_frame.shape}")
    height, width, channels = first_frame.shape
    size = (width, height)

    out = cv2.VideoWriter(output_name,
                          cv2.VideoWriter_fourcc(*codec),
                          fps,
                          size)

    for frame in images:
        frame_path = os.path.join(image_dir, frame)
        frame = cv2.imread(frame_path)
        out.write(frame)
        # if verbose:
        #     print(f"Stitched frame {frame_path}")

    out.release()
    if verbose:
        print(f"Video saved to {output_name}")

def rmdir(directory):
    """Recursively removes a directory. Copied from main.py"""
    for item in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, item)):
            # print(f"Removing {os.path.join(directory, item)}")
            os.remove(os.path.join(directory, item))
        else:
            rmdir(os.path.join(directory, item))
            # subdirectories automatically remove themselves
    os.rmdir(directory)
