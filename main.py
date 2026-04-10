#!/usr/bin/env python3

import sys
from rich.console import Console
import argparse

from model_utils import get_model_names, load_model

console = Console()

# File arguments
# Up here to make -h and -v snappy

# MAGIC NUMBERS
# @TODO: tune these parameters
PATIENCE = 10
BOUNDS_RECALCULATION = 100
VALID_EXTENSIONS = [".avi", ".mp4", ".mov"]
VALID_MODEL_SIZES = ["96", "128"]


parser = argparse.ArgumentParser()
parser.add_argument("--model", "-m", help="path to model", default="")
parser.add_argument("--model-name", help="override checkpoint model architecture", choices=get_model_names())
parser.add_argument("--input", "-i", help="input path (include extension)", default="")
parser.add_argument("--output", "-o", help="output path (include extension)", default="")
parser.add_argument("--debug", "-d", help="enable debug", action='store_true')
parser.add_argument("--verbose", "-V", help="enable verbose")
parser.add_argument("--version", "-v", action='version', version='No versions yet, but this is prerelease')
parser.add_argument("--iterations", "-I", help="run the model ITERATIONS times, default 1", default=1, type=int)
parser.add_argument("--blend", help="the amount of alpha to use \"DDPM\" with, default 0.0", default=0.0, type=float)
parser.add_argument("--device", help="use custom device DEVICE", default="")
parser.add_argument("--smoother", "-s", help="path to smoother model; if none, no smoothing", default="")
args = parser.parse_args()

from rich.progress import track
import os
# import re
import time
import torch
from PIL import Image
import numpy
import cv2

# These are the slow ones to import

# import ffmpeg_wrapper
import test
from cloud_transform import (
    generate_perlin_noise_map,
    make_alpha_image,
    crop_to_center_circle,
    get_random_valid_coords,
    colorize_array,
)
from random import randint
# import constructor
# import cv_composer

arg_input = args.input
arg_output = args.output
arg_custom_model_path = args.model
arg_model_name = args.model_name
iterations = args.iterations
BLEND_STRENGTH = args.blend
debug = args.debug
arg_smoother = args.smoother

if not iterations > 0:
    console.print(f"Error: iterations must be larger than 1, currently at {iterations}.")
    sys.exit(1)

if not (0.0 <= BLEND_STRENGTH and BLEND_STRENGTH <= 1.0):
    console.print(f"Error: blend strength must be between 0.0 and 1.0, currently at {BLEND_STRENGTH}",
                    style="bold red")
    console.print("Clamping blend strength to fit in range . . .", style="yellow")
    BLEND_STRENGTH = min(1.0, BLEND_STRENGTH)
    BLEND_STRENGTH = max(0.0, BLEND_STRENGTH)
    console.print(f"Blend strength is now {BLEND_STRENGTH}")


def resource_path(relative_path):
    """Weirdness for trying to package, too much effort to remove.

    @TODO: remove.
    """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def rmdir(directory):
    """Recursively removes a directory."""
    for item in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, item)):
            # print(f"Removing {os.path.join(directory, item)}")
            os.remove(os.path.join(directory, item))
        else:
            rmdir(os.path.join(directory, item))
            # subdirectories automatically remove themselves
    os.rmdir(directory)


def response_loop(initial_message, options=["y", "n"]):
    """Take an initial message and returns a "y" or a "n" in a do while loop.

    Args:
        initial_message (str): The initial message to be printed. If None, print nothing.
        options (list[str]): A list of valid options

    Returns:
        lowercase char in options, by default "y" or "n"

    Raises:
        -1 if there are too many incorrect attempts, set by PATIENCE. If PATIENCE == -1, it will not check.
    """
    if initial_message:
        response = console.input(initial_message + "\n")
    i = 0
    while True:
        if response.lower() in options:
            return response.lower()
        if i >= PATIENCE and PATIENCE != -1:
            console.print("I give up.", style="bold red")
            sys.exit(1)
        i = i + 1
        response = console.input(
            f"[red]Not a valid response. Please only input[/red] [blue]{options}[/blue]\n"
        )


def main(
    device="cuda" if torch.cuda.is_available() else "cpu",
    # All of the other variables are defined at the top of the file
    # Smart? Probably not, it hurts my C brain
    # Does it work? Yes.
):
    """Do the thing."""
    # print title
    f = open(resource_path("title.txt"), "r")
    title = f.read()
    console.print(title, style="bold green")
    console.print(
        "[italic]A system for enhancing videos of the Aurora Borealis[/italic]",
        style="white",
    )
    console.print()
    console.print(
        "To exit at any time, hit [bold][red]C-c[/red][/bold] (or whatever the exit shortcut is for your terminal)"
    )
    console.print()
    console.print("*" * 50)
    console.print()

    video_path = ""
    if arg_input:
        video_path = arg_input
        if not os.path.exists(video_path):
            console.print(f"{video_path} is not a valid video path", style="red")
            sys.exit(1)
    else:
        while True:
            video_path = console.input(
                "[green]What is the path of the video you want to upscale?[/green]\n"
            )

            if not os.path.exists(video_path):
                console.print(f"{video_path} is not a valid video path", style="red")
                console.print(
                    "There is a known issue where video paths with spaces trigger this warning. If that's this, I am sorry.",
                    style="yellow",
                )
                continue

            _, extension = os.path.splitext(video_path)
            if extension not in VALID_EXTENSIONS:
                console.print(
                    f"Warning: Boredealis does not support {extension} files",
                    style="yellow",
                )
                response = response_loop(
                    initial_message=r"[yellow]Are you sure you want to try anyways?[/yellow][blue] \[y/N] [/blue]",
                )
                if response == "n":
                    continue

            break
    if arg_output:
        save_path = arg_output
    else:
        save_path = console.input(
            "[green]Where do you want your video saved as (include path and extension)?[/green]\n"
        )
        _, extension = os.path.splitext(save_path)
        if extension not in VALID_EXTENSIONS:
            console.print(
                f"Warning: Boredealis does not support {extension} files. Things might go wrong.",
                style="yellow",
            )

    # get model size

    if arg_custom_model_path:
        model_path = arg_custom_model_path
    else:
        while True:
            # response = console.input(
            #     f"[green]What size model do you want to use?[/green][blue] {VALID_MODEL_SIZES} [/blue]\n"
            # )
            # if response not in VALID_MODEL_SIZES:
            #     console.print("Error: not a valid model size", style="red")
            #     continue
            # break
            response = console.input(
                "[green]What is the path to the model you want to use?[/green]\n")
            if not os.path.exists(response):
                console.print(
                    "Warning: model path does not exist.", style="bold red"
                )
                continue
            model_path = response
            break
        console.print()
        console.print("*" * 50)

    if not os.path.exists(model_path):
        console.print(
            "Error: model path does not exist. Exiting . . .", style="bold red"
        )
        sys.exit(1)

    filter_video_in_a_pipeline(model_path, video_path, save_path, device)


def filter_video_in_a_pipeline(
    model_path: str,
    video_path: str,
    save_path: str,
    device: str,
    model_name: str | None = None,
) -> int:
    """Filter a video but good.

    Args:
        model_path (str): Path to the model to use.
        video_path (str): Path to the video to filter.
        save_path (str): Path to save the filtered video to.
        device (str): device to use (e.g. "cuda:0" or "cuda" or "cpu").

    Returns:
        0 if all good
        1 if debug complete
        2 if error
    """
    start_time = time.time()

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    model = load_model(
        model_path,
        verbose=args.verbose,
        device=device,
        model_name=model_name,
    )

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    alpha_image = make_alpha_image(
        blend_strength=BLEND_STRENGTH,
        scale=300,
    )
    fake_clouds_map = generate_perlin_noise_map(
        size=width,
        iterations=randint(1, 2),
        scale=500,
    )
    # just for safety, in case something goes *terribly* awry
    fake_clouds = colorize_array(
        fake_clouds_map, lower_bound=(0, 0, 0), upper_bound=(255, 255, 255)
    )
    r, g, b = fake_clouds.split()

    fake_clouds = Image.merge("RGBA", (r, g, b, alpha_image))

    for i in track(range(total_frames),
                   description=f"[green]Processing video {video_path}. . .[/green]"):
        loop_start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if i % BOUNDS_RECALCULATION == 0:
            # recalculate bounds every BOUNDS_RECALCULATION frames
            # higher = longer compute time
            # can not be zero
            cropped_image = crop_to_center_circle(image_pil)
            upper_bound = get_random_valid_coords(cropped_image, boost=150)
            lower_bound = get_random_valid_coords(cropped_image, boost=-10)
            fake_clouds = colorize_array(
                fake_clouds_map, lower_bound=lower_bound, upper_bound=upper_bound
            )
            r, g, b = fake_clouds.split()
            if debug:
                fake_clouds.show()

            fake_clouds = Image.merge("RGBA", (r, g, b, alpha_image))
            if debug:
                fake_clouds.show()

        for _ in range(0, iterations):
            image_pil = test.test_with_ram(
                image=image_pil,
                preloaded_model=model,
                device=device,
                verbose=args.verbose,
                fake_clouds=fake_clouds,
            )

        filtered_frame_bgr = cv2.cvtColor(numpy.array(image_pil), cv2.COLOR_RGB2BGR)

        out.write(filtered_frame_bgr)
        if debug:
            print(f"Time to complete a loop: {time.time() - loop_start_time}")
            image_pil.show()
            print("Debug complete!")
            return 1

    cap.release()
    out.release()

    console.print(
        f"Done! Finished in {(time.time()-start_time):.2f} seconds.", style="green"
    )
    return 0


if __name__ == "__main__":
    if not args.device:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    output = args.output
    temp = "tmp/tmp_filtered.mp4"

    if args.smoother:
        if os.path.exists("tmp"):
            console.print("[yellow]Warning: temporary directory tmp exists. Removing . . .")
            rmdir("tmp")
        os.makedirs(name="tmp")
        args.output = temp

    if args.input and args.output and args.model and args.device:
        filter_video_in_a_pipeline(
            model_path=args.model,
            video_path=args.input,
            save_path=args.output,
            device=args.device,
            model_name=args.model_name,
        )
    else:
        main(device=args.device)

    if args.smoother:
        import smoother_test
        smoother_test.main(
            model_path=args.smoother,
            input_video_path=temp,
            output_path=output,
            device=args.device,
            verbose=args.verbose
        )
        rmdir("tmp")
