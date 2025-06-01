#!/usr/bin/env python3

import sys

# File arguments
# Up here to make -h and -v snappy

argv_1 = None
argv_2 = None
argv_3 = None
if len(sys.argv) >= 2:
    argv_1 = sys.argv[1]
    if argv_1.lower().endswith("help") or argv_1 == "-h":
        f = open("help.txt", "r")
        help_file = f.read()
        print(help_file)
        sys.exit(0)
    if argv_1.lower().endswith("version") or argv_1 == "-v":
        print("This doesn't have versions, lmao")
        print("But this is a prerelease version")
        sys.exit(0)
if len(sys.argv) >= 3:
    argv_2 = sys.argv[2]
if len(sys.argv) >= 4:
    argv_3 = sys.argv[3]

from rich.console import Console
from rich.progress import track
import os
import re
import time

# These are the slow ones to import

import ffmpeg_wrapper
import test
import constructor

console = Console()
PATIENCE = 10
VALID_EXTENSIONS = [".avi", ".mp4", ".mov"]
VALID_MODEL_SIZES = ["32", "64", "128"]


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
    """Takes an initial message and returns a "y" or a "n" in a do while loop.

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
            sys.exit(-1)
        i = i + 1
        response = console.input(
            f"[red]Not a valid response. Please only input[/red] [blue]{options}[/blue]\n"
        )


def main(argv_1=None, argv_2=None, argv_3=None):
    # print title
    f = open("title.txt", "r")
    title = f.read()
    console.print(title, style="bold green")
    console.print(
        "[italics]A system for enhancing videos of the Aurora Borealis[/italics]",
        style="white",
    )
    console.print()
    console.print(
        "To exit at any time, hit [bold][red]C-c[/red][/bold] (or whatever the exit shortcut is for your terminal)"
    )
    console.print()
    console.print("*" * 50)
    console.print()

    # get video path
    video_path = ""
    if argv_1:
        video_path = argv_1
    else:
        while True:
            video_path = console.input(
                "[green]What is the path of the video you want to upscale?[/green]\n"
            )

            if not os.path.exists(video_path):
                console.print(f"{video_path} is not a valid video path", style="red")
                continue

            filename, extension = os.path.splitext(video_path)
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
    if argv_2:
        save_path = argv_2
    else:
        save_path = console.input(
            "[green]Where do you want your video saved as (include path and extension)?[/green]\n"
        )
        filename, extension = os.path.splitext(save_path)
        if extension not in VALID_EXTENSIONS:
            console.print(
                f"Warning: Boredealis does not support {extension} files. Things might go wrong.",
                style="yellow",
            )

    # get model size
    # @TODO: make 32, 64, and 128 models available. And have them good

    if argv_3:
        response = argv_3
    else:
        console.print("*" * 50)
        while True:
            response = console.input(
                f"[green]What size model do you want to use?[/green][blue] {VALID_MODEL_SIZES} [/blue]\n"
            )
            if response not in VALID_MODEL_SIZES:
                console.print("Error: not a valid model size", style="red")
                continue
            break

    match response:
        case "32":
            model_path = os.path.join("32_filters_models", "checkpoint_best.pth")
            console.print("Error: this does not work (yet).", style="bold red")
            sys.exit(-1)
            # TODO: move this to the proper directory
        case "64":
            model_path = os.path.join(
                "model_milestones", "first_working_64_filter_model.pth"
            )
        case "128":
            model_path = os.path.join(
                "model_milestones", "128_filters_checkpoint_5.pth"
            )
        case _:
            console.print(
                f"Error: {response} is not a valid number of filters", style="bold red"
            )
            return -1

    # make temporary directories
    start_time = time.time()
    tmp = "tmp"
    tmp_original_images = os.path.join("tmp", "original_images")
    tmp_filtered_images = os.path.join("tmp", "filtered_images")
    if os.path.exists(tmp):
        console.print("Warning: tmp directory exists. Removing . . .", style="yellow")
        rmdir(tmp)
    os.makedirs(tmp)
    os.makedirs(tmp_original_images)
    os.makedirs(tmp_filtered_images)
    console.print("Converting the video into images . . .", style="green")
    ffmpeg_wrapper.convert_to_images(video_path, tmp_original_images)

    model = test.load_model(model_path, verbose=True)
    console.print("Upscaling images . . .", style="green")
    for image in track(os.listdir(tmp_original_images)):
        image_path = os.path.join(tmp_original_images, image)
        filtered_image = test.test(
            image_path=image_path, preloaded_model=model, verbose=False
        )
        match = re.match(r"(\w+)_(\d+)\.png", image)
        # print(match.group(2))
        save_name = f"filtered_{match.group(2)}.png"
        # print(save_name)
        filtered_image.save(os.path.join(tmp_filtered_images, save_name))

    console.print("Putting images back together . . .", style="green")
    constructor.convert_directory(input_dir=tmp_filtered_images, output_name=save_path)

    console.print("Cleaning up . . .", style="green")
    rmdir(tmp)

    console.print(
        f"Done! Finished in {time.time()-start_time:2f} seconds.", style="green"
    )


main(argv_1=argv_1, argv_2=argv_2, argv_3=argv_3)
# my_rmdir("test_videos")
# if response_loop("Testing", ["a"]):
#     print("success!")
