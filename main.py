#!/usr/bin/env python3

import sys
from rich.console import Console

console = Console()

# File arguments
# Up here to make -h and -v snappy

arg_input = None
arg_output = None
arg_filters = None
debug = False
arg_custom_model_path = None
remove_tmp = True
use_disk = False
iterations = 1

for arg in sys.argv[1:]:
    # help
    if arg.endswith("help") or arg == "-h":
        f = open("help.txt", "r")
        help_file = f.read()
        print(help_file)
        sys.exit(0)
    # version
    elif arg.endswith("version") or arg == "-v":
        print("This doesn't have versions, lmao")
        print("But this is a prerelease version")
        sys.exit(0)
    # debug mode
    elif arg.endswith("debug") or arg == "-d":
        debug = True
    # remove tmp
    elif arg.startswith("--remove-tmp"):
        remove_tmp = True if arg.lower().endswith("true") else False
        # print(remove_tmp)
    elif arg == "-r" or arg == "-rt":
        remove_tmp = False
        # print(remove_tmp)
    # argument flags
    elif arg.startswith("--input="):
        arg_input = arg[8:]
        # print(argv_1)
    elif arg.startswith("-i="):
        arg_input = arg[3:]

    elif arg.startswith("--output="):
        arg_output = arg[9:]
        # print(argv_2)
    elif arg.startswith("-o="):
        arg_output = arg[3:]

    elif arg.startswith("--filters="):
        arg_filters = int(arg[10:])
        # print(argv_3)
    elif arg.startswith("-f="):
        arg_filters = arg[3:]
    elif arg.startswith("--custom-model-path="):
        arg_custom_model_path = arg[20:]
    elif arg.startswith("-c=") or arg.startswith("-m="):
        arg_custom_model_path = arg[3:]
    elif arg == "-D" or arg == "--disk":
        use_disk=True
    elif arg.startswith("-I="):
        iterations = int(arg[3:])
    elif arg.startswith("--iterations="):
        iterations = int(arg[13:])

    # I am so sorry to the else-if gods, but I don't want to refactor this
    # @TODO: refactor this
    # catch all
    else:
        console.print(
            f"Error: {arg} is not a valid flag. Please check --help for what flags are valid",
            style="bold red",
        )
        sys.exit(1)

if debug:
    for arg in sys.argv[1:]:
        console.print(f"Arg {arg}")


if not iterations > 0:
    console.print(f"Error: iterations must be larger than 1, currently at {iterations}.")
    sys.exit(1)


# if len(sys.argv) >= 2:
#     argv_1 = sys.argv[1]
#     if argv_1.lower().endswith("help") or argv_1 == "-h":
#         f = open("help.txt", "r")
#         help_file = f.read()
#         print(help_file)
#         sys.exit(0)
#     if argv_1.lower().endswith("version") or argv_1 == "-v":
#         print("This doesn't have versions, lmao")
#         print("But this is a prerelease version")
#         sys.exit(0)
# if len(sys.argv) >= 3:
#     argv_2 = sys.argv[2]
# if len(sys.argv) >= 4:
#     argv_3 = sys.argv[3]

from rich.progress import track
import os
import re
import time
import torch
from PIL import Image
import numpy
import cv2

# These are the slow ones to import

# import ffmpeg_wrapper
import test
# import constructor
# import cv_composer

PATIENCE = 10
VALID_EXTENSIONS = [".avi", ".mp4", ".mov"]
VALID_MODEL_SIZES = ["96", "128"]


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
    # For macOS
    if torch.backends.mps.is_available():
        device = "mps"

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

    # get video path
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
    if arg_output:
        save_path = arg_output
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

    if arg_filters:
        response = arg_filters
        model_path = os.path.join("models", f"{response}_checkpoint_best.pth")
    elif arg_custom_model_path:
        model_path = arg_custom_model_path
    else:
        while True:
            response = console.input(
                f"[green]What size model do you want to use?[/green][blue] {VALID_MODEL_SIZES} [/blue]\n"
            )
            if response not in VALID_MODEL_SIZES:
                console.print("Error: not a valid model size", style="red")
                continue
            break
        console.print()
        console.print("*" * 50)

        model_path = os.path.join("models", f"{response}_checkpoint_best.pth")

    if not os.path.exists(model_path):
        console.print(
            "Error: model path does not exist. Exiting . . .", style="bold red"
        )
        sys.exit(1)

    if use_disk:
        filter_video(model_path, video_path, save_path, device)
    else:
        filter_video_in_a_pipeline(model_path, video_path, save_path, device)

def filter_video(model_path, video_path, save_path, device):
    # Make temporary directories
    start_time = time.time()
    tmp = resource_path("tmp")
    tmp_original_images = resource_path(os.path.join("tmp", "original_images"))
    tmp_filtered_images = resource_path(os.path.join("tmp", "filtered_images"))
    if os.path.exists(tmp):
        console.print("Warning: tmp directory exists. Removing . . .", style="yellow")
        rmdir(tmp)
    os.makedirs(tmp)
    os.makedirs(tmp_original_images)
    os.makedirs(tmp_filtered_images)
    console.print("Converting the video into images . . .", style="green")
    # ffmpeg_wrapper.convert_to_images(video_path, tmp_original_images)
    fps = cv_composer.decompose_video(video_path, tmp_original_images, verbose=False)

    # print(f"FPS: {fps}")

    model = test.load_model(model_path, verbose=debug, device=device)
    console.print("Upscaling images . . .", style="green")
    for image in track(os.listdir(tmp_original_images), description="Processing video . . ."):
        image_path = os.path.join(tmp_original_images, image)
        filtered_image = test.test(
            image_path=image_path, preloaded_model=model, verbose=debug, device=device
        )
        match = re.match(r"(\w+)_(\d+)\.png", image)
        # print(match.group(2))
        save_name = f"filtered_{match.group(2)}.png"
        # print(save_name)
        filtered_image.save(os.path.join(tmp_filtered_images, save_name))

    console.print("Putting images back together . . .", style="green")
    # constructor.convert_directory(input_dir=tmp_filtered_images, output_name=save_path)
    cv_composer.compose_video(tmp_filtered_images, save_path, fps=fps, verbose=debug)

    if remove_tmp:
        console.print("Cleaning up . . .", style="green")
        rmdir(tmp)

    console.print(
        f"Done! Finished in {(time.time()-start_time):.2f} seconds.", style="green"
    )

def filter_video_in_a_pipeline(model_path, video_path, save_path, device):
    start_time = time.time()

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    model = test.load_model(model_path, verbose=debug, device=device)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for _ in track(range(total_frames), description="[green]Processing video . . .[/green]"):
        ret, frame = cap.read()
        if not ret:
            break

        image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        for iteration in range(0, iterations):
            image_pil = test.test_with_ram(image=image_pil, preloaded_model=model, device=device, verbose=debug)

        filtered_frame_bgr = cv2.cvtColor(numpy.array(image_pil), cv2.COLOR_RGB2BGR)

        out.write(filtered_frame_bgr)

    cap.release()
    out.release()

    console.print(
        f"Done! Finished in {(time.time()-start_time):.2f} seconds.", style="green"
    )

main()
# my_rmdir("test_videos")
# if response_loop("Testing", ["a"]):
#     print("success!")
