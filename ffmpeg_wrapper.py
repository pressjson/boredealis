#!/usr/bin/env python3

import os
import ffmpeg


def convert_to_images(input, output_path, output_name="output"):
    """Converts an input file (usually .avi) into a series of images at output_path with a default name of "output_%04d.png"

    args:
        input: path to the input file
        output_path: the path to the dir for the images to be output to
        output_name: the name of the file will be "output_name_%04d.png"

    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    ffmpeg.input(input).output(
        os.path.join(output_path, output_name + "_%04d.png")
    ).run()


if __name__ == "__main__":
    # test file
    convert_to_images(
        "test/02032021_221508.avi", "test/images", output_name="02032021_221508"
    )
