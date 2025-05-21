#!/usr/bin/env python3

import os
import ffmpeg
import re


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


# Thank you Perplexity for this. I did not want to deal with regex


def split_filename(filename):
    # This regex captures the filename and extension separately
    match = re.match(r"^(.*?)(\.[^.]*$|$)", filename)
    if match:
        name = match.group(1)
        ext = match.group(2)
        return name, ext
    else:
        return filename, ""


def convert_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file in os.listdir(input_dir):
        base, ext = split_filename(file)
        print(
            f"Converting {input_dir}/{base}{ext}"
        )  # Instead of {file} because I want to see the regex work
        convert_to_images(os.path.join(input_dir, file), output_dir, output_name=base)


if __name__ == "__main__":
    # test file
    # convert_to_images(
    #     "test/02032021_221508.avi", "test/images", output_name="02032021_221508"
    # )
    convert_directory("data/videos", "data/images")
