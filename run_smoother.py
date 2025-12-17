#!/usr/bin/env python3

import argparse
import torch
import smoother_test

# @TODO: switch EVERYTHING to argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model")
parser.add_argument("--input")
parser.add_argument("--output")
parser.add_argument("--device")
args = parser.parse_args()

smoother_test.main(
    model_path=args.model,
    input_video_path=args.input,
    output_path=args.output,
    device=device,
    verbose=True,
    debug=False,
)
