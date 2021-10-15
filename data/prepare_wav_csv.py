#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("audio_directory", type=str, help="directory containing input audio files")
parser.add_argument("output_csv", type=str, help="output wave csv filename")

args = parser.parse_args()

with open(args.output_csv, "w", newline="") as csv_file:
    writer = csv.writer(csv_file, delimiter="\t")
    writer.writerow(["audio_id", "file_name"])
    for file_name in Path(args.audio_directory).iterdir():
        writer.writerow([file_name.name, str(file_name.absolute())])
    
