#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import glob
import json
import argparse
import logging


parser = argparse.ArgumentParser()
parser.add_argument("input_labels", type=str, nargs="+")
parser.add_argument("output_label", type=str)

args = parser.parse_args()

logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=logfmt)

output_data = []
for fname in args.input_labels:
    if Path(fname).is_file():
        data = json.load(open(fname))
        output_data += data
    elif "*" in fname:
        for f in glob.glob(fname):
            data = json.load(open(f))
            output_data += data
    else:
        raise Exception(f"cannot recognize input {fname}")

json.dump(output_data, open(args.output_label, "w"), indent=4)
logging.info("new csv has " + str(len(output_data)) + " items")
