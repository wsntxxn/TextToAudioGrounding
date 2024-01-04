#! /usr/bin/env python3

import argparse
import librosa
import pandas as pd
from tqdm import tqdm
from pypeln import process as pr
import h5py


def extract_duration(row):
    idx, item = row
    aid = item["audio_id"]
    if "file_name" in item:
        fname = item["file_name"]
        duration = librosa.core.get_duration(filename=fname)
    elif "hdf5_path" in item:
        fname = item["hdf5_path"]
        with h5py.File(fname, "r") as hf:
            try:
                length = hf[aid].shape[0]
            except KeyError:
                length = hf["Y" + aid + ".wav"].shape[0]
        duration = length / sample_rate
    return aid, duration


parser = argparse.ArgumentParser()
parser.add_argument("wav_csv", type=str)
parser.add_argument("output", type=str)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--sample_rate", type=int)

args = parser.parse_args()
wav_df = pd.read_csv(args.wav_csv, sep="\t")
sample_rate = args.sample_rate

output_data = []
with tqdm(total=wav_df.shape[0], ascii=True) as pbar:
    for aid, duration in pr.map(extract_duration,
                                wav_df.iterrows(),
                                workers=args.num_workers,
                                maxsize=4):
        output_data.append({
            "audio_id": aid,
            "duration": duration
        })
        pbar.update()

pd.DataFrame(output_data).to_csv(args.output, sep="\t",
                                 index=False, float_format="%.3f")
