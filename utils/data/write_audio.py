import json
from typing import Dict
from pathlib import Path
import h5py
import soundfile as sf
import fire
import numpy as np
import pandas as pd
from tqdm import tqdm



def read_from_h5(key: str, key_to_h5: Dict, cache: Dict):
    hdf5_path = key_to_h5[key]
    if hdf5_path not in cache:
        cache[hdf5_path] = h5py.File(hdf5_path, "r")
    return cache[hdf5_path][key][()]


def write(waveform_csv,
          output_dir,
          sample_rate=32000,
          grounding_label=None):
    df = pd.read_csv(waveform_csv, sep="\t")
    aid_to_h5 = dict(zip(df["audio_id"], df["hdf5_path"]))
    h5_cache = {}

    if grounding_label is not None:
        data = json.load(open(grounding_label))
        audio_ids = []
        for audio_item in data:
            audio_ids.append(audio_item["audio_id"])
        audio_ids = set(audio_ids)
    else:
        audio_ids = None

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    for audio_id, hdf5_path in tqdm(aid_to_h5.items()):
        if audio_ids is not None and audio_id not in audio_ids:
            continue
        waveform = read_from_h5(audio_id, aid_to_h5, h5_cache)
        waveform = np.array(waveform, dtype=np.float32)
        sf.write(str(output_dir / audio_id), waveform, sample_rate)


if __name__ == "__main__":
    fire.Fire(write)
