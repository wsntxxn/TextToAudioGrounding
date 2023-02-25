import math
import pickle
import json
import random
from typing import List, Dict, Tuple
import numpy as np
import torch
import h5py
from torch.utils.data import Dataset

from utils.build_vocab import Vocabulary
from utils.train_util import load_dict_from_csv


def read_from_h5(key: str, key_to_h5: Dict, cache: Dict):
    hdf5_path = key_to_h5[key]
    if hdf5_path not in cache:
        cache[hdf5_path] = h5py.File(hdf5_path, "r")
    return cache[hdf5_path][key][()]


class AudioCaptionDataset(Dataset):

    def __init__(self,
                 waveform: str,
                 label: str,
                 max_audio_length: float = None,
                 sample_rate: int = 32000,
                 ):
        self.aid_to_h5 = load_dict_from_csv(waveform, ("audio_id", "hdf5_path"))
        self.cache = {}
        self.data = json.load(open(label))
        if max_audio_length is not None:
            self.max_audio_len = int(max_audio_length * sample_rate)
        else:
            self.max_audio_len = None

    def __getitem__(self, index):
        item = self.data[index]
        waveform = read_from_h5(item["audio_id"], self.aid_to_h5, self.cache)
        waveform = np.array(waveform, dtype=np.float32)
        if self.max_audio_len is not None and waveform.shape[0] > self.max_audio_len:
            start = random.randint(0, waveform.shape[0] - self.max_audio_len)
            waveform = waveform[start: start + self.max_audio_len]
        return {
            "audiocap_id": item["audiocap_id"],
            "waveform": waveform,
            "caption": item["tokens"]
        }

    def __len__(self):
        return len(self.data)
