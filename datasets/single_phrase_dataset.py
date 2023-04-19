import pickle
import math
import json
import h5py
import numpy as np
import torch
from typing import List, Dict, Tuple
from torch.utils.data import Dataset

from utils.train_util import load_dict_from_csv


def read_from_h5(key: str, key_to_h5: Dict, cache: Dict):
    hdf5_path = key_to_h5[key]
    if hdf5_path not in cache:
        cache[hdf5_path] = h5py.File(hdf5_path, "r")
    return cache[hdf5_path][key][()]


class AudioPhraseEvalDataset(Dataset):

    def __init__(self,
                 waveform,
                 label,
                 sample_rate: int = 32000):
        self.aid_to_h5 = load_dict_from_csv(waveform, ("audio_id", "hdf5_path"))
        self.cache = {}
        self.data = json.load(open(label))
        self.sample_rate = sample_rate
        self.generate_index()
        
    def generate_index(self):
        self.idxs = []
        for audio_idx, audio_item in enumerate(self.data):
            for phrase_idx, phrase_item in enumerate(audio_item["phrases"]):
                self.idxs.append((audio_idx, phrase_idx))

    def __getitem__(self, index):
        audio_idx, phrase_idx = self.idxs[index]
        audio_item = self.data[audio_idx]
        caption = audio_item["tokens"]
        audio_id = audio_item["audio_id"]
        phrase_item = audio_item["phrases"][phrase_idx]
        waveform = read_from_h5(audio_id, self.aid_to_h5, self.cache)
        waveform = np.array(waveform, dtype=np.float32)
        # text = [self.vocabulary(token) for token in phrase_item["phrase"].split()]
        phrase = phrase_item["phrase"]
        output = {
            "audio_id": audio_id,
            "audiocap_id": audio_item["audiocap_id"],
            "start_index": phrase_item["start_index"],
            "end_index": phrase_item["end_index"],
            "waveform": waveform,
            "phrase": phrase,
            "caption": caption
        }
        return output

    def __len__(self):
        return len(self.idxs)


class AudioPhraseDataset(AudioPhraseEvalDataset):

    def __init__(self,
                 waveform: str,
                 label: str,
                 time_resolution: float = 0.02,
                 sample_rate: int = 32000):
        super().__init__(waveform, label, sample_rate)
        self.time_resolution = time_resolution

    def __getitem__(self, index):
        output = super().__getitem__(index)
        audio_idx, phrase_idx = self.idxs[index]
        audio_item = self.data[audio_idx]
        caption = self.data[audio_idx]["tokens"]
        phrase_item = audio_item["phrases"][phrase_idx]
        audio_duration = output["waveform"].shape[0] / self.sample_rate
        n_frame = math.floor(audio_duration / self.time_resolution) + 1
        label = np.zeros(n_frame, dtype=int)
        for start, end in phrase_item["segments"]:
            onset = round(start / self.time_resolution)
            offset = round(end / self.time_resolution)
            label[onset: offset] = 1
        output["label"] = label
        output["caption"] = caption
        return output


