import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.train_util import load_dict_from_csv, read_from_h5



class AudioCaptionDataset(Dataset):

    def __init__(self,
                 waveform: str,
                 label: str,
                 max_audio_length: float = None,
                 max_cap_length: int = None,
                 sample_rate: int = 32000,
                 ):
        self.aid_to_h5 = load_dict_from_csv(waveform, ("audio_id", "hdf5_path"))
        self.cache = {}
        self.data = json.load(open(label))
        if max_audio_length is not None:
            self.max_audio_len = int(max_audio_length * sample_rate)
        else:
            self.max_audio_len = None
        self.max_cap_length = max_cap_length

    def __getitem__(self, index):
        item = self.data[index]
        waveform = read_from_h5(item["audio_id"], self.aid_to_h5, self.cache)
        waveform = np.array(waveform, dtype=np.float32)
        if self.max_audio_len is not None and waveform.shape[0] > self.max_audio_len:
            start = random.randint(0, waveform.shape[0] - self.max_audio_len)
            waveform = waveform[start: start + self.max_audio_len]
        caption = item["tokens"]
        if self.max_cap_length is not None:
            caption = caption[:self.max_cap_length]
        return {
            "audiocap_id": item["audiocap_id"],
            "waveform": waveform,
            "caption": caption
        }

    def __len__(self):
        return len(self.data)


class AudioCaptionPhraseIndicesDataset(AudioCaptionDataset):

    def __getitem__(self, index):
        output = super().__getitem__(index)
        item = self.data[index]
        start_indices = []
        end_indices = []
        for phrase_item in item["phrases"]:
            start_indices.append(phrase_item["start_index"])
            end_indices.append(phrase_item["end_index"])
        output["start_indices"] = start_indices
        output["end_indies"] = end_indices
        return output
