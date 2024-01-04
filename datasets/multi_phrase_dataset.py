import pickle
import json
from typing import List, Dict, Tuple
import random

import numpy as np
import torch
import h5py
from torch.utils.data import Dataset
from sklearn.metrics.pairwise import cosine_similarity

import sys
import os
sys.path.insert(1, os.path.join(os.path.dirname(__file__), ".."))
from utils.train_util import load_dict_from_csv


def read_from_h5(key: str, key_to_h5: Dict, cache: Dict):
    hdf5_path = key_to_h5[key]
    if hdf5_path not in cache:
        cache[hdf5_path] = h5py.File(hdf5_path, "r")
    return cache[hdf5_path][key][()]


class AudioSamplePhrasesDataset(Dataset):

    def __init__(self,
                 waveform: str,
                 label: str,
                 phrase_num: int,
                 fix_neg: bool,
                 neg_samp_stratg: str = "clustering",
                 max_phrase_length: int = None,
                 sample_rate: int = 32000,
                 max_audio_length: float = None,
                 **kwargs):
        self.aid_to_h5 = load_dict_from_csv(waveform, ("audio_id", "hdf5_path"))
        self.cache = {}
        if max_audio_length is not None:
            self.max_audio_len = int(max_audio_length * sample_rate)
        else:
            self.max_audio_len = None

        self.max_phrase_len = max_phrase_length

        self.data = json.load(open(label))
        self.phrase_num = phrase_num
        assert neg_samp_stratg in ("random", "clustering", "similarity")
        self.phrases = []
        fil_data = []
        for audio_item in self.data:
            excluded = True
            for phrase_item in audio_item["phrases"]:
                phrase = phrase_item["phrase"]
                if self.max_phrase_len is not None:
                    if len(phrase.split()) > self.max_phrase_len:
                        continue
                self.phrases.append(phrase_item["phrase"])
                excluded = False
            if not excluded:
                fil_data.append(audio_item)
        self.data = fil_data
        self.phrases = np.array(list(set(self.phrases)))
        self.phrase_to_idx = {phrase: idx for idx, phrase in 
            enumerate(self.phrases)}

        self.fix_neg = fix_neg
        if self.fix_neg:
            self.acid_to_neg = {}

        self.neg_samp_stratg = neg_samp_stratg
        if neg_samp_stratg == "clustering":
            assert "cluster_map" in kwargs, "cluster_map not provided"
            cluster_map = kwargs["cluster_map"]
            self.cluster_idx_to_phrases, self.phrase_to_cluster_idx = \
                self.read_cluster_map(cluster_map)
            self.cluster_idxs = np.array(list(self.cluster_idx_to_phrases.keys()))
            self.cluster_idx_to_idx = {cluster_idx: idx for idx, cluster_idx
                in enumerate(self.cluster_idxs)}
        elif neg_samp_stratg == "similarity":
            assert "phrase_embed" in kwargs, "phrase_embed not provided"
            assert "sim_threshold" in kwargs, "sim_threshold not provided"
            phrase_embed = kwargs["phrase_embed"]
            self.sim_threshold = kwargs["sim_threshold"]
            self.phrase_to_emb = pickle.load(open(phrase_embed, "rb"))
            if "negative_pool" in kwargs:
                self.phrases = []
                with open(kwargs["negative_pool"], "r") as reader:
                    for line in reader.readlines():
                        phrase = line.strip()
                        if self.max_phrase_len is not None:
                            if len(phrase.split()) > self.max_phrase_len:
                                continue
                        self.phrases.append(phrase)
                self.phrases = np.array(self.phrases)

            self.phrase_to_emb = {phrase: emb for phrase, emb in
                self.phrase_to_emb.items() if phrase in self.phrases}

            self.phrase_embs = []
            for phrase in self.phrases:
                self.phrase_embs.append(self.phrase_to_emb[phrase])
            self.phrase_embs = np.stack(self.phrase_embs)


    def read_cluster_map(self, cluster_map):
        mapping = json.load(open(cluster_map))
        phrase_to_cluster_idx = {}
        cluster_idx_to_phrases = {}
        for cluster_idx, phrases in mapping.items():
            cluster_idx = int(cluster_idx)
            filtered_phrases = []
            for phrase in phrases:
                phrase_to_cluster_idx[phrase] = cluster_idx
                if phrase in self.phrases:
                    if self.max_phrase_len is not None:
                        if len(phrase.split()) > self.max_phrase_len:
                            continue
                    filtered_phrases.append(phrase)
            cluster_idx_to_phrases[cluster_idx] = filtered_phrases
        return cluster_idx_to_phrases, phrase_to_cluster_idx


    def sample_negative_phrases(self, pos_phrases, audiocap_id):

        if self.fix_neg and audiocap_id in self.acid_to_neg:
            neg_phrases = [self.phrases[idx] for idx in self.acid_to_neg[audiocap_id]]
            return neg_phrases

        neg_phrase_num = max(0, self.phrase_num - len(pos_phrases))
        pos_idxs = [self.phrase_to_idx[phrase] for phrase in pos_phrases]
        cand_phrases = np.delete(self.phrases, pos_idxs)
        cand_phrase_idxs = np.delete(np.arange(len(self.phrases)), pos_idxs)
        if self.neg_samp_stratg == "random":
            neg_phrases = np.random.choice(cand_phrases,
                                           size=neg_phrase_num,
                                           replace=False)
        elif self.neg_samp_stratg == "similarity":
            pos_embs = self.phrase_embs[pos_idxs]

            neg_idxs = []
            np.random.shuffle(cand_phrase_idxs)
            pointer = 0
            while len(neg_idxs) < neg_phrase_num and pointer < len(cand_phrase_idxs):
                left = neg_phrase_num - len(neg_idxs)
                cand_phrase_idxs_part = cand_phrase_idxs[pointer: pointer + neg_phrase_num]
                cand_phrase_embs = self.phrase_embs[cand_phrase_idxs_part]
                sims = cosine_similarity(pos_embs, cand_phrase_embs)
                sims = sims.max(axis=0)
                cand_idxs = np.where(sims < self.sim_threshold)[0]
                neg_idxs.extend(cand_phrase_idxs_part[cand_idxs[:left]])
                pointer += neg_phrase_num

            while len(neg_idxs) < neg_phrase_num:
                left = neg_phrase_num - len(neg_idxs)
                neg_idxs.extend(neg_idxs[:left])

            neg_phrases = self.phrases[neg_idxs]
        elif self.neg_samp_stratg == "clustering":
            neg_phrases = []
            pos_cluster_idxs = list(set([self.phrase_to_cluster_idx[phrase]
                for phrase in pos_phrases]))
            cand_cluster_idxs = np.delete(
                self.cluster_idxs,
                [self.cluster_idx_to_idx[cluster_idx] for cluster_idx in
                    pos_cluster_idxs])
            if len(cand_cluster_idxs) >= neg_phrase_num:
                # more cluster centers than requested phrase number
                neg_cluster_idxs = np.random.choice(
                    cand_cluster_idxs,
                    size=neg_phrase_num,
                    replace=False)
                for neg_cluster_idx in neg_cluster_idxs:
                    if len(self.cluster_idx_to_phrases[neg_cluster_idx]) > 0:
                        neg_phrases.append(np.random.choice(
                            self.cluster_idx_to_phrases[neg_cluster_idx]))
            else:
                # fewer cluster centers than requested phrase number, some clusters have to be sampled multiple times
                cluster_samp_num = np.zeros_like(cand_cluster_idxs)
                cur_neg_phrase_num = neg_phrase_num
                while cur_neg_phrase_num > len(cand_cluster_idxs):
                    cluster_samp_num += 1
                    cur_neg_phrase_num -= len(cand_cluster_idxs)
                if cur_neg_phrase_num > 0:
                    cluster_samp_num[np.random.choice(
                        np.arange(len(cand_cluster_idxs)),
                        size=cur_neg_phrase_num,
                        replace=False)] += 1
                for idx, samp_num in enumerate(cluster_samp_num):
                    neg_cluster_idx = cand_cluster_idxs[idx]
                    samp_phrase = np.random.choice(
                        self.cluster_idx_to_phrases[neg_cluster_idx],
                        size=samp_num,
                        replace=False)
                    neg_phrases.extend(samp_phrase.tolist())

        while len(neg_phrases) < neg_phrase_num:
            neg_phrases.append(neg_phrases[-1])

        if self.fix_neg:
            self.acid_to_neg[audiocap_id] = [self.phrase_to_idx[phrase] for phrase in neg_phrases]
        
        return neg_phrases
            

    def __getitem__(self, index):
        audio_item = self.data[index]
        audio_id = audio_item["audio_id"]
        audiocap_id = audio_item["audiocap_id"]
        waveform = read_from_h5(audio_id, self.aid_to_h5, self.cache)
        waveform = np.array(waveform, dtype=np.float32)
        if self.max_audio_len is not None and \
                waveform.shape[0] > self.max_audio_len:
            start = random.randint(0, waveform.shape[0] - self.max_audio_len)
            waveform = waveform[start: start + self.max_audio_len]

        pos_phrases = []
        for phrase_item in audio_item["phrases"][:self.phrase_num]:
            phrase = phrase_item["phrase"]
            if self.max_phrase_len is not None:
                if len(phrase.split()) > self.max_phrase_len:
                    continue
            pos_phrases.append(phrase)
        
        neg_phrases = self.sample_negative_phrases(pos_phrases, audiocap_id)
        if isinstance(neg_phrases, np.ndarray):
            neg_phrases = neg_phrases.tolist()

        phrases = pos_phrases + neg_phrases

        label = np.array([1] * len(pos_phrases) + [0] * len(neg_phrases))
        return {
            "audiocap_id": audio_item["audiocap_id"],
            "waveform": waveform,
            "phrases": phrases,
            "label": label
        }

    def __len__(self):
        return len(self.data)


class SamplePhrasesCountDataset(AudioSamplePhrasesDataset):

    def __init__(self,
                 waveform: str,
                 label: str,
                 phrase_num: int,
                 fix_neg: bool,
                 neg_samp_stratg: str = "clustering",
                 max_phrase_length: int = None,
                 sample_rate: int = 32000,
                 max_audio_length: float = None,
                 **kwargs):
        super().__init__(waveform, label, phrase_num, fix_neg, neg_samp_stratg,
                         max_phrase_length, sample_rate, max_audio_length,
                         **kwargs)
        assert "phrase_count" in kwargs
        self.phrase_to_count = json.load(open(kwargs["phrase_count"]))

    def __getitem__(self, index):
        output = super().__getitem__(index)
        phrases = output["phrases"]
        counts = []
        if self.neg_samp_stratg == "similarity":
            for phrase in phrases:
                count = self.phrase_to_count[phrase]
                counts.append(count)
        output["counts"] = counts
        return output


class AudioCaptionPhrasesEvalDataset(Dataset):

    def __init__(self,
                 waveform: str,
                 label: str,
                 max_phrase_words: int = 10) -> None:
        self.aid_to_h5 = load_dict_from_csv(waveform, ("audio_id", "hdf5_path"))
        self.cache = {}
        self.data = json.load(open(label))
        self.generate_index()

    def generate_index(self):
        self.idxs = []
        for audio_idx, audio_item in enumerate(self.data):
            for phrase_idx, phrase_item in enumerate(audio_item["phrases"]):
                self.idxs.append((audio_idx, phrase_idx))

    def __getitem__(self, index):
        audio_idx, phrase_idx = self.idxs[index]
        audio_item = self.data[audio_idx]
        audio_id = audio_item["audio_id"]
        audiocap_id = audio_item["audiocap_id"]
        waveform = read_from_h5(audio_id, self.aid_to_h5, self.cache)
        waveform = np.array(waveform, dtype=np.float32)

        phrase_item = audio_item["phrases"][phrase_idx]
        phrase = phrase_item["phrase"]
        return {
            "audiocap_id": audiocap_id,
            "waveform": waveform,
            "phrases": [phrase,],
            "start_index": phrase_item["start_index"],
            "end_index": phrase_item["end_index"]
        }

    def __len__(self):
        return len(self.idxs)


class AudioCaptionPhrasesDataset(Dataset):

    def __init__(self,
                 waveform: str,
                 label: str,
                 max_phrase_words: int = 10) -> None:
        self.aid_to_h5 = load_dict_from_csv(waveform, ("audio_id", "hdf5_path"))
        self.cache = {}
        data = json.load(open(label))
        self.data = []
        self.max_phrase_words = max_phrase_words
        for audio_item in data:
            use_flag = False
            for phrase_item in audio_item["phrases"]:
                if len(phrase_item["phrase"].split()) <= self.max_phrase_words:
                    use_flag = True
                    break
            if use_flag:
                self.data.append(audio_item)

    def __getitem__(self, index):
        audio_item = self.data[index]
        audio_id = audio_item["audio_id"]
        audiocap_id = audio_item["audiocap_id"]
        waveform = read_from_h5(audio_id, self.aid_to_h5, self.cache)
        waveform = np.array(waveform, dtype=np.float32)
        phrases = []
        for phrase_item in audio_item["phrases"]:
            phrase = phrase_item["phrase"]
            if len(phrase.split()) <= self.max_phrase_words:
                phrases.append(phrase)
        return {
            "audiocap_id": audiocap_id,
            "waveform": waveform,
            "phrases": phrases
        }

    def __len__(self):
        return len(self.data)


class SinglePhraseEvalDataset(Dataset):

    def __init__(self,
                 waveform: str,
                 label: str,
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
        audio_id = audio_item["audio_id"]
        audiocap_id = audio_item["audiocap_id"]
        waveform = read_from_h5(audio_id, self.aid_to_h5, self.cache)
        waveform = np.array(waveform, dtype=np.float32)
        phrase_item = audio_item["phrases"][phrase_idx]
        phrase = [phrase_item["phrase"],]
        return {
            "audiocap_id": audiocap_id,
            "waveform": waveform,
            "phrase": phrase,
            "start_index": phrase_item["start_index"],
            "end_index": phrase_item["end_index"],
        }

    def __len__(self):
        return len(self.idxs)


if __name__ == "__main__":
    import utils.train_util as train_util
    from tqdm import tqdm
    config = train_util.parse_config_or_kwargs(
        "configs/weakly_supervised/audiocaps/phrase_level/" \
        "free_text_tagging/cnn8rnn_w2vmean_similarity.yaml")
    
    cfg = config["data"]["train"].copy()
    dataset = train_util.init_obj_from_str(cfg["dataset"])
    collate_fn = train_util.init_obj_from_str(cfg["collate_fn"])
    kwargs = {
        "collate_fn": collate_fn,
        "shuffle": True,
        "num_workers": 4
    }
    dataloader = torch.utils.data.DataLoader(dataset, **kwargs)
    
    for batch in tqdm(dataloader):
        pass
