import pickle
import json
from typing import List, Dict, Tuple
import random

import numpy as np
import torch
import h5py
from torch.utils.data import Dataset
from sklearn.metrics.pairwise import cosine_similarity

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
                 **kwargs):
        self.aid_to_h5 = load_dict_from_csv(waveform, ("audio_id", "hdf5_path"))
        self.cache = {}
        # self.vocabulary = pickle.load(open(vocabulary, "rb"))
        self.data = json.load(open(label))
        self.phrase_num = phrase_num
        assert neg_samp_stratg in ("random", "clustering", "similarity")
        self.phrases = []
        for audio_item in self.data:
            for phrase_item in audio_item["phrases"]:
                self.phrases.append(phrase_item["phrase"])
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
            cand_phrase_embs = self.phrase_embs[cand_phrase_idxs]
            sims = cosine_similarity(pos_embs, cand_phrase_embs)
            sims = sims.max(axis=0)
            try:
                neg_idxs = np.random.choice(
                    np.where(sims < self.sim_threshold)[0],
                    size=neg_phrase_num,
                    replace=False)
            except ValueError as e:
                print(f"No enough negative phrases for {pos_phrases}, try smaller phrase number")
                raise Exception(e)
            neg_phrases = cand_phrases[neg_idxs]
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
                    neg_phrases.append(np.random.choice(
                        self.cluster_idx_to_phrases[neg_cluster_idx]))
            else:
                # fewer cluster centers than requested phrase number, some clusters have to be sampled multiple times
                cluster_samp_num = np.zeros_like(cand_cluster_idxs)
                cur_neg_phrase_num = neg_phrase_num
                while cur_neg_phrase_num > len(cand_cluster_idxs):
                    cluster_samp_num += 1
                    cur_neg_phrase_num -= len(cand_cluster_idxs)
                cluster_samp_num[np.random.choice(
                    np.arange(len(cand_cluster_idxs)),
                    size=cur_neg_phrase_num,
                    replace=False)] = 1
                for idx, samp_num in enumerate(cluster_samp_num):
                    neg_cluster_idx = cand_cluster_idxs[idx]
                    samp_phrase = np.random.choice(
                        self.cluster_idx_to_phrases[neg_cluster_idx],
                        size=samp_num,
                        replace=False)
                    neg_phrases.extend(samp_phrase.tolist())

        if self.fix_neg:
            self.acid_to_neg[audiocap_id] = [self.phrase_to_idx[phrase] for phrase in neg_phrases]
        
        return neg_phrases
            

    def __getitem__(self, index):
        audio_item = self.data[index]
        audio_id = audio_item["audio_id"]
        audiocap_id = audio_item["audiocap_id"]
        waveform = read_from_h5(audio_id, self.aid_to_h5, self.cache)
        waveform = np.array(waveform, dtype=np.float32)
        pos_phrases = [phrase_item["phrase"] for phrase_item in 
            audio_item["phrases"]]
        
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
