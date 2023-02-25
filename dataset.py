import math
import pickle
import json
import random
from typing import List, Dict, Tuple
import numpy as np
import torch
import h5py
from torch.utils.data import Dataset
from sklearn.metrics.pairwise import cosine_similarity

from utils.build_vocab import Vocabulary
from utils.train_util import load_dict_from_csv


def read_from_h5(key: str, key_to_h5: Dict, cache: Dict):
    hdf5_path = key_to_h5[key]
    if hdf5_path not in cache:
        cache[hdf5_path] = h5py.File(hdf5_path, "r")
    return cache[hdf5_path][key][()]


class AudioTaggingRawTextDataset(Dataset):

    def __init__(self,
                 waveform: str,
                 label: str,
                 audioset_label: str,
                 phrase_embed: str,
                 as_label_embed: str,
                 label_encoder: str,
                 thresholds: List = [0.5, 1.0],
                 min_sim_percent: float = None,
                 use_audioset_label: bool = True,
                 topk: int = 1,
                 max_phrase_words: int = 10,
                 max_audio_length: float = None,
                 sample_rate: int = 32000):
        self.aid_to_h5 = load_dict_from_csv(waveform, ("audio_id", "hdf5_path"))
        self.cache = {}
        self.data = json.load(open(label))
        self.thresholds = thresholds
        self.min_sim_percent = min_sim_percent
        self.topk = topk
        self.max_phrase_words = max_phrase_words
        self.label_encoder = pickle.load(open(label_encoder, "rb"))
        self.classes_num = len(self.label_encoder.classes_)
        self.label_to_idx = { lbl: idx for idx, lbl in enumerate(
            self.label_encoder.classes_) }
        self.phrase_to_emb = pickle.load(open(phrase_embed, "rb"))
        self.label_to_emb = pickle.load(open(as_label_embed, "rb"))
        self.label_embs = []
        for lbl, emb in self.label_to_emb.items():
            self.label_embs.append(emb)
        self.label_embs = np.stack(self.label_embs)
        self.aid_to_aslabel = load_dict_from_csv(audioset_label,
                                                ("audio_id", "event_labels"))
        self.use_audioset_label = use_audioset_label
        if max_audio_length is not None:
            self.max_audio_len = int(max_audio_length * sample_rate)
        else:
            self.max_audio_len = None

        msg = "either one of 'thresholds' or 'min_sim_percent' can be set"
        if min_sim_percent is not None:
            assert thresholds is None, msg
            assert topk == 1, "currently only support topk = 1 when setting similarity percent"
            self.calc_thresholds()
        if thresholds is not None:
            assert min_sim_percent is None, msg

    def calc_thresholds(self):
        phrase_embs = []
        for phrase, emb in self.phrase_to_emb.items():
            phrase_embs.append(emb)
        phrase_embs = np.stack(phrase_embs)
        sims = cosine_similarity(phrase_embs, self.label_embs).max(1)
        min_sim = np.percentile(sims, self.min_sim_percent)
        self.thresholds = [min_sim, 1.0]

    def assign_phrase_label(self, phrase_emb, label_onehot):
        sim = cosine_similarity(phrase_emb.reshape(1, -1), self.label_embs)[0]
        if sim.max() < self.thresholds[0] or sim.min() > self.thresholds[1]:
            return
        sim[np.where((sim < self.thresholds[0]) | 
                     (sim > self.thresholds[1]))[0]] = 0
        if self.topk > 0:
            indices = np.argsort(sim)[::-1][:self.topk]
        else:
            indices = np.where(sim)[0]
        label_onehot[indices] = 1

    def __getitem__(self, index):
        audio_item = self.data[index]
        audio_id = audio_item["audio_id"]
        caption = audio_item["tokens"]
        waveform = read_from_h5(audio_id, self.aid_to_h5, self.cache)
        waveform = np.array(waveform, dtype=np.float32)
        if self.max_audio_len is not None and \
                waveform.shape[0] > self.max_audio_len:
            start = random.randint(0, waveform.shape[0] - self.max_audio_len)
            waveform = waveform[start: start + self.max_audio_len]
        label = np.zeros(self.classes_num)
        for phrase_item in audio_item["phrases"]:
            phrase = phrase_item["phrase"]
            phrase_emb = self.phrase_to_emb[phrase]
            if len(phrase.split()) <= self.max_phrase_words:
                self.assign_phrase_label(phrase_emb, label)
        if self.use_audioset_label:
            for as_label in self.aid_to_aslabel[audio_id].split(";"):
                label[self.label_to_idx[as_label]] = 1
            
        return {
            "audiocap_id": audio_item["audiocap_id"],
            "audio_id": audio_item["audio_id"],
            "text": caption,
            "waveform": waveform,
            "label": label
        }

    def __len__(self):
        return len(self.data)


class AudioTaggingTestRawTextDataset(Dataset):


    def __init__(self,
                 waveform: str,
                 label: str,
                 phrase_embed: str,
                 as_label_embed: str,) -> None:
        self.aid_to_h5 = load_dict_from_csv(waveform, ("audio_id", "hdf5_path"))
        self.cache = {}
        self.data = json.load(open(label))
        self.phrase_to_emb = pickle.load(open(phrase_embed, "rb"))
        self.label_to_emb = pickle.load(open(as_label_embed, "rb"))
        self.label_embs = []
        for lbl, emb in self.label_to_emb.items():
            self.label_embs.append(emb)
        self.label_embs = np.stack(self.label_embs)
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
        phrase_item = audio_item["phrases"][phrase_idx]
        caption = audio_item["tokens"]
        waveform = read_from_h5(audio_id, self.aid_to_h5, self.cache)
        waveform = np.array(waveform, dtype=np.float32)
        text = phrase_item["phrase"]
        emb = self.phrase_to_emb[text]
        sim = cosine_similarity(emb.reshape(1, -1), self.label_embs)
        indice = sim[0].argmax()
        return {
            "audio_id": audio_id,
            "audiocap_id": audio_item["audiocap_id"],
            "start_index": phrase_item["start_index"],
            "end_index": phrase_item["end_index"],
            "waveform": waveform,
            "text": caption,
            "text_idx": indice
        }

    def __len__(self):
        return len(self.idxs)


class AudioTaggingTextSimDataset(Dataset):

    def __init__(self,
                 waveform: str,
                 label: str,
                 phrase_embed: str,
                 as_label_embed: str):
        super().__init__()
        self.aid_to_h5 = load_dict_from_csv(waveform, ("audio_id", "hdf5_path"))
        self.cache = {}
        self.data = json.load(open(label))
        self.idxs = []
        for audio_idx, audio_item in enumerate(self.data):
            for phrase_idx, phrase_item in enumerate(audio_item["phrases"]):
                self.idxs.append((audio_idx, phrase_idx))
        self.phrase_to_emb = pickle.load(open(phrase_embed, "rb"))
        self.label_to_emb = pickle.load(open(as_label_embed, "rb"))
        self.label_embs = []
        for lbl, emb in self.label_to_emb.items():
            self.label_embs.append(emb)
        self.label_embs = np.stack(self.label_embs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, index):
        audio_idx, phrase_idx = self.idxs[index]
        audio_item = self.data[audio_idx]
        audio_id = audio_item["audio_id"]
        phrase_item = audio_item["phrases"][phrase_idx]
        waveform = read_from_h5(audio_id, self.aid_to_h5, self.cache)
        waveform = np.array(waveform, dtype=np.float32)
        phrase_emb = self.phrase_to_emb[phrase_item["phrase"]]
        lbl_sim = cosine_similarity(phrase_emb.reshape(1, -1), self.label_embs)[0]
        return {
            "audiocap_id": audio_item["audiocap_id"],
            "start_index": phrase_item["start_index"],
            "end_index": phrase_item["end_index"],
            "waveform": waveform,
            "label_sim": lbl_sim
        }


class SedRawTextDataset(AudioTaggingRawTextDataset):

    def __init__(self,
                 waveform: str,
                 label: str,
                 audioset_label: str,
                 phrase_embed: str,
                 as_label_embed: str,
                 label_encoder: str,
                 time_resolution: float = 0.02,
                 sample_rate: int = 32000,
                 thresholds: List = [0.5, 1.0],
                 use_audioset_label: bool = True,
                 topk: int = 1,
                 max_phrase_words: int = 10):
        super().__init__(waveform,
                         label,
                         audioset_label,
                         phrase_embed,
                         as_label_embed,
                         label_encoder,
                         thresholds,
                         topk,
                         max_phrase_words)
        self.time_resolution = time_resolution
        self.sample_rate = sample_rate
        self.use_audioset_label = use_audioset_label

    def assign_phrase_label(self, phrase_item, weak_label, strong_label, strong_label_mask):
        phrase_emb = self.phrase_to_emb[phrase_item["phrase"]]
        sim = cosine_similarity(phrase_emb.reshape(1, -1), self.label_embs)[0]
        sim[np.where((sim < self.thresholds[0]) | 
                     (sim > self.thresholds[1]))[0]] = 0
        if self.topk > 0:
            indices = np.argsort(sim)[::-1][:self.topk]
        else:
            indices = np.where(sim)[0]
        weak_label[indices] = 1
        strong_label_mask[indices] = 1
        for start, end in phrase_item["segments"]:
            onset = round(start / self.time_resolution)
            offset = round(end / self.time_resolution)
            strong_label[onset: offset, indices] = 1

    def __getitem__(self, index):
        audio_item = self.data[index]
        audio_id = audio_item["audio_id"]
        caption = audio_item["tokens"]
        waveform = read_from_h5(audio_id, self.aid_to_h5, self.cache)
        waveform = np.array(waveform, dtype=np.float32)
        weak_label = np.zeros(self.classes_num)
        audio_duration = waveform.shape[0] / self.sample_rate
        n_frame = math.floor(audio_duration / self.time_resolution) + 1
        strong_label = np.zeros((n_frame, self.classes_num))
        strong_label_mask = np.zeros(self.classes_num)
        for phrase_item in audio_item["phrases"]:
            phrase = phrase_item["phrase"]
            if len(phrase.split()) <= self.max_phrase_words:
                self.assign_phrase_label(phrase_item, weak_label, strong_label, strong_label_mask)

        if self.use_audioset_label:
            for as_label in self.aid_to_aslabel[audio_id].split(";"):
                weak_label[self.label_to_idx[as_label]] = 1
        return {
            "audiocap_id": audio_item["audiocap_id"],
            "audio_id": audio_item["audio_id"],
            "text": caption,
            "waveform": waveform,
            "weak_label": weak_label,
            "strong_label": strong_label,
            "strong_label_mask": strong_label_mask
        }


class KmeansClusteredDataset(Dataset):

    def __init__(self,
                 waveform: str,
                 label: str,
                 phrase_embed: str,
                 cluster_model: str,
                 label_type: str = "weak",
                 max_dist_percent: float = 95.0,
                 time_resolution: float = 0.02,
                 sample_rate: int = 32000,
                 max_audio_length: float = None,
                 no_waveform: bool = False):
        import joblib
        self.aid_to_h5 = load_dict_from_csv(waveform, ("audio_id", "hdf5_path"))
        self.cache = {}
        self.data = json.load(open(label))
        self.label_type = label_type
        self.cluster_model = joblib.load(cluster_model)
        self.classes_num = self.cluster_model.n_clusters
        self.phrase_to_emb = pickle.load(open(phrase_embed, "rb"))
        self.max_dist_percent = max_dist_percent
        self.prepare_phrase_label()
        self.time_resolution = time_resolution
        self.sample_rate = sample_rate
        self.no_waveform = no_waveform
        if max_audio_length is not None:
            self.max_audio_len = int(max_audio_length * sample_rate)
        else:
            self.max_audio_len = None

    def prepare_phrase_label(self):
        phrases = []
        for item in self.data:
            for phrase_item in item["phrases"]:
                phrases.append(phrase_item["phrase"])
        phrases = list(set(phrases))
        embs = [self.phrase_to_emb[phrase] for phrase in phrases]
        embs = np.stack(embs)
        labels = self.cluster_model.predict(embs)
        distances = self.cluster_model.transform(embs).min(axis=1)
        self.max_distance = np.percentile(distances, self.max_dist_percent)
        self.phrase_to_label = {}
        self.phrase_to_distance = {}
        for i in range(len(phrases)):
            phrase = phrases[i]
            self.phrase_to_label[phrase] = labels[i]
            self.phrase_to_distance[phrase] = distances[i]

    def __getitem__(self, index):
        audio_item = self.data[index]
        audio_id = audio_item["audio_id"]
        caption = audio_item["tokens"]
        waveform = read_from_h5(audio_id, self.aid_to_h5, self.cache)
        waveform = np.array(waveform, dtype=np.float32)
        if self.max_audio_len is not None and \
                waveform.shape[0] > self.max_audio_len:
            start = random.randint(0, waveform.shape[0] - self.max_audio_len)
            waveform = waveform[start: start + self.max_audio_len]
        output = {
            "audiocap_id": audio_item["audiocap_id"],
            "audio_id": audio_item["audio_id"],
            "text": caption,
        }
        if not self.no_waveform:
            output["waveform"] = waveform
        if self.label_type == "weak":
            label = np.zeros(self.classes_num)
            for phrase_item in audio_item["phrases"]:
                phrase = phrase_item["phrase"]
                if self.phrase_to_distance[phrase] <= self.max_distance:
                    label[self.phrase_to_label[phrase]] = 1
            output["label"] = label
        elif self.label_type == "strong":
            audio_duration = waveform.shape[0] / self.sample_rate
            n_frame = math.floor(audio_duration / self.time_resolution) + 1
            weak_label = np.zeros(self.classes_num)
            strong_label = np.zeros((n_frame, self.classes_num))
            for phrase_item in audio_item["phrases"]:
                phrase = phrase_item["phrase"]
                if self.phrase_to_distance[phrase] <= self.max_distance:
                    label_idx = self.phrase_to_label[phrase]
                    weak_label[label_idx] = 1
                    for start, end in phrase_item["segments"]:
                        onset = round(start / self.time_resolution)
                        offset = round(end / self.time_resolution)
                        strong_label[onset: offset, label_idx] = 1
            output["weak_label"] = weak_label
            output["strong_label"] = strong_label

        return output

    def __len__(self):
        return len(self.data)


class KmeansClusteredTestDataset(Dataset):

    def __init__(self,
                 waveform: str,
                 label: str,
                 phrase_embed: str,
                 cluster_model: str):
        import joblib
        self.aid_to_h5 = load_dict_from_csv(waveform, ("audio_id", "hdf5_path"))
        self.cache = {}
        self.data = json.load(open(label))
        self.cluster_model = joblib.load(cluster_model)
        self.classes_num = self.cluster_model.n_clusters
        self.phrase_to_emb = pickle.load(open(phrase_embed, "rb"))
        self.prepare_phrase_label()
        self.generate_index()

    def generate_index(self):
        self.idxs = []
        for audio_idx, audio_item in enumerate(self.data):
            for phrase_idx, phrase_item in enumerate(audio_item["phrases"]):
                self.idxs.append((audio_idx, phrase_idx))

    def prepare_phrase_label(self):
        phrases = []
        for item in self.data:
            for phrase_item in item["phrases"]:
                phrases.append(phrase_item["phrase"])
        phrases = list(set(phrases))
        embs = [self.phrase_to_emb[phrase] for phrase in phrases]
        embs = np.stack(embs)
        labels = self.cluster_model.predict(embs)
        self.phrase_to_label = {}
        for phrase, label in zip(phrases, labels):
            self.phrase_to_label[phrase] = label

    def __getitem__(self, index):
        audio_idx, phrase_idx = self.idxs[index]
        audio_item = self.data[audio_idx]
        audio_id = audio_item["audio_id"]
        caption = audio_item["tokens"]
        waveform = read_from_h5(audio_id, self.aid_to_h5, self.cache)
        waveform = np.array(waveform, dtype=np.float32)
        phrase_item = audio_item["phrases"][phrase_idx]
        output = {
            "audiocap_id": audio_item["audiocap_id"],
            "audio_id": audio_item["audio_id"],
            "text": caption,
            "waveform": waveform,
            "text_idx": self.phrase_to_label[phrase_item["phrase"]],
            "start_index": phrase_item["start_index"],
            "end_index": phrase_item["end_index"],
        }

        return output

    def __len__(self):
        return len(self.idxs)


if __name__ == "__main__":
    import argparse
    from tqdm import tqdm
    from scipy import sparse
    from utils.train_util import load_config, init_obj_from_str

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    config_file = args.config
    config = load_config(config_file)["data"]["train"]
    dataset = init_obj_from_str(config["dataset"])
    collate_fn = init_obj_from_str(config["collate_fn"])
    kwargs = {
        "collate_fn": collate_fn,
        "shuffle": False
    }
    kwargs.update(config["dataloader_args"])
    dataloader = torch.utils.data.DataLoader(
        dataset, **kwargs)

    torch.multiprocessing.set_sharing_strategy('file_system')
    labels = []
    with tqdm(total=len(dataloader), ncols=100, ascii=True, unit="batch") as pbar:
        for batch in dataloader:
            labels.append(batch["label"].numpy())
            pbar.update()
    
    labels = np.concatenate(labels)
    sparse.save_npz(args.output, sparse.csr_matrix(labels))
