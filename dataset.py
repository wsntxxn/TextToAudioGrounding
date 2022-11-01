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


class EvalDataset(Dataset):

    def __init__(self,
                 waveform,
                 label,
                 vocabulary):
        self.aid_to_h5 = load_dict_from_csv(waveform, ("audio_id", "hdf5_path"))
        self.cache = {}
        self.vocabulary = pickle.load(open(vocabulary, "rb"))
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
        phrase_item = audio_item["phrases"][phrase_idx]
        waveform = read_from_h5(audio_id, self.aid_to_h5, self.cache)
        waveform = np.array(waveform, dtype=np.float32)
        text = [self.vocabulary(token) for token in phrase_item["phrase"].split()]
        output = {
            "audiocap_id": audio_item["audiocap_id"],
            "start_index": phrase_item["start_index"],
            "end_index": phrase_item["end_index"],
            "waveform": waveform,
            "text": torch.as_tensor(text)
        }
        return output

    def __len__(self):
        return len(self.idxs)


class EvalSentenceDataset(EvalDataset):

    def __getitem__(self, index):
        output = super().__getitem__(index)
        caption = self.data[self.idxs[index][0]]["tokens"]
        text = [self.vocabulary(token) for token in caption.split()]
        output["text"] = torch.as_tensor(text)
        return output


class TrainDataset(EvalDataset):

    def __init__(self,
                 waveform: str,
                 label: str,
                 vocabulary: str,
                 time_resolution: float = 0.02,
                 sample_rate: int = 32000):
        super().__init__(waveform, label, vocabulary)
        self.time_resolution = time_resolution
        self.sample_rate = sample_rate

    def __getitem__(self, index):
        output = super().__getitem__(index)
        audio_idx, phrase_idx = self.idxs[index]
        audio_item = self.data[audio_idx]
        phrase_item = audio_item["phrases"][phrase_idx]
        audio_duration = output["waveform"].shape[0] / self.sample_rate
        n_frame = math.floor(audio_duration / self.time_resolution) + 1
        label = np.zeros(n_frame, dtype=int)
        for start, end in phrase_item["segments"]:
            onset = round(start / self.time_resolution)
            offset = round(end / self.time_resolution)
            label[onset: offset] = 1
        output["label"] = label
        return output


class AudioSentenceDataset(Dataset):

    def __init__(self,
                 waveform: str,
                 label: str,
                 vocabulary: str):
        self.aid_to_h5 = load_dict_from_csv(waveform, ("audio_id", "hdf5_path"))
        self.cache = {}
        self.vocabulary = pickle.load(open(vocabulary, "rb"))
        self.data = json.load(open(label))

    def __getitem__(self, index):
        item = self.data[index]
        waveform = read_from_h5(item["audio_id"], self.aid_to_h5, self.cache)
        waveform = np.array(waveform, dtype=np.float32)
        text = [self.vocabulary(token) for token in item["tokens"].split()]
        return {
            "audiocap_id": item["audiocap_id"],
            "waveform": waveform,
            "text": torch.as_tensor(text)
        }

    def __len__(self):
        return len(self.data)


class AudioPhraseWeakDataset(Dataset):

    def __init__(self,
                 waveform: str,
                 label: str,
                 vocabulary: str,
                 negative_num: int):
        self.aid_to_h5 = load_dict_from_csv(waveform, ("audio_id", "hdf5_path"))
        self.cache = {}
        self.vocabulary = pickle.load(open(vocabulary, "rb"))
        self.idxs = []
        self.data = json.load(open(label))
        self.negative_num = negative_num
        self.phrases = []
        for audio_idx, audio_item in enumerate(self.data):
            for phrase_idx, phrase_item in enumerate(audio_item["phrases"]):
                self.idxs.append((audio_idx, phrase_idx))
                self.phrases.append(phrase_item["phrase"])
        self.phrases = list(set(self.phrases))

    def __getitem__(self, index):
        audio_idx, phrase_idx = self.idxs[index]
        audio_item = self.data[audio_idx]
        audio_id = audio_item["audio_id"]
        phrase_item = audio_item["phrases"][phrase_idx]
        waveform = read_from_h5(audio_id, self.aid_to_h5, self.cache)
        waveform = np.array(waveform, dtype=np.float32)
        pos_text = phrase_item["phrase"]
        neg_texts = []
        for _ in range(self.negative_num):
            phrase = random.choice(self.phrases)
            while phrase == pos_text or phrase in neg_texts:
                phrase = random.choice(self.phrases)
            neg_texts.append(phrase)
        texts = [pos_text] + neg_texts
        tmp = []
        for text in texts:
            tmp.append(torch.as_tensor(
                [self.vocabulary(token) for token in text.split()]))
        texts, text_lens = pad_sequence(tmp)
        label = np.array([1] + [0] * self.negative_num)
        return {
            "audiocap_id": audio_item["audiocap_id"],
            "start_index": phrase_item["start_index"],
            "end_index": phrase_item["end_index"],
            "waveform": waveform,
            "text": torch.as_tensor(texts),
            "text_len": text_lens,
            "label": label
        }

    def __len__(self):
        return len(self.idxs)


class AudioSentencePhraseDataset(Dataset):

    def __init__(self,
                 waveform: str,
                 label: str,
                 vocabulary: str,
                 text_num: int,
                 max_phrase_words: int = 10):
        self.aid_to_h5 = load_dict_from_csv(waveform, ("audio_id", "hdf5_path"))
        self.cache = {}
        self.vocabulary = pickle.load(open(vocabulary, "rb"))
        self.data = json.load(open(label))
        self.text_num = text_num
        self.max_phrase_words = max_phrase_words
        self.phrases = []
        for audio_item in self.data:
            for phrase_item in audio_item["phrases"]:
                if len(phrase_item["phrase"].split()) <= self.max_phrase_words:
                    self.phrases.append(phrase_item["phrase"])
        self.phrases = list(set(self.phrases))

    def __getitem__(self, index):
        audio_item = self.data[index]
        audio_id = audio_item["audio_id"]
        waveform = read_from_h5(audio_id, self.aid_to_h5, self.cache)
        waveform = np.array(waveform, dtype=np.float32)
        pos_texts = []
        for phrase_item in audio_item["phrases"]:
            if len(phrase_item["phrase"].split()) <= self.max_phrase_words:
                pos_texts.append(phrase_item["phrase"])
        neg_texts = []
        if len(pos_texts) > self.text_num:
            pos_texts = pos_texts[:self.text_num]
        for _ in range(self.text_num - len(pos_texts)):
            phrase = random.choice(self.phrases)
            while phrase in pos_texts or phrase in neg_texts:
                phrase = random.choice(self.phrases)
            neg_texts.append(phrase)
        texts = pos_texts + neg_texts
        # print(texts)
        tmp = []
        for text in texts:
            tmp.append(torch.as_tensor(
                [self.vocabulary(token) for token in text.split()]))
        texts, text_lens = pad_sequence(tmp)
        label = np.array([1] * len(pos_texts) + [0] * len(neg_texts))
        return {
            "audiocap_id": audio_item["audiocap_id"],
            "waveform": waveform,
            "text": torch.as_tensor(texts),
            "text_len": text_lens,
            "label": label
        }

    def __len__(self):
        return len(self.data)


class AudioPhraseWeakRawTextDataset(Dataset):

    def __init__(self,
                 waveform: str,
                 label: str,
                 negative_num: int,
                 max_phrase_words: int = 10):
        self.aid_to_h5 = load_dict_from_csv(waveform, ("audio_id", "hdf5_path"))
        self.cache = {}
        self.idxs = []
        self.data = json.load(open(label))
        self.negative_num = negative_num
        self.max_phrase_words = max_phrase_words
        self.phrases = []
        for audio_idx, audio_item in enumerate(self.data):
            for phrase_idx, phrase_item in enumerate(audio_item["phrases"]):
                if len(phrase_item["phrase"].split()) <= self.max_phrase_words:
                    self.idxs.append((audio_idx, phrase_idx))
                    self.phrases.append(phrase_item["phrase"])
        self.phrases = list(set(self.phrases))

    def __getitem__(self, index):
        audio_idx, phrase_idx = self.idxs[index]
        audio_item = self.data[audio_idx]
        audio_id = audio_item["audio_id"]
        phrase_item = audio_item["phrases"][phrase_idx]
        waveform = read_from_h5(audio_id, self.aid_to_h5, self.cache)
        waveform = np.array(waveform, dtype=np.float32)
        pos_text = phrase_item["phrase"]
        neg_texts = []
        for _ in range(self.negative_num):
            phrase = random.choice(self.phrases)
            while phrase == pos_text or phrase in neg_texts:
                phrase = random.choice(self.phrases)
            neg_texts.append(phrase)
        texts = [pos_text] + neg_texts
        label = np.array([1] + [0] * self.negative_num)
        return {
            "audiocap_id": audio_item["audiocap_id"],
            "start_index": phrase_item["start_index"],
            "end_index": phrase_item["end_index"],
            "waveform": waveform,
            "text": texts,
            "label": label
        }

    def __len__(self):
        return len(self.idxs)


class AudioPhraseRawTextDataset(Dataset):

    def __init__(self,
                 waveform: str,
                 label: str,
                 phrase_embed: str,
                 text_num: int,
                 max_phrase_words: int = 10,
                 phrase_threshold: float = 0.6):
        self.aid_to_h5 = load_dict_from_csv(waveform, ("audio_id", "hdf5_path"))
        self.cache = {}
        data = json.load(open(label))
        self.text_num = text_num
        self.max_phrase_words = max_phrase_words
        self.phrase_to_emb = pickle.load(open(phrase_embed, "rb"))
        self.phrases = []
        self.data = []
        for audio_item in data:
            flag = False
            for phrase_item in audio_item["phrases"]:
                if len(phrase_item["phrase"].split()) <= self.max_phrase_words:
                    self.phrases.append(phrase_item["phrase"])
                    flag = True
            if flag:
                self.data.append(audio_item)
        self.phrases = list(set(self.phrases))
        self.phrase_threshold = phrase_threshold

    def __getitem__(self, index):
        audio_item = self.data[index]
        audio_id = audio_item["audio_id"]
        waveform = read_from_h5(audio_id, self.aid_to_h5, self.cache)
        waveform = np.array(waveform, dtype=np.float32)
        pos_texts = []
        pos_text_embs = []
        for phrase_item in audio_item["phrases"]:
            phrase = phrase_item["phrase"]
            if len(phrase.split()) <= self.max_phrase_words:
                pos_texts.append(phrase)
                pos_text_embs.append(self.phrase_to_emb[phrase])
        pos_text_embs = np.stack(pos_text_embs)
        neg_texts = []
        if len(pos_texts) > self.text_num:
            pos_texts = pos_texts[:self.text_num]
        for _ in range(self.text_num - len(pos_texts)):
            phrase = random.choice(self.phrases)
            sim = cosine_similarity(self.phrase_to_emb[phrase].reshape(1, -1),
                                    pos_text_embs)
            while sim.max() >= self.phrase_threshold or phrase in neg_texts:
                phrase = random.choice(self.phrases)
                sim = cosine_similarity(self.phrase_to_emb[phrase].reshape(1, -1),
                                        pos_text_embs)
            neg_texts.append(phrase)
        texts = pos_texts + neg_texts
        # print(texts)
        label = np.array([1] * len(pos_texts) + [0] * len(neg_texts))
        return {
            "audiocap_id": audio_item["audiocap_id"],
            "waveform": waveform,
            "text": texts,
            "label": label
        }

    def __len__(self):
        return len(self.data)


class AudioPhraseTaggingRawTextDataset(Dataset):

    def __init__(self,
                 waveform: str,
                 label: str,
                 phrase_sim: str,
                 max_phrase_words: int = 10,
                 min_sim: float = 0.5):
        self.aid_to_h5 = load_dict_from_csv(waveform, ("audio_id", "hdf5_path"))
        self.cache = {}
        data = json.load(open(label))
        self.max_phrase_words = max_phrase_words
        self.data = []
        self.min_sim = min_sim
        self.phrase_to_sim = load_dict_from_csv(phrase_sim,
                                                ("phrase", "sim"))
        for audio_item in data:
            use_flag = False
            for phrase_item in audio_item["phrases"]:
                if len(phrase_item["phrase"].split()) <= self.max_phrase_words \
                    and self.phrase_to_sim[phrase_item["phrase"]] >= self.min_sim:
                    use_flag = True
                    break
            if use_flag:
                self.data.append(audio_item)


    def __getitem__(self, index):
        audio_item = self.data[index]
        audio_id = audio_item["audio_id"]
        waveform = read_from_h5(audio_id, self.aid_to_h5, self.cache)
        waveform = np.array(waveform, dtype=np.float32)
        texts = []
        for phrase_item in audio_item["phrases"]:
            phrase = phrase_item["phrase"]
            if len(phrase.split()) <= self.max_phrase_words \
                and self.phrase_to_sim[phrase] >= self.min_sim:
                texts.append(phrase)
        return {
            "audiocap_id": audio_item["audiocap_id"],
            "waveform": waveform,
            "text": texts
        }

    def __len__(self):
        return len(self.data)


class AudioTaggingRawTextDataset(Dataset):

    def __init__(self,
                 waveform: str,
                 label: str,
                 audioset_label: str,
                 # phrase_indice: str,
                 phrase_embed: str,
                 as_label_embed: str,
                 label_encoder: str,
                 thresholds: List = [0.5, 1.0],
                 use_audioset_label: bool = True,
                 topk: int = 1,
                 max_phrase_words: int = 10):
        self.aid_to_h5 = load_dict_from_csv(waveform, ("audio_id", "hdf5_path"))
        self.cache = {}
        self.data = json.load(open(label))
        self.thresholds = thresholds
        self.topk = topk
        self.max_phrase_words = max_phrase_words
        self.label_encoder = pickle.load(open(label_encoder, "rb"))
        self.classes_num = len(self.label_encoder.classes_)
        self.label_to_idx = { lbl: idx for idx, lbl in enumerate(
            self.label_encoder.classes_) }
        # self.min_sim = min_sim
        # self.phrase_to_idx = load_dict_from_csv(phrase_indice,
                                                # ("phrase", "index"))
        # self.phrase_to_sim = load_dict_from_csv(phrase_indice,
                                                # ("phrase", "sim"))
        self.phrase_to_emb = pickle.load(open(phrase_embed, "rb"))
        self.label_to_emb = pickle.load(open(as_label_embed, "rb"))
        self.label_embs = []
        for lbl, emb in self.label_to_emb.items():
            self.label_embs.append(emb)
        self.label_embs = np.stack(self.label_embs)
        self.aid_to_aslabel = load_dict_from_csv(audioset_label,
                                                ("audio_id", "event_labels"))
        self.use_audioset_label = use_audioset_label

    def assign_phrase_label(self, phrase_emb, label_onehot):
        sim = cosine_similarity(phrase_emb.reshape(1, -1), self.label_embs)[0]
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
                 phrase_indice: str) -> None:
        self.aid_to_h5 = load_dict_from_csv(waveform, ("audio_id", "hdf5_path"))
        self.cache = {}
        self.data = json.load(open(label))
        self.phrase_to_idx = load_dict_from_csv(phrase_indice,
                                                ("phrase", "index"))
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
        text_idx = self.phrase_to_idx[text]
        return {
            "audiocap_id": audio_item["audiocap_id"],
            "start_index": phrase_item["start_index"],
            "end_index": phrase_item["end_index"],
            "waveform": waveform,
            "text": caption,
            "text_idx": text_idx
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


def pad_sequence(data):
    if isinstance(data[0], np.ndarray):
        data = [torch.as_tensor(arr) for arr in data]
    padded_seq = torch.nn.utils.rnn.pad_sequence(data,
                                                 batch_first=True)
    length = [x.shape[0] for x in data]
    return padded_seq, length


class CollateFunction:

    def __init__(self, pad_keys=[], sort_key=None):
        self.pad_keys = pad_keys
        self.sort_key = sort_key

    def __call__(self, data_batch):
        if self.sort_key is not None:
            data_batch.sort(key=lambda x: len(x[self.sort_key]), reverse=True)
        
        output = {}
        for data in data_batch:
            for key in data:
                if key not in output:
                    output[key] = []
                output[key].append(data[key])

        for key in data_batch[0].keys():
            try:
                if key in self.pad_keys:
                    padded_seq, length = pad_sequence(output[key])
                    output[key] = padded_seq
                    output[f"{key}_len"] = np.array(length)
                else:
                    data = np.array(output[key])
                    if isinstance(output[key][0], np.ndarray):
                        output[key] = torch.as_tensor(data)
                    else:
                        output[key] = data
            except Exception:
                print(f"error occurred when collating {key}")
                import ipdb; ipdb.set_trace()

        return output


class MultiTextCollate(CollateFunction):

    def __call__(self, data_batch):
        if self.sort_key is not None:
            data_batch.sort(key=lambda x: len(x[self.sort_key]), reverse=True)
        
        output = {}
        for data in data_batch:
            for key in data:
                if key not in output:
                    output[key] = []
                if key == "text":
                    output[key].append(data[key].transpose(0, 1))
                else:
                    output[key].append(data[key])

        for key in data_batch[0].keys():
            try:
                if key in self.pad_keys:
                    if key == "text":
                        padded_seq, length = pad_sequence(output[key])
                        padded_seq = padded_seq.transpose(1, 2)
                        output[key] = padded_seq
                    else:
                        padded_seq, length = pad_sequence(output[key])
                        output[key] = padded_seq
                        output[f"{key}_len"] = np.array(length)
                else:
                    data = np.array(output[key])
                    if isinstance(output[key][0], np.ndarray):
                        output[key] = torch.as_tensor(data)
                    else:
                        output[key] = data
            except Exception:
                print(f"error occurred when collating {key}")
                import ipdb; ipdb.set_trace()

        return output
        

class MultiRawTextCollate(CollateFunction):

    def __call__(self, data_batch):
        if self.sort_key is not None:
            data_batch.sort(key=lambda x: len(x[self.sort_key]), reverse=True)
        
        output = {}
        for data in data_batch:
            for key in data:
                if key not in output:
                    output[key] = []
                output[key].append(data[key])

        for key in data_batch[0].keys():
            try:
                if key in self.pad_keys:
                    padded_seq, length = pad_sequence(output[key])
                    output[key] = padded_seq
                    output[f"{key}_len"] = np.array(length)
                else:
                    data = output[key]
                    if key != "text":
                        data = np.array(data)
                    if isinstance(output[key][0], np.ndarray):
                        output[key] = torch.as_tensor(data)
                    else:
                        output[key] = data
            except Exception:
                print(f"error occurred when collating {key}")
                import ipdb; ipdb.set_trace()

        return output


class TaggingRawTextCollate(CollateFunction):

    def __call__(self, data_batch):
        if self.sort_key is not None:
            data_batch.sort(key=lambda x: len(x[self.sort_key]), reverse=True)
        
        output = {}
        for data in data_batch:
            for key in data:
                if key not in output:
                    output[key] = []
                output[key].append(data[key])

        for key in data_batch[0].keys():
            try:
                if key in self.pad_keys:
                    padded_seq, length = pad_sequence(output[key])
                    output[key] = padded_seq
                    output[f"{key}_len"] = np.array(length)
                else:
                    data = output[key]
                    if key != "text":
                        data = np.array(data)
                    else:
                        output["text_num"] = [len(d) for d in data]
                    if isinstance(output[key][0], np.ndarray):
                        output[key] = torch.as_tensor(data)
                    else:
                        output[key] = data
            except Exception:
                print(f"error occurred when collating {key}")
                import ipdb; ipdb.set_trace()

        return output


if __name__ == "__main__":
    import argparse
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--waveform", default="data/train/waveform.csv", type=str)
    parser.add_argument("--label", default="data/train/label.json", type=str)
    parser.add_argument("--vocabulary", default="data/train/vocab.pkl", type=str)
    args = parser.parse_args()

    dataset = TrainDataset(args.waveform, args.label, args.vocabulary)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8,
        collate_fn=CollateFunction(["waveform", "text", "label"]),
        num_workers=4,
        shuffle=True)

    with tqdm(total=len(dataloader), ncols=100, ascii=True, unit="batch") as pbar:
        for batch in dataloader:
            import ipdb; ipdb.set_trace()
            pbar.update()
