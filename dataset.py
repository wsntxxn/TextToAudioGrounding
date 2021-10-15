from typing import List, Dict
import numpy as np
import torch
import h5py
from utils.build_vocab import Vocabulary


class GroundingEvalDataset(torch.utils.data.Dataset):
    """
    Returns in a sample:
    1. audio feature
    2. text query
    3. other infomation
    """
    def __init__(self,
                 audio_to_h5: Dict,
                 label: List,
                 vocabulary: Vocabulary):
        self.audio_to_h5 = audio_to_h5
        self.label = label
        self.audio_cache = {}
        self.vocabulary = vocabulary

    def __getitem__(self, index):
        item = self.label[index]
        audio_id = item["audio_id"]
        audio_h5 = self.audio_to_h5[audio_id]
        if not audio_h5 in self.audio_cache:
            self.audio_cache[audio_h5] = h5py.File(audio_h5, "r")
        audio_feature = self.audio_cache[audio_h5][audio_id][()]
        text = [self.vocabulary(token) for token in item["phrase"].split()]
        info = {
            "audiocap_id": item["audiocap_id"],
            "start_word": item["start_word"]
        }
        audio_feature = torch.as_tensor(audio_feature)
        text = torch.as_tensor(text)
        return audio_feature, text, info

    def __len__(self):
        return len(self.label)


class GroundingDataset(GroundingEvalDataset):

    def __init__(self,
                 audio_to_h5: Dict,
                 label: List,
                 vocabulary: Vocabulary,
                 time_resolution: float = 0.02):
        super().__init__(audio_to_h5, label, vocabulary)
        self.time_resolution = time_resolution

    def __getitem__(self, index):
        audio_feature, text, _ = super().__getitem__(index)
        item = self.label[index]
        label = np.zeros(audio_feature.shape[0], dtype=int)
        for start, end in item["timestamps"]:
            start = np.ceil(start / self.time_resolution).astype(int)
            end = np.ceil(end / self.time_resolution).astype(int)
            label[start: end] = 1
        label = torch.as_tensor(label)
        return audio_feature, text, label


def collate_fn(length_idxs):

    def collate_wrapper(data_batches):
        # data_batches.sort(key=lambda x: len(x[1]), reverse=True)

        def merge_seq(dataseq, dim=0):
            lengths = [seq.shape for seq in dataseq]
            # Assuming duration is given in the first dimension of each sequence
            maxlengths = tuple(np.max(lengths, axis=dim))
            # For the case that the lengths are 2 dimensional
            lengths = np.array(lengths)[:, dim]
            padded = torch.zeros((len(dataseq),) + maxlengths)
            for i, seq in enumerate(dataseq):
                end = lengths[i]
                padded[i, :end] = seq[:end]
            return padded, lengths
        
        data_out = []
        data_len = []
        for idx, data in enumerate(zip(*data_batches)):
            if isinstance(data[0], torch.Tensor):
                if len(data[0].shape) == 0:
                    data_seq = torch.as_tensor(data)
                else:
                    data_seq, tmp_len = merge_seq(data)
                    if idx in length_idxs:
                        # print(tmp_len)
                        data_len.append(tmp_len)
            else:
                data_seq = data
            data_out.append(data_seq)
        data_out.extend(data_len)

        return data_out

    return collate_wrapper

if __name__ == "__main__":
    import argparse
    import pickle
    import json
    from tqdm import tqdm
    
    from utils.train_util import load_dict_from_csv

    parser = argparse.ArgumentParser()
    parser.add_argument("-audio_feature", default="data/train/lms.csv", type=str)
    parser.add_argument("-label_file", default="data/train/label.json", type=str)
    parser.add_argument("-vocabulary", default="data/train/vocab.pkl", type=str)
    args = parser.parse_args()
    label = json.load(open(args.label_file))
    audio_to_h5 = load_dict_from_csv(args.audio_feature, ("audio_id", "hdf5_path"))
    vocabulary = pickle.load(open(args.vocabulary, "rb"))
    dataset = GroundingDataset(audio_to_h5, label, vocabulary)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        collate_fn=collate_fn([0,]),
        num_workers=4,
        shuffle=True)

    with tqdm(total=len(dataloader), ncols=100, ascii=True, unit="batch") as pbar:
        for batch in dataloader:
            audio = batch[0]
            text = batch[1]
            label = batch[2]
            pbar.set_postfix(audio=audio.shape,
                             text=text.shape,
                             label=label.shape)
            pbar.update()
    print("audio: ", audio.shape, "text: ", text.shape, "label: ", label.shape)
