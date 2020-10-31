from pathlib import Path

import numpy as np
import torch
import h5py

class QueryAudioDatasetEval(torch.utils.data.Dataset):
    """
    Returns in a sample:
    1. audio feature
    2. query [word indexs / raw soundtag word / pretrained embedding]
    3. other infomation
    """
    def __init__(self, 
                 audio_feature, 
                 label_df, 
                 transform=None,
                 query_form="embeds",
                 query_feature=None,
                 vocabulary=None): 
        self.audio_feature = audio_feature
        self.label_df = label_df
        self.transform = transform
        self.audio_store = None
        self.query_store = None

        assert query_form in ("wordids", "embeds")
        self.query_form = query_form
        if self.query_form == "embeds":
            assert query_feature is not None
            self.query_feature = query_feature
        else:
            assert vocabulary is not None
            self.vocabulary = vocabulary

    def __getitem__(self, index):
        if self.audio_store is None:
            self.audio_store = h5py.File(self.audio_feature, "r")
        if self.query_form == "embeds":
            self.query_store = h5py.File(self.query_feature, "r")
        info = self.label_df.iloc[index]
        # info: {"audiocap_id": xxx, "filename": xxx, "start_word": xxx}
        audio_feature = self.audio_store[Path(info["filename"]).name][()]
        if self.transform is not None:
            audio_feature = self.transform(audio_feature)

        info_return = {"audiocap_id": info["audiocap_id"], "start_word": info["start_word"]}

        if self.query_form == "embeds":
            query_feature = self.query_store[str(info["audiocap_id"]) + "/" + str(info["start_word"])][()]
            return torch.as_tensor(audio_feature), torch.as_tensor(query_feature), info_return
        else:
            query = [self.vocabulary(token) for token in info["soundtag"].split()]
            return torch.as_tensor(audio_feature), torch.as_tensor(query), info_return

    def __len__(self):
        return len(self.label_df)


class QueryAudioDataset(QueryAudioDatasetEval):

    def __init__(self, 
                 audio_feature,
                 label_df,
                 win_shift=0.02,
                 transform=None,
                 query_form="embeds",
                 query_feature=None,
                 vocabulary=None):
        super(QueryAudioDataset, self).__init__(audio_feature, label_df, transform, query_form, query_feature, vocabulary)
        self.win_shift = win_shift

    def __getitem__(self, index):
        audio_feature, query, _ = super(QueryAudioDataset, self).__getitem__(index)
        info = self.label_df.iloc[index]
        label = np.zeros(audio_feature.shape[0], dtype=int)
        for start, end in info["timestamps"]:
            start = np.ceil(start / self.win_shift).astype(int)
            end = np.ceil(end / self.win_shift).astype(int)
            label[start: end] = 1
        return audio_feature, query, torch.as_tensor(label)


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
            # if data_batches[0][2] == {'audiocap_id': 106203, 'start_word': 3}:
                # print(idx, data_out)
        data_out.extend(data_len)

        return data_out

    return collate_wrapper

if __name__ == "__main__":
    import argparse
    import pandas as pd
    from tqdm import tqdm
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_feature", default="data/logmel.hdf5", type=str)
    parser.add_argument("query_feature", default="data/query_embedding.hdf5", type=str)
    parser.add_argument("label_file", default="data/label_toy.json", type=str)
    args = parser.parse_args()
    label_df = pd.read_json(args.label_file)
    dataset = QueryAudioDataset(args.audio_feature, args.query_feature, label_df)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        collate_fn=collate_fn([0,]),
        num_workers=4,
        shuffle=True)

    with tqdm(total=len(dataloader), ncols=100, ascii=True, unit="batch") as pbar:
        for batch in dataloader:
            pbar.set_postfix(audio=batch[0].shape,
                             query=batch[1].shape,
                             label=batch[2].shape)
            pbar.update()
