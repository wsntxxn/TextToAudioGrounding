import numpy as np
import torch

from utils.train_util import pad_sequence


class VarLenPadCollate:

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


class TextCollate:

    def __init__(self, tokenizer, text_key="text", pad_keys=[], sort_key=None):
        self.tokenizer = tokenizer
        self.pad_keys = pad_keys
        self.sort_key = sort_key
        self.text_key = text_key

    def __call__(self, data_batch):
        if self.sort_key is not None:
            data_batch.sort(key=lambda x: len(x[self.sort_key]), reverse=True)
        
        output = {}
        for data_dict in data_batch:
            for key in data_dict:
                if key not in output:
                    output[key] = []
                output[key].append(data_dict[key])

        output["text_key"] = self.text_key

        for key in data_batch[0].keys():
            try:
                if key in self.pad_keys:
                    padded_seq, length = pad_sequence(output[key])
                    output[key] = padded_seq
                    output[f"{key}_len"] = np.array(length)
                elif key == self.text_key:
                    output.update(self.tokenizer(output[key]))
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


class VarNumTextCollate:

    def __init__(self, tokenizer, text_key="text", pad_keys=[], sort_key=None):
        self.tokenizer = tokenizer
        self.text_key = text_key
        self.pad_keys = pad_keys
        self.sort_key = sort_key

    def __call__(self, data_batch):
        if self.sort_key is not None:
            data_batch.sort(key=lambda x: len(x[self.sort_key]), reverse=True)

        output = {}
        for data_dict in data_batch:
            for key in data_dict:
                if key not in output:
                    output[key] = []
                output[key].append(data_dict[key])

        output["text_key"] = self.text_key

        for key in data_batch[0].keys():
            try:
                if key in self.pad_keys:
                    padded_seq, length = pad_sequence(output[key])
                    output[key] = padded_seq
                    output[f"{key}_len"] = np.array(length)
                elif key == self.text_key:
                    text_num = [len(x) for x in output[key]]
                    merged_text = sum(output[key], [])
                    output[f"{key}_num"] = text_num
                    tokens = self.tokenizer(merged_text)
                    output.update({
                        key: tokens["text"],
                        f"{key}_len": tokens["text_len"]
                    })
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
