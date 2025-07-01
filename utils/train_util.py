# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import sys
import importlib
import random
import os
import logging
from typing import Dict
from pathlib import Path
from typing import Callable

import yaml
import toml
import hydra
import h5py
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pprint import pformat
from torch.optim.swa_utils import AveragedModel as torch_average_model


def load_dict_from_csv(csv, cols):
    df = pd.read_csv(csv, sep="\t")
    output = dict(zip(df[cols[0]], df[cols[1]]))
    return output


def read_from_h5(key: str, key_to_h5: Dict, cache: Dict):
    hdf5_path = key_to_h5[key]
    if hdf5_path not in cache:
        cache[hdf5_path] = h5py.File(hdf5_path, "r")
    return cache[hdf5_path][key][()]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Mixup(object):
    def __init__(self, mixup_alpha, random_seed=1234):
        """Mixup coefficient generator.
        """
        self.mixup_alpha = mixup_alpha
        self.random_state = np.random.RandomState(random_seed)

    def get_lambda(self, batch_size):
        """Get mixup random coefficients.
        Args:
          batch_size: int
        Returns:
          mixup_lambdas: (batch_size,)
        """
        mixup_lambdas = []
        for n in range(0, batch_size, 2):
            lam = self.random_state.beta(
                self.mixup_alpha, self.mixup_alpha, 1
            )[0]
            mixup_lambdas.append(lam)
            mixup_lambdas.append(1. - lam)

        return np.array(mixup_lambdas)


def do_mixup(x, mixup_lambdas):
    """Mixup x of even indexes (0, 2, 4, ...) with x of odd indexes
    (1, 3, 5, ...).

    Args:
      x: (batch_size * 2, ...)
      mixup_lambda: (batch_size * 2,)

    Returns:
      out: (batch_size, ...)
    """
    mixup_lambdas = torch.as_tensor(mixup_lambdas,
                                    dtype=torch.float).to(x.device)
    out = (x[0 :: 2].transpose(0, -1) * mixup_lambdas[0 :: 2] + \
        x[1 :: 2].transpose(0, -1) * mixup_lambdas[1 :: 2]).transpose(0, -1)
    return out


def init_logger(filename, level="INFO"):
    filename = filename.__str__()
    formatter = logging.Formatter(
        "[ %(levelname)s : %(asctime)s ] - %(message)s"
    )
    logger = logging.getLogger(__name__ + "." + filename)
    logger.setLevel(getattr(logging, level))
    filehandler = logging.FileHandler(filename)
    filehandler.setFormatter(formatter)
    logger.addHandler(filehandler)
    return logger


def pprint_dict(in_dict, print_fn=sys.stdout.write, format='yaml'):
    """pprint_dict

    :param outputfun: function to use, defaults to sys.stdout
    :param in_dict: dict to print
    """
    if format == 'yaml':
        format_fn = yaml.dump
    elif format == 'pretty':
        format_fn = pformat
    else:
        raise Exception(f"format {format} not supported")
    for line in format_fn(in_dict).split('\n'):
        print_fn(line)


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit('.', 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def init_obj_from_str(config, **kwargs):
    obj_args = config["args"].copy()
    obj_args.update(kwargs)
    for k in config:
        if k not in ["type", "args"] and isinstance(config[k], dict) and \
            k not in kwargs:
            obj_args[k] = init_obj_from_str(config[k])
    cls = get_obj_from_str(config["type"])
    obj = cls(**obj_args)
    return obj


def copy_args_recursive(src_cfg, tgt_cfg, keys):
    for k in src_cfg:
        if k == "type":
            continue
        elif k == "args":
            for key in src_cfg["args"]:
                if key in keys:
                    tgt_cfg["args"][key] = src_cfg["args"][key]
        else:
            if k in tgt_cfg:
                copy_args_recursive(src_cfg[k], tgt_cfg[k], keys)


def merge_a_into_b(a, b):
    # merge dict a into dict b. values in a will overwrite b.
    for k, v in a.items():
        if isinstance(v, dict) and k in b:
            assert isinstance(b[k], dict
                             ), "Cannot inherit key '{}' from base!".format(k)
            merge_a_into_b(v, b[k])
        else:
            b[k] = v


def load_config(config_file):
    with open(config_file, "r") as reader:
        config = yaml.load(reader, Loader=yaml.FullLoader)
    if "inherit_from" in config:
        base_config_file = config["inherit_from"]
        base_config_file = os.path.join(
            os.path.dirname(config_file), base_config_file
        )
        assert not os.path.samefile(config_file, base_config_file), \
            "inherit from itself"
        base_config = load_config(base_config_file)
        del config["inherit_from"]
        merge_a_into_b(config, base_config)
        return base_config
    return config


def parse_config_or_kwargs(config_file, **kwargs):
    toml_list = []
    for k, v in kwargs.items():
        if isinstance(v, str):
            toml_list.append(f"{k}='{v}'")
        elif isinstance(v, bool):
            toml_list.append(f"{k}={str(v).lower()}")
        else:
            toml_list.append(f"{k}={v}")
    toml_str = "\n".join(toml_list)
    cmd_config = toml.loads(toml_str)
    yaml_config = load_config(config_file)
    merge_a_into_b(cmd_config, yaml_config)
    return yaml_config


def count_parameters(model):
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    return num_params


def pack_length(padded, lengths):
    packed = []
    for i in range(len(lengths)):
        packed.append(padded[i][:lengths[i], ...])
    return torch.cat(packed)


def pad_sequence(data):
    if isinstance(data[0], np.ndarray):
        data = [torch.as_tensor(arr) for arr in data]
    padded_seq = torch.nn.utils.rnn.pad_sequence(data, batch_first=True)
    length = torch.as_tensor([x.shape[0] for x in data]).long()
    return padded_seq, length


def merge_matched_keys(
    model_dict: 'dict[str, torch.Tensor]',
    pretrained_dict: 'dict[str, torch.Tensor]',
    output_fn: Callable = sys.stdout.write,
    model_name: str = "nn.Module",
) -> 'dict[str, torch.Tensor]':
    """
    Args:
    model_dict:
        The state dict of the current model, which is going to load pretrained parameters
    pretrained_dict:
        A dictionary of parameters from a pre-trained model.

    Returns:
        dict[str, torch.Tensor]:
            The updated state dict, where parameters with matched keys and shape are 
            updated with values in `state_dict`.
    """
    filtered_pretrained_dict = {}
    mismatch_keys = []
    for key, value in pretrained_dict.items():
        if key in model_dict and model_dict[key].shape == value.shape:
            filtered_pretrained_dict[key] = value
        else:
            mismatch_keys.append(key)
    output_fn(
        f"Loading pre-trained {model_name}, with mismatched keys {mismatch_keys}\n"
    )
    model_dict.update(filtered_pretrained_dict)
    return model_dict


def load_pretrained_base(
    model: nn.Module,
    ckpt_or_state_dict: 'str | Path | dict[str, torch.Tensor]',
    state_dict_process_fn: Callable = merge_matched_keys,
    output_fn: Callable = sys.stdout.write,
) -> None:
    pretrained_dict = ckpt_or_state_dict
    if not isinstance(ckpt_or_state_dict, dict):
        pretrained_dict = torch.load(
            ckpt_or_state_dict, map_location="cpu", weights_only=False
        )

    model_dict = model.state_dict()
    state_dict = state_dict_process_fn(
        model_dict, pretrained_dict, output_fn,
        type(model).__name__
    )
    model.load_state_dict(state_dict)


def load_pretrained_model(
    model, pretrained, output_fn=sys.stdout.write, **load_args
):
    if not isinstance(pretrained, dict) and not os.path.exists(pretrained):
        output_fn(f"pretrained {pretrained} not exist!")
        return

    if hasattr(model, "load_pretrained"):
        model.load_pretrained(pretrained, output_fn=output_fn, **load_args)
        return

    if isinstance(pretrained, dict):
        state_dict = pretrained
    else:
        state_dict = torch.load(pretrained, map_location="cpu")

    if "model" in state_dict:
        state_dict = state_dict["model"]
    model_dict = model.state_dict()
    pretrained_dict = {
        k: v
        for k, v in state_dict.items()
        if (k in model_dict) and (model_dict[k].shape == v.shape)
    }
    output_fn(f"Loading pretrained keys {pretrained_dict.keys()}")
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=True)


def instantiate_model(config: dict, print_fn: Callable):
    def add_print_fn(cfg: dict):
        if isinstance(cfg, dict):
            if "pretrained" in cfg:
                cfg["output_fn"] = print_fn
            for key, value in cfg.items():
                add_print_fn(value)
        elif isinstance(cfg, list):
            for item in cfg:
                add_print_fn(item)

    add_print_fn(config)
    model = hydra.utils.instantiate(config, _convert_="all")
    return model


def unsqueeze_at_1(x: "list | np.array | torch.Tensor"):
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(x, np.ndarray):
        x = np.expand_dims(x, axis=1)
    else:
        x = x.unsqueeze(1)
    return x


class MetricImprover:
    def __init__(self, mode):
        assert mode in ("min", "max")
        self.mode = mode
        # min: lower -> better; max: higher -> better
        self.best_value = np.inf if mode == "min" else -np.inf

    def compare(self, x, best_x):
        return x < best_x if self.mode == "min" else x > best_x

    def __call__(self, x):
        if self.compare(x, self.best_value):
            self.best_value = x
            return True
        return False

    def state_dict(self):
        return self.__dict__

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)


class AveragedModel(torch_average_model):
    def update_parameters(self, model):
        for p_swa, p_model in zip(self.parameters(), model.parameters()):
            device = p_swa.device
            p_model_ = p_model.detach().to(device)
            if self.n_averaged == 0:
                p_swa.detach().copy_(p_model_)
            else:
                p_swa.detach().copy_(
                    self.avg_fn(
                        p_swa.detach(), p_model_, self.n_averaged.to(device)
                    )
                )

        for b_swa, b_model in zip(list(self.buffers())[1:], model.buffers()):
            device = b_swa.device
            b_model_ = b_model.detach().to(device)
            if self.n_averaged == 0:
                b_swa.detach().copy_(b_model_)
            else:
                b_swa.detach().copy_(
                    self.avg_fn(
                        b_swa.detach(), b_model_, self.n_averaged.to(device)
                    )
                )
        self.n_averaged += 1
