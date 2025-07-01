from pathlib import Path
from typing import Callable

import torch

from utils.train_util import merge_matched_keys, load_pretrained_base


class LoadPretrainedMixin:
    def process_state_dict(
        self,
        model_dict: 'dict[str, torch.Tensor]',
        pretrained_dict: 'dict[str, torch.Tensor]',
        output_fn: Callable,
        model_name: str,
    ):
        """
        Custom processing functions of each model that transforms `state_dict` loaded from 
        checkpoints to the state that can be used in `load_state_dict`.
        Use `merge_mathced_keys` to update parameters with matched names and shapes by 
        default.  

        Args
            model_dict:
                The state dict of the current model, which is going to load pretrained parameters
            pretrained_dict:
                A dictionary of parameters from a pre-trained model.
            output_fn:
                A function that output logging messages.

            Returns:
                dict[str, torch.Tensor]:
                    The updated state dict, where parameters with matched keys and shape are 
                    updated with values in `state_dict`.      
        """
        state_dict = merge_matched_keys(
            model_dict, pretrained_dict, output_fn, model_name
        )
        return state_dict

    def load_pretrained(self, ckpt_path: 'str | Path', output_fn: Callable):
        load_pretrained_base(
            self,
            ckpt_path,
            state_dict_process_fn=self.process_state_dict,
            output_fn=output_fn
        )
