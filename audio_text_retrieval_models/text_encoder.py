from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class Bert(nn.Module):
    def __init__(self, model_type):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_type)
        self.embed_dim = self.model.config.hidden_size

    def forward(self, **tokens):
        output = self.model(**tokens)
        # [CLS] pooling
        clip_emb = output.last_hidden_state[:, 0, :]
        time_emb = output.last_hidden_state
        output_dict = {
            "clip_emb": clip_emb,
            "time_emb": time_emb,
            # "time_mask": tokens["attention_mask"]
        }
        return output_dict


if __name__ == '__main__':
    input_tdim = 1000
    model = Bert(model_type="prajjwal1/bert-medium")
    print("Bert medium")
    # print(model)
    num_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f"Trainable parameters: {num_params}")
    test_input = [
        ["dog panting and panting",
         "an adult male is speaking and a motor vehicle engine is started up",],
        ["a bird whistling followed by a man speaking",
         "footsteps on gravel followed by a horse walking on gravel and birds chirping in the background"]
    ]
    print(sum(test_input, []))
    test_output = model(test_input)
    print("clip: ", test_output["clip_emb"].shape)
    print("time: ", test_output["time_emb"].shape)
    # print("time_mask: ", test_output["time_mask"].shape)
