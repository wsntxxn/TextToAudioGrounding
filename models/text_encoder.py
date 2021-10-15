from typing import Dict
import torch
import torch.nn as nn

from models.utils import init_weights, mean_with_lens

class EmbeddingMeanEncoder(nn.Module):

    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.apply(init_weights)

    def load_pretrained_embedding(self, weight: torch.Tensor, freeze: bool = True):
        assert weight.shape == self.embedding.weight.shape, \
            f"expect embedding with shape {self.embedding.weight.shape} " \
            f"but {weight.shape} is given"
        self.embedding = nn.Embedding.from_pretrained(weight, freeze)

    def forward(self, input_dict: Dict):
        tokens = input_dict["text"]
        lens = input_dict["text_len"]
        lens = torch.as_tensor(lens)
        tokens = tokens.long()
        embeds = self.embedding(tokens)
        embeds = mean_with_lens(embeds, lens)
        return embeds
