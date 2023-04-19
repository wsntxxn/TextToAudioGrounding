import math
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpNegL2(nn.Module):

    def forward(self, audio: torch.Tensor, text: torch.Tensor, **kwargs):
        # audio: [bs, n_seg, dim], text: [bs, n_txt, dim]
        audio = F.normalize(audio, dim=-1)
        text = F.normalize(text, dim=-1)
        diff = audio.unsqueeze(1) - text.unsqueeze(2)
        # diff: [bs, n_txt, n_seg, dim]
        score = torch.exp(-torch.norm(diff, dim=-1))
        # score: [bs, n_txt, n_seg]
        return score


class DotProduct(nn.Module):

    def __init__(self, activation="sigmoid", l2norm=False) -> None:
        super().__init__()
        self.l2norm = l2norm
        self.activation = activation
        
    def forward(self, audio: torch.Tensor, text: torch.Tensor, **kwargs):
        # audio: [bs, n_seg, dim], text: [bs, n_txt, dim]
        if self.l2norm:
            audio = F.normalize(audio, dim=-1)
            text = F.normalize(text, dim=-1)
        score = torch.bmm(text, audio.transpose(1, 2))
        if self.activation == "sigmoid":
            score = torch.sigmoid(score)
        return torch.clamp(score, 1e-7, 1.0)


class ScaledDotProduct(nn.Module):

    def forward(self, audio: torch.Tensor, text: torch.Tensor, **kwargs):
        d = audio.size(-1)
        score = torch.bmm(text, audio.transpose(1, 2))
        score = torch.sigmoid(score / math.sqrt(d))
        return torch.clamp(score, 1e-7, 1.0)


if __name__ == "__main__":
    pass
