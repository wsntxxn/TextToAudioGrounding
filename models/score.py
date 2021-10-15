import torch
import torch.nn as nn
import torch.nn.functional as F


class ExponentialNegativeL2(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, audio: torch.Tensor, text: torch.Tensor):
        # audio: [N, T, E], text: [N, E]
        audio = F.normalize(audio, dim=-1)
        text = F.normalize(text, dim=-1)
        diff = audio - text.unsqueeze(1) # diff: [N, T, E]
        score = torch.exp(-torch.norm(diff, dim=-1)) # dis: [N, T]
        return score

