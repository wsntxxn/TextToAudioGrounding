import math
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import generate_length_mask


class ExpNegL2(nn.Module):

    def __init__(self, l2norm=True, text_level="seq") -> None:
        super().__init__()
        self.l2norm = l2norm
        self.text_level = text_level
        

    def forward(self, input_dict):
        # audio: [bs, n_seg, dim], text: [bs, dim]
        audio = input_dict["audio_emb"]
        text = input_dict["text_emb"]

        if self.text_level == "seq":
            text = text["seq_emb"] # [bs, dim]
        elif self.text_level == "token":
            text = text["token_emb"] # [bs, n_seg, dim]

        if self.l2norm:
            audio = F.normalize(audio, dim=-1)
            text = F.normalize(text, dim=-1)
        if text.ndim == 2:
            text = text.unsqueeze(1)
        diff = audio - text # diff: [bs, n_seg, dim]
        score = torch.exp(-torch.norm(diff, dim=-1)) # dis: [bs, n_seg]
        return score


class DotProduct(nn.Module):


    def __init__(self, l2norm=False, scaled=False, text_level="seq") -> None:
        super().__init__()
        self.l2norm = l2norm
        self.scaled = scaled
        self.text_level = text_level
        

    def forward(self, input_dict):
        audio = input_dict["audio_emb"] # [bs, n_seg, dim]
        text = input_dict["text_emb"]
        if self.text_level == "seq": # [bs, dim]
            text = text["seq_emb"]
        elif self.text_level == "token":
            text = text["token_emb"] # [bs, n_seg, dim]

        if self.l2norm:
            audio = F.normalize(audio, dim=-1)
            text = F.normalize(text, dim=-1)
        if text.ndim == 2:
            text = text.unsqueeze(1)
        score = (audio * text).sum(-1)
        if self.scaled:
            score = score / math.sqrt(audio.size(-1))
        score = torch.sigmoid(score).clamp(1e-7, 1.0)
        return score


class CrossAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout, kvdim=None) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout,
                                          batch_first=True, kdim=kvdim,
                                          vdim=kvdim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.linear = nn.Linear(embed_dim, 1)

    def forward(self, input_dict):
        audio = input_dict["audio_emb"] # [bs, n_seg, dim]
        text = input_dict["text_emb"]["token_emb"]
        text_len = input_dict["text_len"]
        padding_mask = ~generate_length_mask(text_len).to(audio.device)
        out, attn = self.attn(audio, text, text, key_padding_mask=padding_mask)
        # out: [batch_size, seq_length, embed_dim]
        out = audio + self.dropout(out)
        out = self.norm(out)
        out = self.linear(out)
        return torch.sigmoid(out).squeeze(-1)


if __name__ == "__main__":
    pass
