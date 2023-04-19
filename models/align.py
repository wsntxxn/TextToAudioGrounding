import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DotProduct(nn.Module):

    def __init__(self, l2norm=False, scaled=False) -> None:
        super().__init__()
        self.l2norm = l2norm
        self.scaled = scaled

    def forward(self, audio: torch.Tensor, text: torch.Tensor, **kwargs):
        if self.l2norm:
            audio = F.normalize(audio, dim=-1)
            text = F.normalize(text, dim=-1)
        a_bs, n_seg, a_dim = audio.size()
        t_bs, n_txt, t_dim = text.size()
        assert a_bs == t_bs
        assert a_dim == t_dim
        batch_size = a_bs
        dim = a_dim
        score = torch.matmul(audio.reshape(-1, dim),
                             text.reshape(-1, dim).transpose(0, 1))
        if self.scaled:
            score = score / math.sqrt(dim)
        score = torch.sigmoid(score).clamp(1e-7, 1.0)
        score = score.reshape(batch_size, n_seg, batch_size, n_txt)
        score = score.transpose(1, 2)
        return score


class ExpNegL2(nn.Module):

    def forward(self, audio: torch.Tensor, text: torch.Tensor):
        a_bs, n_seg, a_dim = audio.size()
        t_bs, n_txt, t_dim = text.size()
        assert a_bs == t_bs
        assert a_dim == t_dim
        batch_size = a_bs
        dim = a_dim
        # audio = audio.unsqueeze(2).expand(
            # batch_size, n_seg, batch_size * n_txt, dim).reshape(-1, dim)
        # # [bs, n_seg, bs * n_txt, dim] -> [-1, dim]
        # audio = F.normalize(audio, dim=-1)
        # text = text.unsqueeze(2).expand(batch_size, n_txt, batch_size * n_seg, dim)
        # # [bs, n_txt, bs * n_seg, dim]
        # text = text.permute(2, 0, 1, 3).reshape(-1, dim)
        # # [bs * n_seg, bs, n_txt, dim] -> [-1, dim]
        # text = F.normalize(text, dim=-1)
        # diff = audio - text
        # score = torch.exp(-torch.norm(diff, dim=-1))
        # score = score.reshape(batch_size, n_seg, batch_size, n_txt)
        # score = score.transpose(1, 2)
        audio = F.normalize(audio, dim=-1)
        text = F.normalize(text, dim=-1)
        score = torch.zeros(batch_size, batch_size, n_seg, n_txt)
        for i in range(batch_size):
            for j in range(n_txt):
                diff = audio - text[i, j]
                s = torch.exp(-torch.norm(diff, dim=-1))
                score[:, i, :, j] = s
        return score
