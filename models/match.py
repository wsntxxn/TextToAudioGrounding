import math
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import generate_length_mask


class ExpNegL2(nn.Module):

    def __init__(self, l2norm=True) -> None:
        super().__init__()
        self.l2norm = l2norm

    def forward(self, audio: torch.Tensor, text: torch.Tensor, **kwargs):
        # audio: [N, T, E], text: [N, E]
        if self.l2norm:
            audio = F.normalize(audio, dim=-1)
            text = F.normalize(text, dim=-1)
        if text.ndim == 2:
            text = text.unsqueeze(1)
        diff = audio - text # diff: [N, T, E]
        score = torch.exp(-torch.norm(diff, dim=-1)) # dis: [N, T]
        return score


class ConcatFc(nn.Module):
    
    def __init__(self, embed_dim):
        super().__init__()
        self.fc = nn.Linear(embed_dim * 2, 1)

    def forward(self, audio: torch.Tensor, text: torch.Tensor, **kwargs):
        text = text.unsqueeze(1).repeat(1, audio.size(1), 1)
        x = torch.cat((audio, text), dim=-1)
        score = self.fc(x) # [N, T, 1]
        score = torch.sigmoid(score.squeeze(-1)).clamp(1e-7, 1.)
        return score


class DotProduct(nn.Module):

    def __init__(self, activation="sigmoid", l2norm=False) -> None:
        super().__init__()
        self.l2norm = l2norm
        assert activation in ("clamp", "sigmoid"), f"unsupported activation {activation}"
        self.activation = activation
        
    def forward(self, audio: torch.Tensor, text: torch.Tensor, **kwargs):
        if self.l2norm:
            audio = F.normalize(audio, dim=-1)
            text = F.normalize(text, dim=-1)
        if text.ndim == 2:
            text = text.unsqueeze(1)
        score = torch.bmm(audio, text.transpose(1, 2)).squeeze(2)
        if self.activation == "sigmoid":
            score = torch.sigmoid(score)
        return torch.clamp(score, 1e-7, 1.0)


class ScaledDotProduct(nn.Module):

    def forward(self, audio: torch.Tensor, text: torch.Tensor, **kwargs):
        d = audio.size(-1)
        text = text.unsqueeze(1)
        score = torch.bmm(audio, text.transpose(1, 2)).squeeze(2)
        score = torch.sigmoid(score / math.sqrt(d))
        return score


class Seq2SeqAttention(nn.Module):

    def __init__(self, d_q, d_kv, d_attn):
        super().__init__()
        self.h2attn = nn.Linear(d_q + d_kv, d_attn)
        self.v = nn.Parameter(torch.randn(d_attn))

    def forward(self, query, kv, query_len, kv_len):
        # query: [batch_size, max_q_len, d_q]
        # kv: [batch_size, max_kv_len, d_kv]
        batch_size, max_kv_len, d_kv = kv.size()
        batch_size, max_q_len, d_q = query.size()
        query_len = torch.as_tensor(query_len)
        kv_len = torch.as_tensor(kv_len)
        
        q_repeat = query.repeat(1, 1, max_kv_len).reshape(
            batch_size, max_kv_len * max_q_len, d_q)
        kv_repeat = kv.repeat(1, max_q_len, 1).reshape(
            batch_size, max_kv_len * max_q_len, d_kv)

        attn_input = torch.cat((q_repeat, kv_repeat), dim=-1)
        attn_out = torch.tanh(self.h2attn(attn_input)) # [bs, q_len * kv_len, d_attn]

        v = self.v.repeat(batch_size, 1).unsqueeze(1) # [bs, 1, d_attn]
        score = torch.bmm(v, attn_out.transpose(1, 2)).squeeze(1) # [bs, q_len * kv_len]
        score = score.reshape(batch_size, max_q_len, max_kv_len)

        mask1 = torch.arange(max_q_len).repeat(batch_size).view(
            batch_size, max_q_len) < query_len.view(-1, 1) # [bs, max_q_len]
        mask1 = mask1.unsqueeze(-1).repeat(1, 1, max_kv_len).to(score.device)
        score = score.masked_fill(mask1 == 0, -1e10)
        mask2 = torch.arange(max_kv_len).repeat(batch_size).view(
            batch_size, max_kv_len) < kv_len.view(-1, 1)
        mask2 = mask2.unsqueeze(1).repeat(1, max_q_len, 1).to(score.device)
        score = score.masked_fill(mask2 == 0, -1e10)
        attn = torch.softmax(score, dim=-1) # [bs, max_q_len, max_kv_len]
        out = torch.bmm(attn, kv) # [bs, max_q_len, d_kv]
        return out


class CrossGating(nn.Module):

    def __init__(self, d_model) -> None:
        super().__init__()
        self.fc_u = nn.Linear(d_model, d_model)
        self.fc_s = nn.Linear(d_model, d_model)

    def forward(self, u, s):
        g_u = torch.sigmoid(self.fc_u(u))
        s_out = s * g_u
        g_s = torch.sigmoid(self.fc_s(s))
        u_out = u * g_s
        return u_out, s_out


class AttnGatingExpL2(nn.Module):

    def __init__(self, embed_dim, l2norm=True):
        super().__init__()
        self.sim = ExpNegL2(l2norm=l2norm)
        self.attn = Seq2SeqAttention(embed_dim, embed_dim, embed_dim)
        self.gating = CrossGating(embed_dim)

    def forward(self,
                audio: torch.Tensor,
                text: torch.Tensor,
                audio_len: List,
                text_len: List):
        # audio: [batch_size, seq_length, embed_dim]
        # text: [batch_size, text_len, embed_dim]
        snippet = self.attn(audio, text, audio_len, text_len)
        audio, snippet = self.gating(audio, snippet)
        return self.sim(audio, snippet)


class AttnGatingDotProduct(nn.Module):

    def __init__(self, embed_dim):
        super().__init__()
        self.attn = Seq2SeqAttention(embed_dim, embed_dim, embed_dim)
        self.gating = CrossGating(embed_dim)

    def forward(self,
                audio: torch.Tensor,
                text: torch.Tensor,
                audio_len: List,
                text_len: List):
        # audio: [batch_size, seq_length, embed_dim]
        # text: [batch_size, text_len, embed_dim]
        snippet = self.attn(audio, text, audio_len, text_len)
        audio, snippet = self.gating(audio, snippet)
        sim = (audio * snippet).sum(-1)
        sim = torch.sigmoid(sim).clamp(1e-7, 1.0)
        return sim


class CrossAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout, kvdim=None) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout,
                                          batch_first=True, kdim=kvdim,
                                          vdim=kvdim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.linear = nn.Linear(embed_dim, 1)

    def forward(self,
                audio: torch.Tensor,
                text: torch.Tensor,
                audio_len: List,
                text_len: List):
        # audio: [batch_size, seq_length, embed_dim]
        # text: [batch_size, text_len, embed_dim]
        padding_mask = ~generate_length_mask(text_len).to(audio.device)
        out, attn = self.attn(audio, text, text, key_padding_mask=padding_mask)
        # out: [batch_size, seq_length, embed_dim]
        out = audio + self.dropout(out)
        out = self.norm(out)
        out = self.linear(out)
        return torch.sigmoid(out).squeeze(-1)


if __name__ == "__main__":
    attn = Seq2SeqAttention(4, 8, 16)
    q = torch.randn(3, 2, 4)
    kv = torch.randn(3, 5, 8)
    q_len = torch.tensor([2, 1, 1])
    kv_len = torch.tensor([5, 3, 4])
    with torch.no_grad():
        output = attn(q, kv, q_len, kv_len)
