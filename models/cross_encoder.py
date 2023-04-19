import torch
import torch.nn as nn


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


class CrossAttentionGating(nn.Module):

    def __init__(self, embed_dim):
        super().__init__()
        self.attn = Seq2SeqAttention(embed_dim, embed_dim, embed_dim)
        self.gating = CrossGating(embed_dim)

    def forward(self, input_dict):
        audio_emb = input_dict["audio_emb"]
        text_emb = input_dict["text_emb"]
        audio_len = input_dict["audio_len"]
        text_len = input_dict["text_len"]
        if isinstance(text_emb, dict):
            text_emb = text_emb["token_emb"]
        text_emb = self.attn(audio_emb, text_emb, audio_len, text_len)
        audio_emb, text_emb = self.gating(audio_emb, text_emb)
        return {
            "audio_emb": audio_emb,
            "text_emb": {"token_emb": text_emb}
        }
