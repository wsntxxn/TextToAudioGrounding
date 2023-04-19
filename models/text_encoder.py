from typing import Dict
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

from models.utils import init_weights, mean_with_lens, generate_length_mask


class EmbeddingLayer(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 pretrained_embedding: str = None,
                 freeze_embedding: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.core = nn.Embedding(vocab_size, embed_dim)
        self.apply(init_weights)
        if pretrained_embedding is not None:
            self.load_pretrained_embedding(pretrained_embedding,
                                           freeze_embedding)

    def load_pretrained_embedding(self, weight: str, freeze: bool = True):
        weight = np.load(weight)
        assert weight.shape == self.core.weight.shape, \
            f"expect embedding with shape {self.core.weight.shape} " \
            f"but {weight.shape} is given"
        weight = torch.as_tensor(weight, dtype=torch.float)
        self.core = nn.Embedding.from_pretrained(weight, freeze)

    def forward(self, input_dict: Dict):
        tokens = input_dict["text"]
        tokens = tokens.long()
        embs = self.core(tokens)
        return embs


class AttentionPooling(nn.Module):

    def __init__(self, emb_dim):
        super().__init__()
        self.fc = nn.Linear(emb_dim, 1)

    def forward(self, x, lens):
        # x: [bs, seq_len, emb_dim]
        score = self.fc(x).squeeze(-1)
        mask = generate_length_mask(lens).to(x.device)
        score = score.masked_fill(mask == 0, -1e10)
        weight = torch.softmax(score, dim=1)
        out = (x * weight.unsqueeze(-1)).sum(1)
        return out


class EmbeddingAgg(nn.Module):

    def __init__(self,
                 vocab_size,
                 embed_dim,
                 pretrained_embedding: str = None,
                 freeze_embedding: bool = False,
                 aggregation: str = "mean"):
        super().__init__()
        self.embedding = EmbeddingLayer(vocab_size, embed_dim,
            pretrained_embedding, freeze_embedding)
        self.embed_dim = self.embedding.embed_dim
        self.agg = aggregation
        if aggregation == "attention":
            self.attn = AttentionPooling(embed_dim)

    def forward(self, input_dict):
        embs = self.embedding(input_dict)
        lens = torch.as_tensor(input_dict["text_len"])
        if self.agg == "mean":
            out = mean_with_lens(embs, lens)
        elif self.agg == "attention":
            out = self.attn(embs, lens)
        else:
            raise Exception(f"{self.agg} not supported")
        return {
            "token_emb": embs,
            "seq_emb": out
        }


class RnnEncoder(nn.Module):
    
    def __init__(self,
                 vocab_size,
                 embed_dim,
                 hidden_dim,
                 num_layers,
                 dropout,
                 bidirectional,
                 rnn_type,
                 pooling="mean"):
        super().__init__()
        self.embedding = EmbeddingLayer(vocab_size, embed_dim)
        assert rnn_type in ("RNN", "GRU", "LSTM")
        self.rnn = getattr(nn.modules.rnn, rnn_type)(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )
        self.embed_dim = hidden_dim * (bidirectional + 1)
        self.pooling = pooling

    
    def forward(self, input_dict):
        x = self.embedding(input_dict)
        token_emb, h = self.rnn(x) # token_emb: (bs, token_num, hid_dim(*2) )
        
        text_len = torch.as_tensor(input_dict["text_len"])
        if self.pooling == "mean":
            seq_emb = mean_with_lens(token_emb, text_len)

        return {
            "token_emb": token_emb,
            "seq_emb": seq_emb
        }


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class ConvGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size=1):
        super(ConvGRUCell,self).__init__()
        self.input_size  = input_size
        self.cuda_flag   = True
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reset_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.update_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.out_gate = nn.Linear(input_size + hidden_size, hidden_size)

        init.orthogonal_(self.reset_gate.weight)
        init.orthogonal_(self.update_gate.weight)
        init.orthogonal_(self.out_gate.weight)
        init.constant_(self.reset_gate.bias, 0.)
        init.constant_(self.update_gate.bias, 0.)
        init.constant_(self.out_gate.bias, 0.)

    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.size()[0]
        spatial_size = input_.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            if torch.cuda.is_available():
                prev_state = torch.zeros(state_size).cuda()
            else:
                prev_state = torch.zeros(state_size)

        # data size is [batch, channel, len]
        stacked_inputs = torch.cat([input_, prev_state], dim=2)
        update = torch.sigmoid(self.update_gate(stacked_inputs))
        reset = torch.sigmoid(self.reset_gate(stacked_inputs))
        out_inputs = torch.tanh(self.out_gate(torch.cat([input_, prev_state * reset], dim=2)))
        new_state = prev_state * (1 - update) + out_inputs * update

        return new_state


class IntraAttention(nn.Module):

    def __init__(self,
                 embedding: nn.Module,
                 num_layers: int,
                 pooling: str = "mean") -> None:
        super().__init__()
        self.embedding = embedding
        self.embed_dim = embedding.embed_dim
        self.pe = PositionalEncoding(self.embed_dim, 0.2)
        self.conv_gru = ConvGRUCell(self.embed_dim, self.embed_dim)
        self.num_layers = num_layers
        self.pooling = pooling

    def forward(self, input_dict):
        len = input_dict["text_len"]
        x = self.embedding(input_dict)
        if isinstance(x, dict) and "token_emb" in x:
            x = x["token_emb"]
        batch_size, max_len, emb_dim = x.size()

        for _ in range(self.num_layers):
            score = torch.bmm(self.pe(x), torch.transpose(self.pe(x), 1, 2).contiguous())
            len = torch.as_tensor(len)
            mask1 = torch.arange(max_len).repeat(max_len * batch_size).view(
                batch_size, max_len, max_len) < len.view(-1, 1, 1)
            mask2 = torch.arange(max_len).repeat(max_len * batch_size).view(
                batch_size, max_len, max_len).transpose(1, 2) < len.view(-1, 1, 1)
            mask = mask1 * mask2
            mask = mask.to(score.device)
            score = score.masked_fill(mask == 0, 1e-10)
            attn = torch.softmax(score, dim=2)
            message = torch.bmm(attn, x)

            x = self.conv_gru(message, x)

        if self.pooling == "mean":
            seq_emb = mean_with_lens(x, len)

        return {
            "token_emb": x,
            "seq_emb": seq_emb
        }


class SelfAttention(nn.Module):

    def __init__(self,
                 vocab_size,
                 embed_dim,
                 num_heads,
                 dropout=0.2,
                 pretrained_embedding=None,
                 freeze_embedding=False) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = EmbeddingLayer(vocab_size, embed_dim,
            pretrained_embedding, freeze_embedding)
        self.pe = PositionalEncoding(embed_dim, dropout)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout,
                                         batch_first=True)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def forward(self, input_dict):
        x = self.embedding(input_dict)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pe(x)
        lens = torch.as_tensor(input_dict["text_len"]) + 1
        padding_mask = ~generate_length_mask(lens).to(x.device)
        x, attn = self.mha(x, x, x, padding_mask)
        return {
            "token_emb": x[:, 1:],
            "seq_emb": x[:, 0]
        }


class Bert(nn.Module):

    def __init__(self, model_type, max_length):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        tokenizer_cache = {
            "prajjwal1/bert-medium": "/mnt/lustre/sjtu/home/xnx98/work/AudioTextPretrain/bert_cache/bert_medium_tokenizer",
            "bert-base-uncased": "/mnt/lustre/sjtu/home/xnx98/work/AudioTextPretrain/bert_cache/bert_base_uncased_tokenizer"
        }
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_type)
        except:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_cache[model_type])
        self.model = AutoModel.from_pretrained(model_type, proxies={
            "http": "http://cu01:14227",
            "https": "http://cu01:14227",
        })
        self.dummy_param = nn.Parameter(torch.zeros(1))
        self.max_length = max_length
        self.embed_dim = self.model.config.hidden_size

    def load_pretrained(self, pretrained, output_fn):
        pass

    def forward(self, text):
        tokens = self.tokenizer(text, padding="max_length", return_tensors="pt",
                                truncation=True, max_length=self.max_length)
        for k in tokens:
            tokens[k] = tokens[k].to(self.dummy_param.device)
        output = self.model(**tokens)
        # [CLS] pooling
        text_emb = output.last_hidden_state[:, 0, :]
        return {
            "seq_emb": text_emb,
            "token_emb": output.last_hidden_state,
            "attention_mask": tokens["attention_mask"]
        }


class SentenceBert(nn.Module):

    def __init__(self, model_type):
        super().__init__()
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_type)
        self.dummy_param = nn.Parameter(torch.zeros(1))
        self.embed_dim = self.model[0].auto_model.config.hidden_size

    def forward(self, text):
        tokens = self.model.tokenize(text)
        for k in tokens:
            tokens[k] = tokens[k].to(self.dummy_param.device)
        output = self.model(tokens)
        return {
            "seq_emb": output["sentence_embedding"],
            "token_emb": output
        }


