import math
import json
import os
from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio import transforms
from transformers import PreTrainedModel, PretrainedConfig
from transformers.utils.hub import cached_file


def sum_with_lens(features, lens):
    lens = torch.as_tensor(lens)
    if max(lens) != features.size(1):
        max_length = features.size(1)
        mask = generate_length_mask(lens, max_length)
    else:
        mask = generate_length_mask(lens)
    mask = mask.to(features.device)  # [N, T]

    while mask.ndim < features.ndim:
        mask = mask.unsqueeze(-1)
    feature_masked = features * mask
    feature_sum = feature_masked.sum(1)
    return feature_sum


def generate_length_mask(lens, max_length=None):
    lens = torch.as_tensor(lens)
    N = lens.size(0)
    if max_length is None:
        max_length = max(lens)
    idxs = torch.arange(max_length).repeat(N).view(N, max_length)
    mask = (idxs < lens.view(-1, 1))
    return mask


def mean_with_lens(features, lens):
    """
    features: [N, T, ...] (assume the second dimension represents length)
    lens: [N,]
    """
    feature_sum = sum_with_lens(features, lens)
    while lens.ndim < feature_sum.ndim:
        lens = lens.unsqueeze(1)
    feature_mean = feature_sum / lens.to(features.device)
    return feature_mean


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):

        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1),
                               bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1),
                               bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')

        return x


class Cnn8Rnn(nn.Module):

    def __init__(
        self,
        sample_rate,
    ):

        super().__init__()

        self.downsample_ratio = 4  # Downsampled ratio
        self.time_resolution = 0.04

        # Logmel spectrogram extractor
        self.hop_length = int(0.010 * sample_rate)
        self.win_length = int(0.032 * sample_rate)
        if sample_rate == 32000:
            f_max = 14000
        else:
            f_max = int(sample_rate / 2)
        self.melspec_extractor = transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=self.win_length,
            win_length=self.win_length,
            hop_length=self.hop_length,
            f_min=50,
            f_max=f_max,
            n_mels=64,
            norm="slaney",
            mel_scale="slaney")
        self.db_transform = transforms.AmplitudeToDB()

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.fc1 = nn.Linear(512, 512, bias=True)
        self.rnn = nn.GRU(512, 256, bidirectional=True, batch_first=True)
        self.embed_dim = 512

    def forward(self, input_dict: Dict):
        """
        Input: (batch_size, n_samples)"""

        waveform = input_dict["waveform"]
        x = self.melspec_extractor(waveform)
        x = self.db_transform(x)  # (batch_size, mel_bins, time_steps)
        x = x.transpose(1, 2)
        x = x.unsqueeze(1)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg+max')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg+max')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(1, 2), pool_type='avg+max')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(1, 2), pool_type='avg+max')
        x = F.dropout(x, p=0.2, training=self.training
                      )  # (batch_size, 256, time_steps / 4, mel_bins / 16)
        x = torch.mean(x, dim=3)

        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        x, _ = self.rnn(x)

        length = torch.div(torch.as_tensor(input_dict["waveform_len"]),
                           self.hop_length,
                           rounding_mode="floor") + 1

        length = torch.div(length,
                           self.downsample_ratio,
                           rounding_mode="floor")

        return {"embedding": x, "length": length}


class EmbeddingLayer(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.core = nn.Embedding(vocab_size, embed_dim)

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

    def __init__(self, vocab_size, embed_dim, aggregation: str = "mean"):
        super().__init__()
        self.embedding = EmbeddingLayer(vocab_size, embed_dim)
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
        return {"token_emb": embs, "seq_emb": out}


class DotProduct(nn.Module):

    def __init__(self, l2norm=False, scaled=False, text_level="seq"):
        super().__init__()
        self.l2norm = l2norm
        self.scaled = scaled
        self.text_level = text_level

    def forward(self, input_dict):
        audio = input_dict["audio_emb"]  # [bs, n_seg, dim]
        text = input_dict["text_emb"]
        if self.text_level == "seq":  # [bs, dim]
            text = text["seq_emb"]
        elif self.text_level == "token":
            text = text["token_emb"]  # [bs, n_seg, dim]

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


class BiEncoder(nn.Module):

    def __init__(self,
                 audio_encoder,
                 text_encoder,
                 match_fn,
                 shared_dim,
                 cross_encoder=None,
                 add_proj=False,
                 upsample=False,
                 freeze_audio_encoder=False,
                 freeze_text_encoder=False):
        super().__init__()
        self.audio_encoder = audio_encoder
        self.text_encoder = text_encoder
        self.match_fn = match_fn
        self.cross_encoder = cross_encoder
        if audio_encoder.embed_dim != text_encoder.embed_dim or add_proj:
            self.audio_proj = nn.Linear(audio_encoder.embed_dim, shared_dim)
            self.text_proj = nn.Linear(text_encoder.embed_dim, shared_dim)
        self.interpolate_ratio = self.audio_encoder.downsample_ratio
        self.upsample = upsample
        if freeze_audio_encoder:
            for param in self.audio_encoder.parameters():
                param.requires_grad = False
        if freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

    def forward(self, input_dict):
        """
        keys in input_dict:
            waveform, waveform_len,
            text, text_len
        """
        audio_output = self.audio_encoder(input_dict)
        audio_emb = audio_output["embedding"]
        text_emb = self.text_encoder(input_dict)  # [batch_size, emb_dim]
        forward_dict = {
            "audio_emb": audio_emb,
            "text_emb": text_emb,
            "audio_len": audio_output["length"]
        }
        if "text_len" in input_dict:
            forward_dict["text_len"] = input_dict["text_len"]
        if self.cross_encoder is not None:
            cross_encoded = self.cross_encoder(forward_dict)
            # cross_encoded: audio_emb, text_emb, ...
            forward_dict.update(cross_encoded)
        if hasattr(self, "audio_proj"):
            forward_dict["audio_emb"] = self.audio_proj(
                forward_dict["audio_emb"])
        if hasattr(self, "text_proj"):
            text_emb = forward_dict["text_emb"]
            if "seq_emb" in text_emb:
                text_emb["seq_emb"] = self.text_proj(text_emb["seq_emb"])
            if "token_emb" in text_emb:
                text_emb["token_emb"] = self.text_proj(text_emb["token_emb"])
        frame_sim = self.match_fn(forward_dict)  # [batch_size, max_len]
        length = audio_output["length"]
        if self.interpolate_ratio != 1 and self.upsample:
            frame_sim = F.interpolate(frame_sim.unsqueeze(1),
                                      frame_sim.size(1) *
                                      self.interpolate_ratio,
                                      mode="linear",
                                      align_corners=False).squeeze(1)
            length = length * self.interpolate_ratio
        return {"frame_sim": frame_sim, "length": length}


class Cnn8RnnW2vMeanGroundingConfig(PretrainedConfig):

    def __init__(self,
                 sample_rate: int = 32000,
                 vocab_size: int = 5221,
                 embed_dim: int = 512,
                 shared_dim: int = 512,
                 add_proj: bool = False,
                 **kwargs):
        self.sample_rate = sample_rate
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.shared_dim = shared_dim
        self.add_proj = add_proj
        super().__init__(**kwargs)


class Cnn8RnnW2vMeanGroundingModel(PreTrainedModel):
    config_class = Cnn8RnnW2vMeanGroundingConfig

    def __init__(self, config):
        super().__init__(config)
        audio_encoder = Cnn8Rnn(sample_rate=config.sample_rate)
        text_encoder = EmbeddingAgg(embed_dim=config.embed_dim,
                                    vocab_size=config.vocab_size)
        match_fn = DotProduct()
        self.model = BiEncoder(
            audio_encoder=audio_encoder,
            text_encoder=text_encoder,
            match_fn=match_fn,
            shared_dim=config.shared_dim,
            add_proj=config.add_proj,
        )
        self.vocab_mapping = {}

    def forward(self, audio: torch.Tensor, audio_len: torch.Tensor,
                text: List[str]):
        device = self.device
        text_len = torch.as_tensor([len(t.split()) for t in text]).to(device)
        text_tensor = torch.zeros(len(text), text_len.max()).long().to(device)
        for i, txt in enumerate(text):
            token_list = []
            for word in txt.split():
                if not word in self.vocab_mapping:
                    token = self.vocab_mapping["<unk>"]
                else:
                    token = self.vocab_mapping[word]
                token_list.append(token)
            text_tensor[i, :len(token_list)] = torch.tensor(token_list)
        input_dict = {
            "waveform": audio.to(device),
            "waveform_len": audio_len,
            "text": text_tensor,
            "text_len": text_len
        }
        output = self.model(input_dict)
        return output["frame_sim"]

    def save_pretrained(self, save_directory, *args, **kwargs):
        super().save_pretrained(save_directory, *args, **kwargs)
        json.dump(self.vocab_mapping,
                  open(os.path.join(save_directory, "vocab.json"), "w"))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args,
                        **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path,
                                        *model_args, **kwargs)
        vocab_path = cached_file(pretrained_model_name_or_path, "vocab.json")
        model.vocab_mapping = json.load(open(vocab_path))
        return model