import math
import json
import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio import transforms
from transformers import PreTrainedModel, PretrainedConfig, ClapModel, ClapProcessor, AutoTokenizer
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

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False
        )

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False
        )

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
            mel_scale="slaney"
        )
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
        x = F.dropout(
            x, p=0.2, training=self.training
        )  # (batch_size, 256, time_steps / 4, mel_bins / 16)
        x = torch.mean(x, dim=3)

        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        x, _ = self.rnn(x)

        length = torch.div(
            torch.as_tensor(input_dict["waveform_len"]),
            self.hop_length,
            rounding_mode="floor"
        ) + 1

        length = torch.div(
            length, self.downsample_ratio, rounding_mode="floor"
        )

        return {"embedding": x, "length": length}


class LaionClapEncoder(nn.Module):
    def __init__(self, model_type: str):
        super().__init__()
        self.tokenizer = ClapProcessor.from_pretrained(model_type)
        model = ClapModel.from_pretrained(model_type)
        self.model = model.text_model
        self.projection = model.text_projection
        self.embed_dim = model.text_projection.config.projection_dim

    def forward(self, input_dict):
        required_keys = ["input_ids", "attention_mask"]
        tokens = {k: input_dict[k].long() for k in required_keys}
        output = self.model(**tokens)
        token_emb = self.projection(output.last_hidden_state)
        seq_emb = self.projection(output.pooler_output)
        seq_emb = F.normalize(seq_emb, dim=-1)
        return {"seq_emb": seq_emb, "token_emb": token_emb}


class DotProduct(nn.Module):
    def __init__(self, l2norm=False, scale=True, text_level="seq") -> None:
        super().__init__()
        self.l2norm = l2norm
        self.scale = scale
        self.text_level = text_level

    def forward(self, input_dict):
        audio = input_dict["audio_emb"]  # [bs, n_seg, dim]
        text = input_dict["text_emb"]
        if self.text_level == "seq":  # [bs, dim]
            text = text["seq_emb"]
        elif self.text_level == "token":
            text = text["token_emb"]  # [bs, n_token, dim]

        if self.l2norm:
            audio = F.normalize(audio, dim=-1)
            text = F.normalize(text, dim=-1)
        if text.ndim == 2:
            text = text.unsqueeze(1)
        score = (audio * text).sum(-1)
        if self.scale:
            score = score / math.sqrt(audio.size(-1))
        score = torch.sigmoid(score).clamp(1e-7, 1.0)
        return score


class BiEncoder(nn.Module):
    def __init__(
        self,
        audio_encoder: nn.Module,
        text_encoder: nn.Module,
        match_fn: nn.Module,
        shared_dim: int,
        cross_encoder: Optional[nn.Module] = None,
        add_proj: bool = False,
        upsample: bool = False,
        freeze_audio_encoder: bool = False,
        freeze_text_encoder: bool = False,
        pretrained: Optional[str] = None
    ):
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
        if pretrained is not None and type(self) is BiEncoder:
            self.load_pretrained(pretrained)
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
                forward_dict["audio_emb"]
            )
        if hasattr(self, "text_proj"):
            text_emb = forward_dict["text_emb"]
            if "seq_emb" in text_emb:
                text_emb["seq_emb"] = self.text_proj(text_emb["seq_emb"])
            if "token_emb" in text_emb:
                text_emb["token_emb"] = self.text_proj(text_emb["token_emb"])
        frame_sim = self.match_fn(forward_dict)  # [batch_size, max_len]
        length = audio_output["length"]
        if self.interpolate_ratio != 1 and self.upsample:
            frame_sim = F.interpolate(
                frame_sim.unsqueeze(1),
                frame_sim.size(1) * self.interpolate_ratio,
                mode="linear",
                align_corners=False
            ).squeeze(1)
            length = length * self.interpolate_ratio
        return {"frame_sim": frame_sim, "length": length}


class Cnn8RnnLaionClapGroundingConfig(PretrainedConfig):
    def __init__(
        self,
        sample_rate: int = 32000,
        shared_dim: int = 512,
        text_encoder_name: str = "laion/clap-htsat-fused",
        **kwargs
    ):
        self.sample_rate = sample_rate
        self.shared_dim = shared_dim
        self.text_encoder_name = text_encoder_name
        super().__init__(**kwargs)


class Cnn8RnnLaionClapGroundingModel(PreTrainedModel):
    config_class = Cnn8RnnLaionClapGroundingConfig

    def __init__(self, config):
        super().__init__(config)
        audio_encoder = Cnn8Rnn(sample_rate=config.sample_rate)
        text_encoder = LaionClapEncoder(model_type=config.text_encoder_name)
        match_fn = DotProduct()
        self.text_tokenizer = AutoTokenizer.from_pretrained(
            config.text_encoder_name
        )
        self.model = BiEncoder(
            audio_encoder=audio_encoder,
            text_encoder=text_encoder,
            match_fn=match_fn,
            shared_dim=config.shared_dim,
            add_proj=True,
        )

    def forward(
        self, audio: torch.Tensor, audio_len: torch.Tensor, text: List[str]
    ):
        device = self.device
        tokens = self.text_tokenizer(
            text, padding=True, return_tensors="pt", truncation=True
        ).to(device)
        tokens["text_len"] = tokens.attention_mask.sum(dim=-1)
        input_dict = {
            "waveform": audio.to(device),
            "waveform_len": audio_len,
        }
        input_dict.update(tokens)
        output = self.model(input_dict)
        return output["frame_sim"]
