import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import init_weights, linear_softmax_with_lens, max_with_lens, \
        mean_by_group
from utils.train_util import do_mixup


class GroundingModel(nn.Module):

    def __init__(self, audio_encoder, text_encoder, match_fn):
        super().__init__()
        self.audio_encoder = audio_encoder
        self.text_encoder = text_encoder
        self.match_fn = match_fn

    def forward(self, input_dict):
        raise NotImplementedError


class BiEncoder(GroundingModel):

    def __init__(self,
                 audio_encoder,
                 text_encoder,
                 match_fn,
                 shared_dim,
                 add_proj=False,
                 upsample=False,
                 freeze_audio_encoder=False,
                 freeze_text_encoder=False):
        super().__init__(audio_encoder, text_encoder, match_fn)
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
        if hasattr(self, "audio_proj"):
            audio_emb = self.audio_proj(audio_output["embedding"])
        else:
            audio_emb = audio_output["embedding"]
        text_emb = self.text_encoder(input_dict) # [batch_size, emb_dim]
        if hasattr(self, "text_proj"):
            text_emb = self.text_proj(text_emb)
        frame_sim = self.match_fn(**{
            "audio": audio_emb,
            "text": text_emb,
            "audio_len": audio_output["length"],
            "text_len": input_dict["text_len"]
        }) # [batch_size, max_len]
        length = audio_output["length"]
        if self.interpolate_ratio != 1 and self.upsample:
            frame_sim = F.interpolate(
                frame_sim.unsqueeze(1),
                frame_sim.size(1) * self.interpolate_ratio,
                mode="linear",
                align_corners=False
            ).squeeze(1)
            length = length * self.interpolate_ratio
        return {
            "frame_sim": frame_sim,
            "length": length
        }


class MultiTextBiEncoder(BiEncoder):

    def __init__(self,
                 audio_encoder,
                 text_encoder,
                 match_fn,
                 shared_dim,
                 text_forward_keys,
                 pooling="linear_softmax",
                 add_proj=False,
                 upsample=False,
                 freeze_audio_encoder=False,
                 freeze_text_encoder=False):
        super().__init__(audio_encoder, text_encoder, match_fn, shared_dim,
                         add_proj, upsample, freeze_audio_encoder,
                         freeze_text_encoder)
        self.text_forward_keys = text_forward_keys
        self.pooling = pooling

    def forward(self, input_dict):
        audio_output = self.audio_encoder(input_dict)
        audio_emb = audio_output["embedding"]
        if hasattr(self, "audio_proj"):
            audio_emb = self.audio_proj(audio_emb)

        batch_size = audio_emb.size(0)
        text_num = input_dict[self.text_forward_keys[0]].shape[1]
        text_forward_dict = {}
        for key in self.text_forward_keys:
            x = input_dict[key]
            text_forward_dict[key] = x.reshape(x.shape[0] * x.shape[1],
                                               *x.shape[2:])

        text_emb = self.text_encoder(text_forward_dict)
        if hasattr(self, "text_proj"):
            text_emb = self.text_proj(text_emb)

        audio_emb = audio_emb.unsqueeze(1).expand(-1, text_num, -1, -1) # (batch_size, text_num, max_len, emb_dim)
        audio_emb = audio_emb.reshape(-1, *audio_emb.shape[2:])
        audio_len = audio_output["length"].unsqueeze(1).expand(-1, text_num)
        audio_len = audio_len.reshape(-1)
        text_len = text_forward_dict["text_len"]
        frame_sim = self.match_fn(**{
            "audio": audio_emb,
            "text": text_emb,
            "audio_len": audio_len,
            "text_len": text_len
        }) # [batch_size * text_num, max_len]
        length = audio_output["length"]
        frame_sim = frame_sim.reshape(batch_size, text_num, -1).transpose(1, 2)
        if self.pooling == "linear_softmax":
            clip_sim = linear_softmax_with_lens(frame_sim, length)
        elif self.pooling == "max":
            clip_sim = max_with_lens(frame_sim, length)
        else:
            raise Exception(f"Unsupported pooling {self.pooling}")
        if self.interpolate_ratio != 1 and self.upsample:
            frame_sim = F.interpolate(
                frame_sim,
                frame_sim.size(-1) * self.interpolate_ratio,
                mode="linear",
                align_corners=False
            )
            length = length * self.interpolate_ratio
        return {
            "frame_sim": frame_sim,
            "clip_sim": clip_sim,
            "length": length
        }


class AudioTagging(nn.Module):

    def __init__(self, audio_encoder, classes_num, pooling="linear_softmax"):
        super().__init__()
        self.backbone = audio_encoder
        self.fc_output = nn.Linear(audio_encoder.embed_dim, classes_num)
        self.pooling = pooling

    def load_pretrained(self, pretrained, output_fn, training=True, cnn_only=False):
        if isinstance(pretrained, dict):
            state_dict = pretrained
        else:
            state_dict = torch.load(pretrained, map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]
        model_dict = self.state_dict()
        pretrained_dict = {
            k: v for k, v in state_dict.items() if (k in model_dict) and (
                model_dict[k].shape == v.shape)
        }
        if cnn_only and training:
            filtered_dict = {}
            for k, v in pretrained_dict.items():
                if k.startswith("backbone.rnn") or k.startswith("backbone.fc1") \
                    or k.startswith("fc_output"):
                    continue
                filtered_dict[k] = v
            pretrained_dict = filtered_dict
        output_fn(f"Loading pretrained keys {pretrained_dict.keys()}")
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict, strict=True)
    
    def forward(self, input_dict):
        output = self.backbone(input_dict)
        logit = self.fc_output(output["embedding"])
        prob = torch.sigmoid(logit)
        if self.pooling == "linear_softmax":
            clip_prob = linear_softmax_with_lens(prob, output["length"])
        elif self.pooling == "max":
            clip_prob = max_with_lens(prob, output["length"])
        else:
            raise Exception(f"Unsupported pooling {self.pooling}")
        return {
            "frame_sim": prob,
            "clip_sim": clip_prob,
            "length": output["length"]
        }


class CDurTextBlock(nn.Module):

    def __init__(self, cin, cout, text_emb_dim, kernel_size=3, padding=1) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(cin)
        self.conv = nn.Conv2d(cin, cout, kernel_size, padding=padding, bias=False)
        self.activation = nn.LeakyReLU(0.1, True)
        self.fc_text = nn.Linear(text_emb_dim, cout)

    def forward(self, x, text):
        x = self.bn(x)
        x = self.conv(x)
        text = self.fc_text(text)
        x = x + text.unsqueeze(-1).unsqueeze(-1)
        x = self.activation(x)
        return x


class CrossCrnn(nn.Module):

    def __init__(self, sample_rate, text_encoder, upsample=False) -> None:
        from torchaudio import transforms
        super().__init__()
        self.text_encoder = text_encoder
        self.melspec_extractor = transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=2048,
            win_length=40 * sample_rate // 1000,
            hop_length=20 * sample_rate // 1000,
            n_mels=64
        )
        self.hop_length = 20 * sample_rate // 1000
        self.db_transform = transforms.AmplitudeToDB()
        self.text_emb_dim = text_encoder.embed_dim
        self.block1 = CDurTextBlock(1, 32, self.text_emb_dim)
        self.pool1 = nn.LPPool2d(4, (2, 4))
        self.block2 = CDurTextBlock(32, 128, self.text_emb_dim)
        self.block3 = CDurTextBlock(128, 128, self.text_emb_dim)
        self.pool2 = nn.LPPool2d(4, (2, 4))
        self.block4 = CDurTextBlock(128, 128, self.text_emb_dim)
        self.block5 = CDurTextBlock(128, 128, self.text_emb_dim)
        self.pool3 = nn.LPPool2d(4, (1, 4))
        self.dropout = nn.Dropout(0.3)
        rnn_input_dim = self.get_rnn_input_dim()
        self.gru = nn.GRU(rnn_input_dim, 128, bidirectional=True, batch_first=True)
        self.fc_text = nn.Linear(self.text_emb_dim, 256)
        self.fc_output = nn.Linear(256, 1)
        self.apply(init_weights)
        self.interpolate_ratio = 4
        self.upsample = upsample

    def forward_cnn(self, audio, text):
        x = self.block1(audio, text)
        x = self.pool1(x)
        x = self.block2(x, text)
        x = self.block3(x, text)
        x = self.pool2(x)
        x = self.block4(x, text)
        x = self.block5(x, text)
        x = self.pool3(x)
        x = self.dropout(x)
        return x

    def get_rnn_input_dim(self):
        with torch.no_grad():
            audio = self.db_transform(self.melspec_extractor(
                torch.randn(1, 16000)))
            audio = audio.transpose(1, 2).unsqueeze(1)
            text = torch.randn(1, self.text_emb_dim)
            x = self.forward_cnn(audio, text)
            rnn_input_dim = x.size(1) * x.size(-1)
        return rnn_input_dim

    def forward(self, input_dict):
        text_emb = self.text_encoder(input_dict)
        waveform = input_dict["waveform"]
        mel_spec = self.melspec_extractor(waveform)
        lms = self.db_transform(mel_spec)

        x = lms.transpose(1, 2).unsqueeze(1) # [bs, 1, timesteps, n_mels]
        x = self.forward_cnn(x, text_emb)
        x = x.transpose(1, 2).contiguous().flatten(-2)
        x, _ = self.gru(x)
        x = x + self.fc_text(text_emb).unsqueeze(1)
        logit = self.fc_output(x)
        prob = torch.sigmoid(logit).squeeze(-1).clamp(1e-7, 1.)
        length = torch.div(
            torch.as_tensor(input_dict["waveform_len"]),
            self.hop_length,
            rounding_mode="floor") + 1
        length = torch.div(length, self.interpolate_ratio, rounding_mode="floor")
        if self.interpolate_ratio != 1 and self.upsample:
            prob = F.interpolate(
                prob.unsqueeze(1),
                prob.size(1) * self.interpolate_ratio,
                mode="linear",
                align_corners=False
            ).squeeze(1)
            length = length * self.interpolate_ratio
        return {
            "prob": prob,
            "length": length
        }


class ConvTextBlock(nn.Module):

    def __init__(self, in_channels, out_channels, text_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.fc_text = nn.Linear(text_emb_dim, out_channels)

        self.init_weight()

    def init_layer(self, layer):
        """Initialize a Linear or Convolutional layer. """
        nn.init.xavier_uniform_(layer.weight)
     
        if hasattr(layer, 'bias'):
            if layer.bias is not None:
                layer.bias.data.fill_(0.)
        
    def init_bn(self, bn):
        """Initialize a Batchnorm layer. """
        bn.bias.data.fill_(0.)
        bn.weight.data.fill_(1.)

    def init_weight(self):
        self.init_layer(self.conv1)
        self.init_layer(self.conv2)
        self.init_layer(self.fc_text)
        self.init_bn(self.bn1)
        self.init_bn(self.bn2)
        
    def forward(self, audio, text, pool_size=(2, 2), pool_type='avg'):
        # audio: [batch_size, n_channel, time_steps, n_freq] 
        # text: [batch_size, n_channel]
        x = audio
        text = self.fc_text(text)
        x = F.relu_(self.bn1(self.conv1(x)) + text.unsqueeze(-1).unsqueeze(-1))
        x = F.relu_(self.bn2(self.conv2(x)) + text.unsqueeze(-1).unsqueeze(-1))
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


class CrossCnn8_Rnn(nn.Module):

    def __init__(self, sample_rate, text_encoder, freeze_cnn=False,
                 freeze_bn=False, upsample=False) -> None:
        from torchaudio import transforms
        from torchlibrosa import SpecAugmentation
        super().__init__()
        self.text_encoder = text_encoder

        self.interpolate_ratio = 4
        self.upsample = upsample
        self.freeze_cnn = freeze_cnn
        self.freeze_bn = freeze_bn

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
        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64, time_stripes_num=2,
            freq_drop_width=8, freq_stripes_num=2)

        self.text_emb_dim = text_encoder.embed_dim

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvTextBlock(1, 64, self.text_emb_dim)
        self.conv_block2 = ConvTextBlock(64, 128, self.text_emb_dim)
        self.conv_block3 = ConvTextBlock(128, 256, self.text_emb_dim)
        self.conv_block4 = ConvTextBlock(256, 512, self.text_emb_dim)

        self.fc1 = nn.Linear(512, 512, bias=True)
        self.fc1_text = nn.Linear(self.text_emb_dim, 512)
        self.rnn = nn.GRU(512, 256, bidirectional=True, batch_first=True)
        self.rnn_text = nn.Linear(self.text_emb_dim, 512)

        self.fc_output = nn.Linear(512, 1)
        
        self.init_weight()

        if self.freeze_cnn:
            for param in self.parameters():
                param.requires_grad = False
            for param in self.rnn.parameters():
                param.requires_grad = True

    def load_pretrained(self, pretrained, output_fn, training=True, cnn_only=False):
        if isinstance(pretrained, dict):
            state_dict = pretrained
        else:
            state_dict = torch.load(pretrained, map_location="cpu")

        if "model" in state_dict:
            state_dict = state_dict["model"]

        model_dict = self.state_dict()
        pretrained_dict = {
            k: v for k, v in state_dict.items() if (k in model_dict) and (
                model_dict[k].shape == v.shape)
        }
        if cnn_only and training:
            filtered_dict = {}
            for k, v in pretrained_dict.items():
                if k.startswith("rnn") or k.startswith("fc1") \
                    or k.startswith("fc_output"):
                    continue
                filtered_dict[k] = v
            pretrained_dict = filtered_dict
        output_fn(f"Loading pretrained keys {pretrained_dict.keys()}")
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict, strict=True)

    def train(self, mode: bool = True):
        super().train(mode=mode)
        if self.freeze_bn:
            def bn_eval(module):
                class_name = module.__class__.__name__
                if class_name.find("BatchNorm") != -1:
                    module.eval()
            self.apply(bn_eval)
        return self

    def init_layer(self, layer):
        """Initialize a Linear or Convolutional layer. """
        nn.init.xavier_uniform_(layer.weight)
     
        if hasattr(layer, 'bias'):
            if layer.bias is not None:
                layer.bias.data.fill_(0.)
        
    def init_bn(self, bn):
        """Initialize a Batchnorm layer. """
        bn.bias.data.fill_(0.)
        bn.weight.data.fill_(1.)

    def init_weight(self):
        self.init_bn(self.bn0)
        self.init_layer(self.fc1)
        self.init_layer(self.fc1_text)
        self.init_layer(self.rnn_text)
        self.init_layer(self.fc_output)

    def forward_cnn(self, audio, text):
        x = self.conv_block1(audio, text, pool_size=(2, 2), pool_type='avg+max')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, text, pool_size=(2, 2), pool_type='avg+max')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, text, pool_size=(1, 2), pool_type='avg+max')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, text, pool_size=(1, 2), pool_type='avg+max')
        x = F.dropout(x, p=0.2, training=self.training) # (batch_size, 512, time_steps / 4, mel_bins / 16)
        return x

    def forward(self, input_dict):
        text_emb = self.text_encoder(input_dict)
        waveform = input_dict["waveform"]
        
        x = self.melspec_extractor(waveform)
        x = self.db_transform(x)    # (batch_size, mel_bins, time_steps)
        x = x.transpose(1, 2)
        x = x.unsqueeze(1)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        specaug = input_dict["specaug"]
        # SpecAugment
        if self.training and specaug:
            x = self.spec_augmenter(x)

        mixup_lambda = input_dict.get("mixup_lambda", None)
        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        x = self.forward_cnn(x, text_emb)
        x = torch.mean(x, dim=3) # (batch_size, 512, time_steps / 4)

        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x) + self.fc1_text(text_emb).unsqueeze(1))

        x, _  = self.rnn(x)
        x = x + self.rnn_text(text_emb).unsqueeze(1)
    
        logit = self.fc_output(x)
        prob = torch.sigmoid(logit).clamp(1e-7, 1.)

        length = torch.div(
            torch.as_tensor(input_dict["waveform_len"]),
            self.hop_length,
            rounding_mode="floor") + 1

        length = torch.div(length, self.interpolate_ratio, rounding_mode="floor")

        if self.interpolate_ratio != 1 and self.upsample:
            prob = F.interpolate(
                prob.transpose(1, 2),
                prob.size(1) * self.interpolate_ratio,
                mode="linear",
                align_corners=False
            ).transpose(1, 2)
            length = length * self.interpolate_ratio

        return {
            "prob": prob,
            "length": length
        }


class AudioTextAlignByWord(nn.Module):

    def __init__(self,
                 audio_encoder,
                 text_encoder,
                 match_fn,
                 sim_pooling,
                 shared_dim,
                 add_proj=False,
                 freeze_audio_encoder=False,
                 freeze_text_encoder=False
                 ):
        super().__init__()
        self.audio_encoder = audio_encoder
        self.text_encoder = text_encoder
        self.match_fn = match_fn
        self.sim_pooling = sim_pooling
        if audio_encoder.embed_dim != text_encoder.embed_dim or add_proj:
            self.audio_proj = nn.Linear(audio_encoder.embed_dim, shared_dim)
            self.text_proj = nn.Linear(text_encoder.embed_dim, shared_dim)
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
            text, text_len,
            pool (True/False)
        """
        audio_output = self.audio_encoder(input_dict)
        if hasattr(self, "audio_proj"):
            audio_emb = self.audio_proj(audio_output["embedding"])
        else:
            audio_emb = audio_output["embedding"]
        # audio_emb: [bs, n_seg, emb_dim]
        text_emb = self.text_encoder(input_dict)
        # word_emb: [bs, n_word, emb_dim]
        if hasattr(self, "text_proj"):
            text_emb = self.text_proj(text_emb)
        sim_matrix = self.match_fn(audio_emb, text_emb)
        # sim: [bs, bs, n_seg, n_word]
        output_matrix = input_dict.get("output_matrix", False)

        sim = self.sim_pooling({
            "sim": sim_matrix,
            "audio_len": audio_output["length"],
            "text_len": input_dict["text_len"]
        })
        
        output = {
            "sim": sim,
        }
        
        if output_matrix:
            output["sim_matrix"] = sim_matrix

        return output


class AudioTextAlignByPhrase(nn.Module):

    def __init__(self,
                 audio_encoder,
                 text_encoder,
                 match_fn,
                 shared_dim,
                 add_proj=False,
                 freeze_audio_encoder=False,
                 freeze_text_encoder=False
                 ):
        super().__init__()
        self.audio_encoder = audio_encoder
        self.text_encoder = text_encoder
        self.match_fn = match_fn
        if audio_encoder.embed_dim != text_encoder.embed_dim or add_proj:
            self.audio_proj = nn.Linear(audio_encoder.embed_dim, shared_dim)
            self.text_proj = nn.Linear(text_encoder.embed_dim, shared_dim)
        if freeze_audio_encoder:
            for param in self.audio_encoder.parameters():
                param.requires_grad = False
        if freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

    def forward(self, input_dict):
        """
        input_dict:
            waveform: [bs, n_samples]
            waveform_len: [bs]
            phrases: [txt_num, max_txt_len]
            phrases_num: [bs]
            phrases_len: [txt_num]
            text_key: str
        """
        audio_output = self.audio_encoder(input_dict)
        if hasattr(self, "audio_proj"):
            audio_emb = self.audio_proj(audio_output["embedding"])
        else:
            audio_emb = audio_output["embedding"]
        # audio_emb: [bs, n_seg, emb_dim]
        phrases_emb = self.text_encoder({
            "text": input_dict["phrases"],
            "text_len": input_dict["phrases_len"]
        })
        if hasattr(self, "text_proj"):
            phrases_emb = self.text_proj(phrases_emb)
        # text_emb: [txt_num, max_txt_len, emb_dim]

        batch_phrase_num = phrases_emb.size(0)
        phrases_num = input_dict["phrases_num"]
        phrases_len = input_dict["phrases_len"]
        batch_size = audio_emb.size(0)
        sim_matrixes = []
        sims = torch.empty(batch_size, batch_size)

        output_matrix = input_dict.get("output_matrix", False)


        # if input_dict["method"] == "method1":
            # phrases_emb = torch.split(phrases_emb, phrases_num, dim=0)
            # phrases_len = torch.split(torch.as_tensor(phrases_len), phrases_num, dim=0)
            # for i in range(batch_size):
                # for j in range(batch_size):
                    
                    # ph_num_j = phrases_num[j]
                    # sim_matrix = self.match_fn(**{
                        # "audio": audio_emb[i: i+1].expand(ph_num_j, -1, -1),
                        # "text": phrases_emb[j],
                        # "audio_len": [audio_output["length"][i]] * ph_num_j,
                        # "text_len": phrases_len[j]
                    # }) # [ph_num_j, n_seg]
                    # sim_matrix = sim_matrix.transpose(0, 1)
                    
                    # if output_matrix:
                        # sim_matrixes.append(sim_matrix)
                    # sim_pooled = linear_softmax_with_lens(
                        # sim_matrix.mean(1).unsqueeze(0),
                        # [audio_output["length"][i]]
                    # )
                    # sims[i, j] = sim_pooled

        # elif input_dict["method"] == "method2":


        for i in range(batch_size):
            sim_i = self.match_fn(**{
                "audio": audio_emb[i: i+1].expand(batch_phrase_num, -1, -1),
                "text": phrases_emb,
                "audio_len": [audio_output["length"][i]] * batch_phrase_num,
                "text_len": phrases_len
            })
            # sim_i: [txt_num, n_seg]
            if output_matrix:
                for j in range(batch_size):
                    if j == 0:
                        start = 0
                    else:
                        start = end
                    end = phrases_num[j]
                    sim_matrixes.append(sim_i[start: end].transpose(0, 1))
            sim_i = linear_softmax_with_lens(
                sim_i.transpose(0, 1).unsqueeze(0), [audio_output["length"][i]]
            )[0]
            # sim_i: [txt_num]
            sim_i = mean_by_group(sim_i, phrases_num)
            # sim_i: [batch_size]
            sims[i] = sim_i

        return {
            "sim": sims,
            "sim_matrix": sim_matrixes
        }


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.getcwd())
    from models.audio_encoder import CrnnEncoder
    from models.text_encoder import EmbeddingAgg
    from models.match import ExpNegL2
    vocab_size = 1500
    # audio_encoder = CrnnEncoder(32000, 256)
    text_encoder = EmbeddingAgg(vocab_size, 256, aggregation="mean")
    # match_fn = ExpNegL2()
    # model = BiEncoder(audio_encoder, text_encoder, match_fn, 256, True)
    model = CrossCnn8_Rnn(32000, text_encoder)
    print(model)
    input_dict = {
        "waveform": torch.randn(4, 320000),
        "waveform_len": torch.tensor([320000] * 4),
        "text": torch.empty(4, 3).random_(vocab_size),
        "text_len": torch.tensor([2, 3, 3, 2])
    }
    output = model(input_dict)
    print("prob: ", output["prob"].shape, "length: ", output["length"])
