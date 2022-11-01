import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import init_weights, linear_softmax_with_lens, max_with_lens


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
        text_emb = self.text_encoder(input_dict) # [N, E]
        if hasattr(self, "text_proj"):
            text_emb = self.text_proj(text_emb)
        prob = self.match_fn(**{
            "audio": audio_emb,
            "text": text_emb,
            "audio_len": audio_output["length"],
            "text_len": input_dict["text_len"]
        }) # [N, T]
        length = audio_output["length"]
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


class AudioTextAlign(nn.Module):

    def __init__(self,
                 audio_encoder,
                 text_encoder,
                 match_fn,
                 pooling,
                 shared_dim,
                 add_proj=False,
                 freeze_audio_encoder=False,
                 freeze_text_encoder=False
                 ):
        super().__init__()
        self.audio_encoder = audio_encoder
        self.text_encoder = text_encoder
        self.match_fn = match_fn
        self.pooling = pooling
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
        sim = self.match_fn(audio_emb, text_emb)
        # sim: [bs, bs, n_seg, n_word]
        pool = input_dict.get("pool", True)
        if pool:
            sim = self.pooling({
                "sim": sim,
                "audio_len": audio_output["length"],
                "text_len": input_dict["text_len"]
            })
        return {
            "sim": sim,
        }


class VectorQuantizer(nn.Module):
    """
    see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
    ____________________________________________
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    _____________________________________________
    """

    def __init__(self, n_e, e_dim, beta):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        2d: z.shape = (batch, channel, height, width)
        1d: z.shape = (batch, channel, time)
        quantization pipeline:
            1. get encoder input 2d: (B,C,H,W) or 1d: (B, C, T)
            2. flatten input to 2d: (B*H*W,C) or 1d: (B*T, C)
        """
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        
        d = torch.sum(z ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.matmul(z, self.embedding.weight.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)

        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z.detach() - z_q) ** 2) + \
            self.beta * torch.mean((z - z_q.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        # e_mean = torch.mean(min_encodings, dim=0)
        # perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        return {
            "emb": z_q,
            "loss": loss
        }

    def get_codebook_entry(self, indices):
        min_encodings = torch.zeros(indices.shape[0], self.n_e).to(indices)
        min_encodings.scatter_(1, indices[:, None], 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)

        return z_q


class AudioMultiText(AudioTextAlign):

    def __init__(self, audio_encoder, text_encoder, match_fn, pooling,
                 shared_dim, quantize=True, codebook_size=300, add_proj=False,
                 freeze_audio_encoder=False, freeze_text_encoder=False):
        super().__init__(audio_encoder, text_encoder, match_fn, pooling,
                         shared_dim, add_proj=add_proj,
                         freeze_audio_encoder=freeze_audio_encoder,
                         freeze_text_encoder=freeze_text_encoder)
        self.quantize = quantize
        if quantize:
            self.vq = VectorQuantizer(codebook_size,
                                      self.text_encoder.embed_dim,
                                      0.25)

    def forward(self, input_dict):
        audio_output = self.audio_encoder(input_dict)
        if hasattr(self, "audio_proj"):
            audio_emb = self.audio_proj(audio_output["embedding"])
        else:
            audio_emb = audio_output["embedding"]
        # audio_emb: [bs, n_seg, emb_dim]
        if isinstance(input_dict["text"], torch.Tensor):
            batch_size, txt_num, max_txt_len = input_dict["text"].size()
            forward_dict = {}
            forward_dict["text"] = input_dict["text"].reshape(-1, max_txt_len)
            forward_dict["text_len"] = input_dict["text_len"].reshape(-1)
            text_emb = self.text_encoder(forward_dict)
        elif isinstance(input_dict["text"], list):
            batch_size = len(input_dict["text"])
            txt_num = len(input_dict["text"][0])
            text = sum(input_dict["text"], [])
            text_emb = self.text_encoder(text)["sentence_emb"]
        # text_emb: [bs, txt_num, emb_dim]
        if self.quantize:
            vq_out = self.vq(text_emb)
            text_emb = vq_out["emb"]
            vq_loss = vq_out["loss"]
        if hasattr(self, "text_proj"):
            text_emb = self.text_proj(text_emb)
        text_emb = text_emb.reshape(batch_size, txt_num, -1)
        sim = self.match_fn(audio_emb, text_emb)
        # sim: [bs, txt_num, n_seg]
        clip_sim = self.pooling({
            "sim": sim,
            "audio_len": audio_output["length"]
        })
        output = {
            "frame_sim": sim,
            "clip_sim": clip_sim
        }
        if self.quantize:
            output["vq_loss"] = vq_loss
        return output


class TaggingQuantizer(nn.Module):

    def __init__(self, n_e, e_dim, beta,
                 distance_measure="cosine", pretrained_weight=None,
                 freeze_weight=False, threshold=None):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.distance_measure = distance_measure
        self.threshold = threshold

        if pretrained_weight is None:
            self.embedding = nn.Embedding(self.n_e, self.e_dim)
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        else:
            weight = np.load(pretrained_weight)
            weight = torch.as_tensor(weight, dtype=torch.float)
            self.embedding = nn.Embedding.from_pretrained(weight, freeze_weight)

    def forward(self, z):

        if self.distance_measure == "l2":
            d = torch.sum(z ** 2, dim=1, keepdim=True) + \
                torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
                torch.matmul(z, self.embedding.weight.t())
        elif self.distance_measure == "cosine":
            d = -F.normalize(z, dim=-1) @ F.normalize(self.embedding.weight, dim=-1).transpose(0, 1)
        
        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1)
        
        z_q = self.embedding(min_encoding_indices)

        # compute loss for embedding
        loss = torch.mean((z.detach() - z_q) ** 2) + \
            self.beta * torch.mean((z - z_q.detach()) ** 2)

        if self.threshold is not None:
            sim = -d
            pos_indices = torch.where(sim > self.threshold)[1].unique()
        else:
            pos_indices = min_encoding_indices.unique()

        z_q = self.embedding(pos_indices)
        
        neg_indices = np.delete(np.arange(self.n_e), pos_indices.cpu().numpy())
        neg_indices = torch.as_tensor(neg_indices).to(z_q).long()
        z_q_neg = self.embedding(neg_indices)
        label = torch.as_tensor([1] * pos_indices.shape[0] + [0] * neg_indices.shape[0])
        label = label.to(z_q)
        z_q = torch.cat((z_q, z_q_neg))

        return {
            "emb": z_q,
            "label": label,
            "loss": loss
        }


class AudioTextTagging(nn.Module):

    def __init__(self, audio_encoder, text_encoder, match_fn, pooling,
                 shared_dim, codebook_embed_dim, codebook_size=300,
                 add_proj=False, freeze_audio_encoder=False,
                 freeze_text_encoder=False, distance_measure="cosine",
                 pretrained_quantize_weight=None, freeze_quantize_weight=False,
                 quantize_threshold=None):
        super().__init__()
        self.audio_encoder = audio_encoder
        self.text_encoder = text_encoder
        self.match_fn = match_fn
        self.pooling = pooling
        self.quantize = True
        self.vq = TaggingQuantizer(codebook_size, codebook_embed_dim, 0.25,
                                   distance_measure, pretrained_quantize_weight,
                                   freeze_quantize_weight, quantize_threshold)
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
        audio_output = self.audio_encoder(input_dict)
        if hasattr(self, "audio_proj"):
            audio_emb = self.audio_proj(audio_output["embedding"])
        else:
            audio_emb = audio_output["embedding"]
        # audio_emb: [bs, n_seg, emb_dim]
        batch_size = len(input_dict["text"])
        txt_num = input_dict["text_num"]
        text = sum(input_dict["text"], [])
        text_emb = self.text_encoder(text)
        text_embs = torch.split(text_emb, txt_num, dim=0)
        vq_loss = 0
        labels = []
        vq_embs = []
        for text_emb in text_embs:
            vq_out = self.vq(text_emb)
            vq_embs.append(vq_out["emb"])
            vq_loss += vq_out["loss"]
            labels.append(vq_out["label"])
        vq_loss = vq_loss / len(text_embs)
        label = torch.stack(labels)
        vq_emb = torch.stack(vq_embs)
        if hasattr(self, "text_proj"):
            vq_emb = self.text_proj(vq_emb)
        prob = self.match_fn(audio_emb, vq_emb)
        # prob: [bs, q_num, n_seg]
        clip_prob = self.pooling({
            "sim": prob, "audio_len": audio_output["length"]
        })
        output = {
            "frame_sim": prob,
            "clip_sim": clip_prob,
            "vq_loss": vq_loss,
            "label": label
        }
        return output


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
        return {
            "frame_sim": prob,
            "clip_sim": clip_prob,
            "length": output["length"]
        }


class AudioTaggingWithText(nn.Module):

    def __init__(self, audio_encoder, text_encoder, classes_num,
                 freeze_audio_encoder=False, freeze_text_encoder=False) -> None:
        super().__init__()
        self.backbone = audio_encoder
        self.text_encoder = text_encoder
        self.fc_output = nn.Linear(
            audio_encoder.embed_dim + text_encoder.embed_dim,
            classes_num)
        if freeze_audio_encoder:
            for param in self.backbone.parameters():
                param.requires_grad = False
        if freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

    def forward(self, input_dict):
        output = self.backbone(input_dict)
        text_emb = self.text_encoder(input_dict["text"])["sentence_emb"]
        audio_emb = output["embedding"]
        emb = torch.cat((audio_emb, text_emb.unsqueeze(1).expand_as(audio_emb)), dim=-1)
        logit = self.fc_output(emb)
        prob = torch.sigmoid(logit)
        clip_prob = linear_softmax_with_lens(prob, output["length"])
        return {
            "frame_sim": prob,
            "clip_sim": clip_prob
        }


class AudioTaggingWithTextCrossAttn(nn.Module):

    def __init__(self, audio_encoder, text_encoder, classes_num,
                 num_heads=8, dropout=0.2,
                 freeze_audio_encoder=False, freeze_text_encoder=False) -> None:
        super().__init__()
        self.backbone = audio_encoder
        self.text_encoder = text_encoder
        embed_dim = audio_encoder.embed_dim
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
            dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.fc_output = nn.Linear(embed_dim, classes_num)
        if freeze_audio_encoder:
            for param in self.backbone.parameters():
                param.requires_grad = False
        if freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

    def forward(self, input_dict):
        audio_output = self.backbone(input_dict)
        text_output = self.text_encoder(input_dict["text"])
        audio_emb = audio_output["embedding"]
        padding_mask = ~text_output["attention_mask"]
        text_emb = text_output["token_emb"]
        emb, attn = self.attn(audio_emb, text_emb, text_emb, key_padding_mask=padding_mask)
        emb = self.norm(audio_emb + self.dropout(emb))
        logit = self.fc_output(emb)
        prob = torch.sigmoid(logit)
        clip_prob = linear_softmax_with_lens(prob, audio_output["length"])
        return {
            "frame_sim": prob,
            "clip_sim": clip_prob
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


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.getcwd())
    from models.audio_encoder import CrnnEncoder
    from models.text_encoder import EmbeddingMeanEncoder
    from models.match import ExpNegL2
    vocab_size = 1500
    audio_encoder = CrnnEncoder(32000, 256)
    text_encoder = EmbeddingMeanEncoder(vocab_size, 256)
    match_fn = ExpNegL2()
    model = BiEncoder(audio_encoder, text_encoder, match_fn, 256, True)
    print(model)
    input_dict = {
        "waveform": torch.randn(4, 320000),
        "waveform_len": torch.tensor([320000] * 4),
        "text": torch.empty(4, 3).random_(vocab_size),
        "text_len": torch.tensor([2, 3, 3, 2])
    }
    output = model(input_dict)
    print("prob: ", output["prob"].shape, "length: ", output["length"])
