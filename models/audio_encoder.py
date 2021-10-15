from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import init_weights


class Block2D(nn.Module):
    def __init__(self, cin, cout, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(cin),
            nn.Conv2d(cin,
                      cout,
                      kernel_size=kernel_size,
                      padding=padding,
                      bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.1))

    def forward(self, x):
        return self.block(x)


class CrnnEncoder(nn.Module):
    
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.features = nn.Sequential(
            Block2D(1, 32),
            nn.LPPool2d(4, (2, 4)),
            Block2D(32, 128),
            Block2D(128, 128),
            nn.LPPool2d(4, (2, 4)),
            Block2D(128, 128),
            Block2D(128, 128),
            nn.LPPool2d(4, (1, 4)),
            nn.Dropout(0.3),
        )
        with torch.no_grad():
            rnn_input_dim = self.features(torch.randn(1, 1, 500,
                                                      input_dim)).shape
            rnn_input_dim = rnn_input_dim[1] * rnn_input_dim[-1]

        self.gru = nn.GRU(rnn_input_dim,
                          embed_dim // 2,
                          bidirectional=True,
                          batch_first=True)
        self.apply(init_weights)

    def forward(self, input_dict: Dict):
        audio_feat = input_dict["audio_feat"]
        audio_feat_len = input_dict["audio_feat_len"]
        upsample = input_dict.get("upsample", True)
        x = audio_feat
        batch, time, dim = x.shape
        x = x.unsqueeze(1)
        x = self.features(x)
        x = x.transpose(1, 2).contiguous().flatten(-2)
        x, _ = self.gru(x)
        if upsample:
            x = F.interpolate(
                x.transpose(1, 2),
                time,
                mode="linear",
                align_corners=False).transpose(1, 2)
            audio_len = audio_feat_len
        else:
            audio_len = audio_feat_len // 4
        return {
            "audio": x,
            "audio_len": audio_len
        }


def load_pretrained(model, path):
    trained_model = torch.load(path, map_location="cpu")
    model_dict = model.state_dict()
    # filter unnecessary keys
    pretrained_dict = {
        k: v
        for k, v in trained_model.items() if (k in model_dict) and (
            model_dict[k].shape == trained_model[k].shape)
    }
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=True)
    return model

if __name__ == "__main__":
    audio_feats = torch.randn(4, 501, 64)
    audio_feat_lens = torch.tensor([501] * 4)
    model = CrnnEncoder(64, 256)
    print(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    audio_feats = audio_feats.to(device)
    output = model({"audio_feat": audio_feats, "audio_feat_len": audio_feat_lens})
    print("audio: ", output["audio"].shape, output["audio_len"].shape)
    if torch.cuda.is_available():
        from pytorch_memlab import MemReporter
        rep = MemReporter()
        rep.report()
