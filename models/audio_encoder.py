from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio import transforms
from torchlibrosa import SpecAugmentation

from utils.train_util import do_mixup
from models.utils import init_weights
from models.panns import ConvBlock, init_layer, init_bn


def cdur_block(cin, cout, kernel_size=3, padding=1):
    return nn.Sequential(
        nn.BatchNorm2d(cin),
        nn.Conv2d(cin,
                  cout,
                  kernel_size=kernel_size,
                  padding=padding,
                  bias=False),
        nn.LeakyReLU(inplace=True, negative_slope=0.1)
    )


class CrnnEncoder(nn.Module):
    
    def __init__(self, sample_rate, embed_dim):
        super().__init__()
        self.downsample_ratio = 4
        self.melspec_extractor = transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=2048,
            win_length=40 * sample_rate // 1000,
            hop_length=20 * sample_rate // 1000,
            n_mels=64
        )
        self.hop_length = 20 * sample_rate // 1000
        self.db_transform = transforms.AmplitudeToDB()
        self.embed_dim = embed_dim
        self.cnn = nn.Sequential(
            cdur_block(1, 32),
            nn.LPPool2d(4, (2, 4)),
            cdur_block(32, 128),
            cdur_block(128, 128),
            nn.LPPool2d(4, (2, 4)),
            cdur_block(128, 128),
            cdur_block(128, 128),
            nn.LPPool2d(4, (1, 4)),
            nn.Dropout(0.3),
        )
        with torch.no_grad():
            dummy_input = self.db_transform(self.melspec_extractor(
                torch.randn(1, 16000)))
            dummy_input = dummy_input.transpose(1, 2).unsqueeze(1)
            rnn_input_dim = self.cnn(dummy_input).shape
            rnn_input_dim = rnn_input_dim[1] * rnn_input_dim[-1]

        self.gru = nn.GRU(rnn_input_dim,
                          embed_dim // 2,
                          bidirectional=True,
                          batch_first=True)
        self.apply(init_weights)

    def forward(self, input_dict: Dict):
        waveform = input_dict["waveform"]
        mel_spec = self.melspec_extractor(waveform)
        lms = self.db_transform(mel_spec)

        x = lms.transpose(1, 2)
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = x.transpose(1, 2).contiguous().flatten(-2)
        x, _ = self.gru(x)

        length = torch.div(
            torch.as_tensor(input_dict["waveform_len"]),
            self.hop_length,
            rounding_mode="floor") + 1
        length = torch.div(length, self.downsample_ratio, rounding_mode="floor")

        return {
            "embedding": x,
            "length": length
        }


class Cnn8_Rnn(nn.Module):

    def __init__(self, sample_rate, freeze_cnn=False, freeze_bn=False):
        
        super(Cnn8_Rnn, self).__init__()

        self.downsample_ratio = 4     # Downsampled ratio
        self.time_resolution = 0.04
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

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.fc1 = nn.Linear(512, 512, bias=True)
        self.rnn = nn.GRU(512, 256, bidirectional=True, batch_first=True)
        self.embed_dim = 512
        
        self.init_weight()

        if self.freeze_cnn:
            for param in self.parameters():
                param.requires_grad = False
            for param in self.rnn.parameters():
                param.requires_grad = True

    def train(self, mode):
        super().train(mode=mode)
        if self.freeze_bn:
            def bn_eval(module):
                class_name = module.__class__.__name__
                if class_name.find("BatchNorm") != -1:
                    module.eval()
            self.apply(bn_eval)
        return self

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
 
    def forward(self, input_dict: Dict):
        """
        Input: (batch_size, n_samples)"""

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

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg+max')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg+max')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(1, 2), pool_type='avg+max')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(1, 2), pool_type='avg+max')
        x = F.dropout(x, p=0.2, training=self.training) # (batch_size, 256, time_steps / 4, mel_bins / 16)
        x = torch.mean(x, dim=3)
        
        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        x, _  = self.rnn(x)

        length = torch.div(
            torch.as_tensor(input_dict["waveform_len"]),
            self.hop_length,
            rounding_mode="floor") + 1

        length = torch.div(length, self.downsample_ratio, rounding_mode="floor")

        if self.training and mixup_lambda is not None:
            length = do_mixup(length, mixup_lambda)
    
        return {
            "embedding": x,
            "length": length
        }


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
