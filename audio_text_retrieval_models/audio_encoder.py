# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import wget
import timm
from timm.models.layers import to_2tuple, trunc_normal_
from torchaudio import transforms
from torchlibrosa.augmentation import SpecAugmentation
from audio_text_retrieval_models.utils import mean_with_lens, max_with_lens


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ConvBlock, self).__init__()
        
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

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

        
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


class Cnn14(nn.Module):
    def __init__(self, sample_rate=32000, pretrained=None):
        
        super(Cnn14, self).__init__()

        sr_to_fmax = {
            32000: 14000,
            16000: 8000
        }

        self.sample_rate = sample_rate
    
        # Logmel spectrogram extractor
        self.melspec_extractor = transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=32 * sample_rate // 1000,
            win_length=32 * sample_rate // 1000,
            hop_length=10 * sample_rate // 1000,
            f_min=50,
            f_max=sr_to_fmax[sample_rate],
            n_mels=64,
            norm="slaney",
            mel_scale="slaney"
        )
        self.hop_length = 10 * sample_rate // 1000
        self.db_transform = transforms.AmplitudeToDB()
        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
            freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.downsample_ratio = 32

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        
        if pretrained:
            state_dict = torch.load(pretrained, map_location="cpu")["model"]
            model_dict = self.state_dict()
            pretrained_dict = {
                k: v for k, v in state_dict.items() if (k in model_dict) and (
                    model_dict[k].shape == v.shape)
            }
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict, strict=True)
        else:
            self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
 
    def forward(self, waveform, wave_length, specaug=False):
        """
        Input: (batch_size, n_samples)"""
        x = self.melspec_extractor(waveform)
        x = self.db_transform(x)    # (batch_size, mel_bins, time_steps)
        x = x.transpose(1, 2)
        x = x.unsqueeze(1)      # (batch_size, 1, time_steps, mel_bins)

        # SpecAugment
        if self.training and specaug:
            x = self.spec_augmenter(x)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        time_emb = x.transpose(1, 2)
        
        feat_length = torch.div(wave_length, self.hop_length,
            rounding_mode="floor") + 1
        feat_length = torch.div(feat_length, self.downsample_ratio,
            rounding_mode="floor")
        x_max = max_with_lens(time_emb, feat_length)
        x_mean = mean_with_lens(time_emb, feat_length)
        x = x_max + x_mean
        # (x1, _) = torch.max(x, dim=2)
        # x2 = torch.mean(x, dim=2)
        # x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        clip_emb = F.dropout(x, p=0.5, training=self.training)
        
        output_dict = {
            'clip_emb': clip_emb,
            'time_emb': time_emb
        }

        return output_dict


class ConvPreWavBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ConvPreWavBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=3, stride=1,
                              padding=1, bias=False)
                              
        self.conv2 = nn.Conv1d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=3, stride=1, dilation=2, 
                              padding=2, bias=False)
                              
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

        
    def forward(self, input, pool_size):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, kernel_size=pool_size)
        
        return x


class Wavegram_Logmel_Cnn14(nn.Module):
    def __init__(self, sample_rate=32000, pretrained=None):
        
        super(Wavegram_Logmel_Cnn14, self).__init__()

        assert sample_rate == 32000

        # Logmel spectrogram extractor
        self.melspec_extractor = transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=32 * sample_rate // 1000,
            win_length=32 * sample_rate // 1000,
            hop_length=10 * sample_rate // 1000,
            f_min=50,
            f_max=14000,
            n_mels=64,
            norm="slaney",
            mel_scale="slaney"
        )
        self.hop_length = 10 * sample_rate // 1000
        self.db_transform = transforms.AmplitudeToDB()
        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
            freq_drop_width=8, freq_stripes_num=2)

        self.pre_conv0 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=11, stride=5, padding=5, bias=False)
        self.pre_bn0 = nn.BatchNorm1d(64)
        self.pre_block1 = ConvPreWavBlock(64, 64)
        self.pre_block2 = ConvPreWavBlock(64, 128)
        self.pre_block3 = ConvPreWavBlock(128, 128)
        self.pre_block4 = ConvBlock(in_channels=4, out_channels=64)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=128, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.downsample_ratio = 32

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        
        if pretrained:
            state_dict = torch.load(pretrained, map_location="cpu")["model"]
            model_dict = self.state_dict()
            pretrained_dict = {
                k: v for k, v in state_dict.items() if (k in model_dict) and (
                    model_dict[k].shape == v.shape)
            }
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict, strict=True)
        else:
            self.init_weight()

    def init_weight(self):
        init_layer(self.pre_conv0)
        init_bn(self.pre_bn0)
        init_bn(self.bn0)
        init_layer(self.fc1)
 
    def forward(self, waveform, wave_length, specaug=False):
        """
        Input: (batch_size, data_length)"""

        # Wavegram
        a1 = F.relu_(self.pre_bn0(self.pre_conv0(waveform[:, None, :])))
        a1 = self.pre_block1(a1, pool_size=4)
        a1 = self.pre_block2(a1, pool_size=4)
        a1 = self.pre_block3(a1, pool_size=4)
        a1 = a1.reshape((a1.shape[0], -1, 32, a1.shape[-1])).transpose(2, 3)
        a1 = self.pre_block4(a1, pool_size=(2, 1))

        # Log mel spectrogram
        x = self.melspec_extractor(waveform)
        x = self.db_transform(x)    # (batch_size, mel_bins, time_steps)
        x = x.transpose(1, 2)
        x = x.unsqueeze(1)      # (batch_size, 1, time_steps, mel_bins)

        if self.training and specaug:
            x = self.spec_augmenter(x)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')

        # Concatenate Wavegram and Log mel spectrogram along the channel dimension
        if x.size(2) > a1.size(2):
            x = x[:, :, :a1.size(2), :]
        elif a1.size(2) > x.size(2):
            a1 = a1[:, :, :x.size(2), :]
        x = torch.cat((x, a1), dim=1)

        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        time_emb = x.transpose(1, 2)
        
        feat_length = torch.div(wave_length, self.hop_length,
            rounding_mode="floor") + 1
        feat_length = torch.div(feat_length, self.downsample_ratio,
            rounding_mode="floor")
        x_max = max_with_lens(time_emb, feat_length)
        x_mean = mean_with_lens(time_emb, feat_length)
        x = x_max + x_mean
        # (x1, _) = torch.max(x, dim=2)
        # x2 = torch.mean(x, dim=2)
        # x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        clip_emb = F.dropout(x, p=0.5, training=self.training)
        
        output_dict = {
            'clip_emb': clip_emb,
            'time_emb': time_emb
        }

        return output_dict


def _resnet_conv3x3(in_planes, out_planes):
    #3x3 convolution with padding
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, groups=1, bias=False, dilation=1)


def _resnet_conv1x1(in_planes, out_planes):
    #1x1 convolution
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)


class _ResnetBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(_ResnetBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('_ResnetBasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in _ResnetBasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        self.stride = stride

        self.conv1 = _resnet_conv3x3(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _resnet_conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_bn(self.bn1)
        init_layer(self.conv2)
        init_bn(self.bn2)
        nn.init.constant_(self.bn2.weight, 0)

    def forward(self, x):
        identity = x

        if self.stride == 2:
            out = F.avg_pool2d(x, kernel_size=(2, 2))
        else:
            out = x

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = F.dropout(out, p=0.1, training=self.training)

        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out


class _ResnetBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(_ResnetBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.stride = stride
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = _resnet_conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = _resnet_conv3x3(width, width)
        self.bn2 = norm_layer(width)
        self.conv3 = _resnet_conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_bn(self.bn1)
        init_layer(self.conv2)
        init_bn(self.bn2)
        init_layer(self.conv3)
        init_bn(self.bn3)
        nn.init.constant_(self.bn3.weight, 0)

    def forward(self, x):
        identity = x

        if self.stride == 2:
            x = F.avg_pool2d(x, kernel_size=(2, 2))

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = F.dropout(out, p=0.1, training=self.training)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out


class _ResNet(nn.Module):
    def __init__(self, block, layers, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(_ResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if stride == 1:
                downsample = nn.Sequential(
                    _resnet_conv1x1(self.inplanes, planes * block.expansion),
                    norm_layer(planes * block.expansion),
                )
                init_layer(downsample[0])
                init_bn(downsample[1])
            elif stride == 2:
                downsample = nn.Sequential(
                    nn.AvgPool2d(kernel_size=2), 
                    _resnet_conv1x1(self.inplanes, planes * block.expansion),
                    norm_layer(planes * block.expansion),
                )
                init_layer(downsample[1])
                init_bn(downsample[2])

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class ResNet38(nn.Module):
    def __init__(self, sample_rate=32000, pretrained=None):
        
        super(ResNet38, self).__init__()

        # Logmel spectrogram extractor
        self.melspec_extractor = transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=32 * sample_rate // 1000,
            win_length=32 * sample_rate // 1000,
            hop_length=10 * sample_rate // 1000,
            f_min=50,
            f_max=14000,
            n_mels=64,
            norm="slaney",
            mel_scale="slaney"
        )
        self.hop_length = 10 * sample_rate // 1000
        self.db_transform = transforms.AmplitudeToDB()
        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64,
            time_stripes_num=2, freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        # self.conv_block2 = ConvBlock(in_channels=64, out_channels=64)

        self.resnet = _ResNet(block=_ResnetBasicBlock, layers=[3, 4, 6, 3], zero_init_residual=True)

        self.conv_block_after1 = ConvBlock(in_channels=512, out_channels=2048)

        self.downsample_ratio = 32

        self.fc1 = nn.Linear(2048, 2048)

        if pretrained:
            state_dict = torch.load(pretrained, map_location="cpu")["model"]
            model_dict = self.state_dict()
            pretrained_dict = {
                k: v for k, v in state_dict.items() if (k in model_dict) and (
                    model_dict[k].shape == v.shape)
            }
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict, strict=True)
        else:
            self.init_weights()

    def init_weights(self):
        init_bn(self.bn0)
        init_layer(self.fc1)


    def forward(self, waveform, wave_length, specaug=False):
        """
        Input: (batch_size, data_length)"""

        # Log mel spectrogram
        x = self.melspec_extractor(waveform)
        x = self.db_transform(x)    # (batch_size, mel_bins, time_steps)
        x = x.transpose(1, 2)
        x = x.unsqueeze(1)      # (batch_size, 1, time_steps, mel_bins)

        if self.training and specaug:
            x = self.spec_augmenter(x)
        
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = self.resnet(x)
        x = F.avg_pool2d(x, kernel_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = self.conv_block_after1(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = torch.mean(x, dim=3)
        time_emb = x.transpose(1, 2)
        
        feat_length = torch.div(wave_length, self.hop_length,
            rounding_mode="floor") + 1
        feat_length = torch.div(feat_length, self.downsample_ratio,
            rounding_mode="floor")
        x_max = max_with_lens(time_emb, feat_length)
        x_mean = mean_with_lens(time_emb, feat_length)
        x = x_max + x_mean

        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        clip_emb = F.dropout(x, p=0.5, training=self.training)
        
        output_dict = {
            'clip_emb': clip_emb,
            'time_emb': time_emb
        }

        return output_dict


# override the timm package to relax the input shape constraint.
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class AudioSpectrogramTransformer(nn.Module):
    """
    The AST model.
    :param fstride: the stride of patch spliting on the frequency dimension, for 16*16 patchs, fstride=16 means no overlap, fstride=10 means overlap of 6
    :param tstride: the stride of patch spliting on the time dimension, for 16*16 patchs, tstride=16 means no overlap, tstride=10 means overlap of 6
    :param input_fdim: the number of frequency bins of the input spectrogram
    :param input_tdim: the number of time frames of the input spectrogram
    :param imagenet_pretrain: if use ImageNet pretrained model
    :param audioset_pretrain: if use full AudioSet and ImageNet pretrained model
    :param model_size: the model size of AST, should be in [tiny224, small224, base224, base384], base224 and base 384 are same model, but are trained differently during ImageNet pretraining.
    """
    def __init__(self, 
                 sample_rate=32000,
                 fstride=10,
                 tstride=10,
                 input_fdim=128,
                 input_tdim=1024,
                 init_norm=True,
                 imagenet_pretrain=True,
                 audioset_pretrain=False,
                 pretrained=None,
                 model_size="base384",
                 verbose=True):

        super().__init__()
        self.sample_rate = sample_rate
        self.input_tdim = input_tdim
        # Logmel spectrogram extractor
        self.melspec_extractor = transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=32 * sample_rate // 1000,
            win_length=32 * sample_rate // 1000,
            hop_length=10 * sample_rate // 1000,
            f_min=50,
            f_max=14000,
            n_mels=input_fdim,
            norm="slaney",
            mel_scale="slaney"
        )
        self.db_transform = transforms.AmplitudeToDB()
        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64,
            time_stripes_num=2, freq_drop_width=8, freq_stripes_num=2)
        # self.stats = [-4.2677393, 4.5689974]
        # self.std_norm = std_norm
        self.init_norm = init_norm
        if init_norm:
            self.init_bn = nn.BatchNorm2d(input_fdim)
            init_bn(self.init_bn)
        assert timm.__version__ == '0.4.5', 'Please use timm == 0.4.5, the code might not be compatible with newer versions.'

        if verbose == True:
            print('---------------AST Model Summary---------------')
            print('ImageNet pretraining: {:s}, AudioSet pretraining: {:s}'.format(str(imagenet_pretrain),str(audioset_pretrain)))
        # override timm input shape restriction
        timm.models.vision_transformer.PatchEmbed = PatchEmbed

        # if AudioSet pretraining is not used (but ImageNet pretraining may still apply)
        if audioset_pretrain == False:
            if model_size == 'tiny224':
                self.v = timm.create_model('vit_deit_tiny_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'small224':
                self.v = timm.create_model('vit_deit_small_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'base224':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'base384':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=imagenet_pretrain)
            else:
                raise Exception('Model size must be one of tiny224, small224, base224, base384.')
            self.original_num_patches = self.v.patch_embed.num_patches
            self.original_hw = int(self.original_num_patches ** 0.5)
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            # self.mlp_head = nn.Sequential(
                # nn.LayerNorm(self.original_embedding_dim),
                # nn.Linear(self.original_embedding_dim, label_dim)
            # )

            # automatcially get the intermediate shape
            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if verbose == True:
                print('frequency stride={:d}, time stride={:d}'.format(fstride, tstride))
                print('number of patches={:d}'.format(num_patches))

            # the linear projection layer
            new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
            if imagenet_pretrain == True:
                new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
                new_proj.bias = self.v.patch_embed.proj.bias
            self.v.patch_embed.proj = new_proj

            # the positional embedding
            if imagenet_pretrain == True:
                # get the positional embedding from deit model, skip the first two tokens (cls token and distillation token), reshape it to original 2D shape (24*24).
                new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(
                    1, self.original_num_patches, self.original_embedding_dim).transpose(
                        1, 2).reshape(1, self.original_embedding_dim, self.original_hw, self.original_hw)
                # cut (from middle) or interpolate the second dimension of the positional embedding
                if t_dim <= self.original_hw:
                    new_pos_embed = new_pos_embed[:, :, :, int(self.original_hw / 2) - int(t_dim / 2): int(self.original_hw / 2) - int(t_dim / 2) + t_dim]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(self.original_hw, t_dim), mode='bilinear')
                # cut (from middle) or interpolate the first dimension of the positional embedding
                if f_dim <= self.original_hw:
                    new_pos_embed = new_pos_embed[:, :, int(self.original_hw / 2) - int(f_dim / 2): int(self.original_hw / 2) - int(f_dim / 2) + f_dim, :]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
                # flatten the positional embedding
                new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1,2)
                # concatenate the above positional embedding with the cls token and distillation token of the deit model.
                self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))
            else:
                # if not use imagenet pretrained model, just randomly initialize a learnable positional embedding
                # TODO can use sinusoidal positional embedding instead
                new_pos_embed = nn.Parameter(torch.zeros(1, self.v.patch_embed.num_patches + 2, self.original_embedding_dim))
                self.v.pos_embed = new_pos_embed
                trunc_normal_(self.v.pos_embed, std=.02)

        # now load a model that is pretrained on both ImageNet and AudioSet
        elif audioset_pretrain == True:
            if audioset_pretrain == True and imagenet_pretrain == False:
                raise ValueError('currently model pretrained on only audioset is not supported, please set imagenet_pretrain = True to use audioset pretrained model.')
            if model_size != 'base384':
                raise ValueError('currently only has base384 AudioSet pretrained model.')
            assert pretrained is not None
            # this model performs 0.4593 mAP on the audioset eval set
            # audioset_mdl_url = 'https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1'
            # wget.download(audioset_mdl_url, out='./experiments/pretrained_weights/ast_4593.pth')
            sd = torch.load(pretrained, map_location="cpu")
            # audio_model = AudioSpectrogramTransformer(fstride=10,
                # tstride=10, input_fdim=128, input_tdim=1024, imagenet_pretrain=False,
                # audioset_pretrain=False, model_size='base384', verbose=False)
            audio_model = AudioSpectrogramTransformer(fstride=fstride,
                tstride=tstride, input_fdim=128, input_tdim=1024, imagenet_pretrain=False,
                audioset_pretrain=False, model_size='base384', verbose=False)
            audio_model = torch.nn.DataParallel(audio_model)

            model_dict = audio_model.state_dict()
            pretrained_dict = {
                k: v for k, v in sd.items() if (k in model_dict) and (
                    model_dict[k].shape == v.shape)
            }
            model_dict.update(pretrained_dict)
            audio_model.load_state_dict(model_dict, strict=True)

            self.v = audio_model.module.v
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            # self.mlp_head = nn.Sequential(
                # nn.LayerNorm(self.original_embedding_dim),
                # nn.Linear(self.original_embedding_dim, label_dim)
            # )

            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if verbose:
                print('frequncey stride={:d}, time stride={:d}'.format(fstride, tstride))
                print('number of patches={:d}'.format(num_patches))

            if tstride == 10 and fstride == 10:
                new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, 1212, 768).transpose(1, 2).reshape(1, 768, 12, 101)
                # if the input sequence length is larger than the original audioset (10s), then cut the positional embedding
                if t_dim < 101:
                    new_pos_embed = new_pos_embed[:, :, :, 50 - int(t_dim/2): 50 - int(t_dim/2) + t_dim]
                # otherwise interpolate
                else:
                    new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(12, t_dim), mode='bilinear')
                new_pos_embed = new_pos_embed.reshape(1, 768, t_dim * 12).transpose(1, 2)
                self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))

        if hasattr(self.v, "head"):
            del self.v.head
        if hasattr(self.v, "head_dist"):
            del self.v.head_dist
        if hasattr(self.v, "pre_logits"):
            del self.v.pre_logits

    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    def forward(self, waveform, wave_length, specaug=False):
        """
        :param input: (batch_size, data_length)
        """
        x = self.melspec_extractor(waveform)
        x = self.db_transform(x)
        x = x.transpose(1, 2)
        x = x.unsqueeze(1)

        if self.training and specaug:
            x = self.spec_augmenter(x)

        if self.init_norm:
            x = x.transpose(1, 3)
            x = self.init_bn(x)
            x = x.transpose(1, 3)

        n_frames = x.shape[2]
        p = self.input_tdim - n_frames
        if p < 0:
            x = x[:, :, 0: self.input_tdim, :]

        x = x.transpose(2, 3)

        B = x.shape[0]
        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed[:, :x.size(1), :]
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)

        output_dict = {
            "clip_emb": (x[:, 0] + x[:, 1]) / 2,
            "time_emb": x
        }

        return output_dict


if __name__ == '__main__':
    model = AudioSpectrogramTransformer(input_tdim=1024, input_fdim=128, audioset_pretrain=True,
                                        pretrained="./experiments/pretrained_weights/ast_0.4593.pth")
    print("Audio Spectrogram Transformer")
    num_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f"Trainable parameters: {num_params}")
    test_input = torch.randn([4, 320000])
    test_output = model(test_input)
    print("clip: ", test_output["clip_emb"].shape)
    print("time: ", test_output["time_emb"].shape)

    model = Wavegram_Logmel_Cnn14()
    print("Wavegram Logmel Cnn14")
    num_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f"Trainable parameters: {num_params}")
    test_input = torch.randn([4, 320000])
    test_output = model(test_input)
    print("clip: ", test_output["clip_emb"].shape)
    print("time: ", test_output["time_emb"].shape)

    model = Cnn14()
    print("Cnn14")
    num_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f"Trainable parameters: {num_params}")
    test_input = torch.randn([4, 320000])
    test_output = model(test_input)
    print("clip: ", test_output["clip_emb"].shape)
    print("time: ", test_output["time_emb"].shape)
