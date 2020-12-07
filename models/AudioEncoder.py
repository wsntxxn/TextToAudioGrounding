from itertools import zip_longest
import torch.nn.functional as F
import math
import logging
import itertools
import numpy as np

import torch
import torch.nn as nn


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=dilation,
                     groups=groups,
                     bias=False,
                     dilation=dilation)


class StandardBlock(nn.Module):
    """docstring for StandardBlock"""
    def __init__(self,
                 inputfilter,
                 outputfilter,
                 kernel_size,
                 stride,
                 padding,
                 bn=True,
                 **kwargs):
        super(StandardBlock, self).__init__()
        self.activation = kwargs.get('activation', nn.ReLU(True))
        self.batchnorm = nn.Sequential() if not bn else nn.BatchNorm2d(
            inputfilter)
        if self.activation.__class__.__name__ == 'GLU':
            outputfilter = outputfilter * 2
        self.conv = nn.Conv2d(inputfilter,
                              outputfilter,
                              kernel_size=kernel_size,
                              stride=stride,
                              bias=not bn,
                              padding=padding)

    def forward(self, x):
        x = self.batchnorm(x)
        x = self.conv(x)
        return self.activation(x)


class ChannelAttnBlock(nn.Module):
    """docstring for"""
    def __init__(self,
                 inputfilter,
                 outputfilter,
                 kernel_size,
                 stride,
                 padding,
                 bn=True,
                 **kwargs):
        super(ChannelAttnBlock, self).__init__()
        self.activation = kwargs.get('activation', nn.ReLU(True))
        self.batchnorm = nn.Sequential() if not bn else nn.BatchNorm2d(
            inputfilter)
        if self.activation.__class__.__name__ == 'GLU':
            outputfilter = outputfilter * 2
        self.conv = nn.Conv2d(inputfilter,
                              outputfilter,
                              kernel_size=kernel_size,
                              stride=stride,
                              bias=not bn,
                              padding=padding)
        self.attn = ChannelAttention(poolingmethod='avg')

    def forward(self, x):
        x = self.batchnorm(x)
        x = self.conv(x)
        x = self.activation(x)
        x = self.attn(x)
        return x


class FreqAttentionBlock(nn.Module):
    """docstring for StandardBlock"""
    def __init__(self,
                 inputfilter,
                 outputfilter,
                 kernel_size,
                 stride,
                 padding,
                 bn=True,
                 **kwargs):
        super(FreqAttentionBlock, self).__init__()
        self.activation = kwargs.get('activation', nn.ReLU(True))
        self.batchnorm = nn.Sequential() if not bn else nn.BatchNorm2d(
            inputfilter)
        if self.activation.__class__.__name__ == 'GLU':
            outputfilter = outputfilter * 2
        self.conv = nn.Conv2d(inputfilter,
                              outputfilter,
                              kernel_size=kernel_size,
                              stride=stride,
                              bias=not bn,
                              padding=padding)
        self.attn = DimAttention(input_filters=inputfilter, dim=-1)

    def forward(self, x):
        x = self.attn(x)
        x = self.batchnorm(x)
        x = self.conv(x)
        x = self.activation(x)
        return x


class MaxPool(nn.Module):
    def __init__(self, pooldim=1):
        super().__init__()
        self.pooldim = pooldim

    def forward(self, logits, decision):
        return torch.max(decision, dim=self.pooldim)[0]

class PowerPool(nn.Module):
    """LinearSoftPool

    Linear softmax, takes logits and returns a probability, near to the actual maximum value.
    Taken from the paper:

        A Comparison of Five Multiple Instance Learning Pooling Functions for Sound Event Detection with Weak Labeling
    https://arxiv.org/abs/1810.09050

    """
    def __init__(self, inputdim=None, outputdim=10, pooldim=1):
        super().__init__()
        self.pooldim = pooldim
        self.weights = nn.Parameter(torch.full((outputdim, ), 1.2))

    def forward(self, logits, time_decision):
        scaled = time_decision**self.weights
        return (time_decision*scaled).sum(self.pooldim) / scaled.sum(
            self.pooldim)

class LinearSoftPool(nn.Module):
    """LinearSoftPool

    Linear softmax, takes logits and returns a probability, near to the actual maximum value.
    Taken from the paper:

        A Comparison of Five Multiple Instance Learning Pooling Functions for Sound Event Detection with Weak Labeling
    https://arxiv.org/abs/1810.09050

    """
    def __init__(self, pooldim=1):
        super().__init__()
        self.pooldim = pooldim

    def forward(self, logits, time_decision):
        return (time_decision**2).sum(self.pooldim) / time_decision.sum(
            self.pooldim)


class MeanPool(nn.Module):
    def __init__(self, pooldim=1):
        super().__init__()
        self.pooldim = pooldim

    def forward(self, logits, decision):
        return torch.mean(decision, dim=self.pooldim)


class AutoExpPool(nn.Module):
    def __init__(self, outputdim=10, pooldim=1):
        super().__init__()
        self.outputdim = outputdim
        self.alpha = nn.Parameter(torch.full((outputdim, ), 1))
        self.pooldim = pooldim

    def forward(self, logits, decision):
        scaled = self.alpha * decision  # \alpha * P(Y|x) in the paper
        return (logits * torch.exp(scaled)).sum(
            self.pooldim) / torch.exp(scaled).sum(self.pooldim)


class SoftPool(nn.Module):
    def __init__(self, T=1, pooldim=1):
        super().__init__()
        self.pooldim = pooldim
        self.T = T

    def forward(self, logits, decision):
        w = torch.softmax(decision / self.T, dim=self.pooldim)
        return torch.sum(decision * w, dim=self.pooldim)


class AutoPool(nn.Module):
    """docstring for AutoPool"""
    def __init__(self, outputdim=10, pooldim=1):
        super().__init__()
        self.outputdim = outputdim
        self.alpha = nn.Parameter(torch.ones(outputdim))
        self.dim = pooldim

    def forward(self, logits, decision):
        scaled = self.alpha * decision  # \alpha * P(Y|x) in the paper
        weight = torch.softmax(scaled, dim=self.dim)
        return torch.sum(decision * weight, dim=self.dim)  # B x C


class AttentionPool(nn.Module):
    """docstring for AttentionPool"""
    def __init__(self, inputdim, outputdim=10, pooldim=1, **kwargs):
        super().__init__()
        self.inputdim = inputdim
        self.outputdim = outputdim
        self.pooldim = pooldim
        self.transform = nn.Linear(inputdim, outputdim)
        self.activ = nn.Softmax(dim=self.pooldim)
        self.eps = 1e-7

    def forward(self, logits, decision):
        # Input is (B, T, D)
        # B, T , D
        w = self.activ(self.transform(logits))
        detect = (decision * w).sum(
            self.pooldim) / (w.sum(self.pooldim) + self.eps)
        # B, T, D
        return detect


class AddContainer(nn.Module):
    """docstring for ChainContainer"""
    def __init__(self, *chain):
        super(AddContainer, self).__init__()
        self._containers = chain

    def forward(self, x):
        r = [cont(x) for cont in self._containers]
        return torch.stack(r).sum(0)


class MulContainer(nn.Module):
    """docstring for ChainContainer"""
    def __init__(self, *chain):
        super(MulContainer, self).__init__()
        self._containers = chain

    def forward(self, x):
        r = [cont(x) for cont in self._containers]
        return torch.stack(r).prod(0)


class ChannelAttention(nn.Module):
    """docstring for ChannelAttention"""
    def __init__(self, **kwargs):
        super(ChannelAttention, self).__init__()
        poolingmethod = kwargs.get('poolingmethod', 'avg')

        if poolingmethod == 'avg':
            self.pooling = nn.AdaptiveAvgPool2d(1)
        elif poolingmethod == 'max':
            self.pooling = nn.AdaptiveMaxPool2d(1)
        elif poolingmethod == 'avgmax':
            self.pooling = AddContainer(nn.AdaptiveAvgPool2d(1),
                                        nn.AdaptiveMaxPool2d(1))
        else:
            raise ValueError(
                "Poolingmethod ({}) not found".format(poolingmethod))

    def forward(self, x):
        """forward

        :param x: Shape of [BxCxWxH]
        """
        # hid_trans = torch.tanh(self.in_to_hid(self.pooling(x).squeeze(-1).squeeze(-1)))
        weights = torch.softmax(torch.tanh(
            self.pooling(x).squeeze(-1).squeeze(-1)),
                                dim=1)
        # pooled = torch.sigmoid(self.pooling(x))  # [BxCx1x1]
        # weights = pooled / torch.sum(pooled, dim=1)[:, :, None]
        # return weights * x
        return weights[:, :, None, None] * x


class DimAttention(nn.Module):
    """docstring for TFAttention"""
    def __init__(self, input_filters, dim=-1, **kwargs):
        super(DimAttention, self).__init__()
        self.conv = nn.Conv2d(input_filters, 1, kernel_size=1)
        self.activ = nn.Sigmoid()
        self._dim = dim

    def forward(self, x):
        y = self.conv(x)
        weights = self.activ(y)  # B x 1 x T x F
        return weights * x


def parse_poolingfunction(poolingfunction_name='mean', **kwargs):
    """parse_poolingfunction
    A heler function to parse any temporal pooling
    Pooling is done on dimension 1

    :param poolingfunction_name:
    :param **kwargs:
    """
    poolingfunction_name = poolingfunction_name.lower()
    if poolingfunction_name == 'mean':
        return MeanPool(pooldim=1)
    elif poolingfunction_name == 'max':
        return MaxPool(pooldim=1)
    elif poolingfunction_name == 'linear':
        return LinearSoftPool(pooldim=1)
    elif poolingfunction_name == 'expalpha':
        return AutoExpPool(outputdim=kwargs['outputdim'], pooldim=1)

    elif poolingfunction_name == 'soft':
        return SoftPool(pooldim=1)
    elif poolingfunction_name == 'auto':
        return AutoPool(outputdim=kwargs['outputdim'])
    elif poolingfunction_name == 'power':
        return PowerPool(outputdim = kwargs['outputdim'],pooldim=1)
    elif poolingfunction_name == 'attention':
        return AttentionPool(inputdim=kwargs['inputdim'],
                             outputdim=kwargs['outputdim'])


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


class Attention(nn.Module):
    """docstring for Att"""
    def __init__(self, hidden_dim, return_attention=False):
        super().__init__()
        self.attn = nn.Linear(2 * hidden_dim, hidden_dim, bias=False)
        self.return_attention = return_attention

    def forward(self, query, context):
        scores = query @ context.transpose(1, 2).contiguous()
        attn_weights = torch.softmax(scores, dim=-1)
        mix = attn_weights @ context
        combined = torch.cat((mix, query), dim=-1)
        output = torch.tanh(self.attn(combined))
        if self.return_attention:
            return output, attn_weights
        else:
            return output


class CRNNEncoder(nn.Module):
    def __init__(self, inputdim, embeddim, **kwargs):
        super(CRNNEncoder, self).__init__()
        features = nn.ModuleList()
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
                                                      inputdim)).shape
            rnn_input_dim = rnn_input_dim[1] * rnn_input_dim[-1]

        self.gru = nn.GRU(rnn_input_dim,
                          embeddim // 2,
                          bidirectional=True,
                          batch_first=True)
        # self.temp_pool = parse_poolingfunction(kwargs.get('temppool', 'linear'),
                                               # inputdim=embeddim,
                                               # outputdim=outputdim)
        self.features.apply(init_weights)

    def forward(self, x, upsample=True):
        batch, time, dim = x.shape
        x = x.unsqueeze(1)
        x = self.features(x)
        x = x.transpose(1, 2).contiguous().flatten(-2)
        x, _ = self.gru(x)
        # decision_time = torch.sigmoid(self.outputlayer(x)).clamp(1e-7, 1.)
        # decision = self.temp_pool(x, decision_time).clamp(1e-7, 1.).squeeze(1)
        if upsample:
            x = torch.nn.functional.interpolate(
                x.transpose(1, 2),
                time,
                mode='linear',
                align_corners=False).transpose(1, 2)
        return x


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Conv1d)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def load_pretrained(model, path, feature_extract=False):
    trained_model = torch.load(path, map_location='cpu')
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
    from pytorch_model_summary import summary
    from pytorch_memlab import MemReporter, profile
    x = torch.randn(64, 501, 64)
    model = CRNNEncoder(64, 256)
    print(model)

    print(summary(model, x, show_input=True))
    # device = torch.device('cuda')
    device = torch.device('cpu')
    model = model.to(device)
    x = x.to(device)
    y = model(x)
    rep = MemReporter()
    rep.report()
