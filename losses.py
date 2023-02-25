from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import generate_length_mask


class FrameBceLoss(nn.Module):

    def forward(self, output: Dict):
        frame_sim = output["frame_sim"] # [batch_size, max_len]
        if frame_sim.ndim == 3 and frame_sim.size(2) == 1:
            frame_sim = frame_sim.squeeze(2)
        length = output["length"] # [N]
        label = output["label"] # [N, T]
        loss = F.binary_cross_entropy(frame_sim, label, reduction="none") # [N, T]
        mask = generate_length_mask(length).to(frame_sim.device)
        loss *= mask
        loss = loss.sum() / mask.sum()
        return loss
    
    def forward_tensor(self, frame_sim, label, length):
        loss = F.binary_cross_entropy(frame_sim, label, reduction="none") # [N, T]
        mask = generate_length_mask(length).to(frame_sim.device)
        if loss.ndim == 3:
            mask = mask.unsqueeze(-1).expand(*loss.size())
        loss *= mask
        loss = loss.sum() / mask.sum()
        return loss


class ClipBceLoss(nn.Module):

    def forward(self, output: Dict):
        return F.binary_cross_entropy(output["clip_sim"], output["label"])

    def forward_tensor(self, prob, label):
        return F.binary_cross_entropy(prob, label)


class MaskedClipBceLoss(nn.Module):

    def forward(self, output: Dict):

        loss = F.binary_cross_entropy(output["clip_sim"], output["label"], reduce="none")
        cls_mask = output["label_mask"]
        loss *= cls_mask
        return loss.sum() / cls_mask.sum()


class MaskedFrameBceLoss(nn.Module):

    def forward(self, output: Dict):
        prob = output["frame_sim"] # [N, T, C]
        length = output["length"] # [N]
        label = output["strong_label"] # [N, T, C]
        loss = F.binary_cross_entropy(prob, label, reduction="none") # [N, T, C]
        len_mask = generate_length_mask(length).to(prob.device)
        loss *= len_mask.unsqueeze(-1)
        cls_mask = output["strong_label_mask"]
        loss *= cls_mask.unsqueeze(1)
        mask = len_mask.unsqueeze(-1) * cls_mask.unsqueeze(1)
        return loss.sum() / mask.sum()


class ClipMaskedFrameBceLoss(nn.Module):

    def __init__(self, frame_weight):
        super().__init__()
        self.clip_loss_fn = ClipBceLoss()
        self.frame_loss_fn = MaskedFrameBceLoss()
        self.frame_weight = frame_weight
    
    def forward(self, output: Dict):
        return (1 - self.frame_weight) * self.clip_loss_fn.forward_tensor(
            output["clip_sim"], output["weak_label"]) + \
            self.frame_weight * self.frame_loss_fn(output)


class ClipFrameBceLoss(nn.Module):

    def __init__(self, frame_weight):
        super().__init__()
        self.clip_loss_fn = ClipBceLoss()
        self.frame_loss_fn = FrameBceLoss()
        self.frame_weight = frame_weight
    
    def forward(self, output: Dict):
        return (1 - self.frame_weight) * self.clip_loss_fn.forward_tensor(
            output["clip_sim"], output["weak_label"]) + \
            self.frame_weight * self.frame_loss_fn.forward_tensor(
                output["frame_sim"], output["strong_label"], output["length"])


class VectorQuantizeLoss(nn.Module):

    def __init__(self, loss_fn, vq_weight=1.0):
        super().__init__()
        self.loss_fn = loss_fn
        self.vq_weight = vq_weight

    def forward(self, output):
        return self.vq_weight * output["vq_loss"] + self.loss_fn(output)

    def get_separate_loss(self, output):
        return {
            "cls_loss": self.loss_fn(output),
            "vq_loss": output["vq_loss"]
        }


class MaxMarginRankingLoss(nn.Module):

    def __init__(self, margin=1, fix_norm=True, lamda1=1):
        super().__init__()
        self.fix_norm = fix_norm
        self.margin = margin
        self.lamda1 = lamda1

    def forward(self, x):
        x = x["sim"]
        n = x.size()[0]

        x1 = torch.diag(x)
        x1 = x1.unsqueeze(1)
        x1 = x1.expand(n, n)
        x1 = x1.contiguous().view(-1, 1)
        x1 = torch.cat((x1, x1), 0)

        x2 = x.view(-1, 1)
        x3 = x.transpose(0, 1).contiguous().view(-1, 1)

        x3 = self.lamda1 * x3
        x2 = torch.cat((x2, x3), 0)
        max_margin = F.relu(self.margin - (x1 - x2))

        if self.fix_norm:
            # remove the elements from the diagonal
            keep = torch.ones(x.shape) - torch.eye(x.shape[0])  # 128 x 128
            keep1 = keep.view(-1, 1)
            keep2 = keep.transpose(0, 1).contiguous().view(-1, 1)
            keep_idx = torch.nonzero(torch.cat((keep1, keep2), 0).flatten()).flatten()
            if x1.is_cuda:
                keep_idx = keep_idx.cuda()
            x1_ = torch.index_select(x1, dim=0, index=keep_idx)
            x2_ = torch.index_select(x2, dim=0, index=keep_idx)
            max_margin = F.relu(self.margin - (x1_ - x2_))

        return max_margin.mean()


class InfoNceLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, output: Dict):
        sim = output["sim"]
        logit = sim.T 
        batch_size = sim.size(0)
        label = torch.arange(batch_size).to(logit.device)
        loss_a = self.loss_fn(logit.T, label)
        loss_t = self.loss_fn(logit, label)
        loss = (loss_a + loss_t) / 2
        return loss
