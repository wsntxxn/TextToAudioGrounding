import math
from typing import Dict
import numpy as np
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


class MilNceLoss(nn.Module):

    def __init__(self, tau=1.0) -> None:
        super().__init__()
        self.tau = tau

    def forward(self, output: Dict):
        clip_sim = output["clip_sim"]
        label = output["label"]
        nominator = torch.logsumexp(clip_sim * label / self.tau, dim=1)
        denominator = torch.logsumexp(clip_sim / self.tau, dim=1)
        return torch.mean(denominator - nominator)


class FocalClipBceLoss(nn.Module):

    def __init__(self, gamma=2, alpha=0.25) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, output: Dict):
        clip_sim = output["clip_sim"]
        label = output["label"]

        loss = -self.alpha * torch.pow(1 - clip_sim, self.gamma) * label * \
            torch.log(clip_sim) - (1 - self.alpha) * torch.pow(clip_sim, self.gamma) * \
            (1.0 - label) * torch.log(1 - clip_sim)
        return loss.mean()


class ClipBceLossFreqWeight(nn.Module):

    def __init__(self, C, gamma) -> None:
        super().__init__()
        self.C = C
        self.gamma = gamma

    def forward(self, output: Dict):
        counts = output["counts"]
        label = output["label"]
        weight = (self.C / (self.C + counts)) ** self.gamma
        weight = torch.as_tensor(weight).to(output["clip_sim"].device)
        weight = torch.where(label == 0.0, 1.0, weight)
        return F.binary_cross_entropy(output["clip_sim"], label, weight=weight)


class SymmetricClipBceLoss(nn.Module):

    def __init__(self, a=1, b=1, eps=1e-3) -> None:
        super().__init__()
        self.a = a
        self.b = b
        self.eps = eps

    def forward(self, output: Dict):
        clip_sim = output["clip_sim"]
        label = output["label"]
        loss = F.binary_cross_entropy(clip_sim, label)
        loss += F.binary_cross_entropy(label.clamp(self.eps, 1.0 - self.eps), clip_sim)
        return loss


class OriginSymmetricClipBceLoss(nn.Module):

    def __init__(self, a=1, b=1, eps=1e-3) -> None:
        super().__init__()
        self.a = a
        self.b = b
        self.A = math.log(eps)

    def forward(self, output: Dict):
        clip_sim = output["clip_sim"]
        label = output["label"]
        loss = F.binary_cross_entropy(clip_sim, label)
        reverse_loss = - (label * (1 - clip_sim) * self.A + (1 - label) * self.A * clip_sim).mean()
        tot_loss = self.a * loss + self.b * reverse_loss
        return tot_loss


class PriorAdjustedClipBceLoss(nn.Module):

    def __init__(self, data_size, tau=1) -> None:
        super().__init__()
        self.data_size = data_size # [n_classes, ]
        self.tau = tau

    def forward(self, output: Dict):
        clip_sim = output["clip_sim"]
        label = output["label"]
        counts = output["counts"]
        counts = torch.as_tensor(counts).to(clip_sim.device)
        prior = counts / self.data_size
        adjusted_one_logit = clip_sim * ((prior) ** self.tau)
        adjusted_zero_logit = (1 - clip_sim) * ((1 - prior) ** self.tau)
        adjusted_clip_sim = adjusted_one_logit / (adjusted_one_logit + adjusted_zero_logit)
        loss = F.binary_cross_entropy(adjusted_clip_sim, label)
        return loss


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

    def __init__(self,
                 frame_weight,
                 clip_label_key="weak_label",
                 clip_prob_key="clip_sim",
                 frame_label_key="strong_label",
                 frame_prob_key="frame_sim"):
        super().__init__()
        self.clip_loss_fn = ClipBceLoss()
        self.frame_loss_fn = FrameBceLoss()
        self.frame_weight = frame_weight
        self.clip_label_key = clip_label_key
        self.clip_prob_key = clip_prob_key
        self.frame_label_key = frame_label_key
        self.frame_prob_key = frame_prob_key
    
    def forward(self, output: Dict):
        return (1 - self.frame_weight) * self.clip_loss_fn.forward_tensor(
            output[self.clip_prob_key], output[self.clip_label_key]) + \
            self.frame_weight * self.frame_loss_fn.forward_tensor(
                output[self.frame_prob_key],
                output[self.frame_label_key],
                output["length"])


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


class MaxMarginRankingLoss(nn.Module): # triplet weighted

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

    def __init__(self, tau=0.07):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        self.tau = tau

    def forward(self, output: Dict):
        sim = output["sim"]
        logit = sim.T / self.tau
        batch_size = sim.size(0)
        label = torch.arange(batch_size).to(logit.device)
        loss_a = self.loss_fn(logit.T, label)
        loss_t = self.loss_fn(logit, label)
        loss = (loss_a + loss_t) / 2
        return loss


# triplet max
class MaxTripletLoss(nn.Module):

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, output):
        sim = output["sim"]
        n = sim.size(0)  # batch size
        label = torch.arange(n)

        sim_ap = torch.diag(sim).view(n, 1)
        d1 = sim_ap.expand_as(sim)
        d2 = sim_ap.t().expand_as(sim)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = F.relu(self.margin + sim - d1)
        # compare every diagonal score to scores in its row
        # audio retrieval
        cost_a = F.relu(self.margin + sim - d2)

        # clear diagonals
        mask = label.expand(n, n).eq(label.expand(n, n).t()).to(cost_a.device)
        cost_s = cost_s.masked_fill(mask, 0)
        cost_a = cost_a.masked_fill(mask, 0)

        cost_s = cost_s.max(1)[0]
        cost_a = cost_a.max(0)[0]
        loss = (cost_s.sum() + cost_a.sum()) / n

        return loss


# triplet random
class RandomTripletLoss(nn.Module):

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, output):
        sim = output["sim"]
        n = sim.size(0)  # batch size
        label = torch.arange(n)

        sim_ap = torch.diag(sim).view(n, 1)
        d1 = sim_ap.expand_as(sim)
        d2 = sim_ap.t().expand_as(sim)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = F.relu(self.margin + sim - d1)
        # compare every diagonal score to scores in its row
        # audio retrieval
        cost_a = F.relu(self.margin + sim - d2)

        # clear diagonals
        mask = label.expand(n, n).eq(label.expand(n, n).t()).to(cost_a.device)
        cost_s = cost_s.masked_fill(mask, 0)
        cost_a = cost_a.masked_fill(mask, 0)

        s_index = torch.as_tensor(np.random.randint(n, size=n))
        cost_s = cost_s[torch.arange(n), s_index]
        a_index = torch.as_tensor(np.random.randint(n, size=n))
        cost_a = cost_a[torch.arange(n), a_index]
        loss = (cost_s.sum() + cost_a.sum()) / n

        return loss


# weighted triplet
class WeightedTripletLoss(nn.Module):

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def polyloss(self, sim_mat, label):
        epsilon = 1e-5
        size = sim_mat.size(0)
        hh = sim_mat.t()

        loss = list()
        for i in range(size):
            pos_pair_ = sim_mat[i][i]
            # pos_pair_ = pos_pair_[pos_pair_ < 1 - epsilon]
            neg_pair_ = sim_mat[i][label != label[i]]

            neg_pair = neg_pair_[neg_pair_ + self.margin > pos_pair_]

            pos_pair = pos_pair_
            if len(neg_pair) < 1:
                continue

            pos_loss = torch.clamp(0.2 * torch.pow(pos_pair, 2) - 0.7 * pos_pair + 0.5, min=0)
            neg_pair = max(neg_pair)
            neg_loss = torch.clamp(0.9 * torch.pow(neg_pair, 2) - 0.4 * neg_pair + 0.03, min=0)

            loss.append(pos_loss + neg_loss)
        for i in range(size):
            pos_pair_ = hh[i][i]
            # pos_pair_ = pos_pair_[pos_pair_ < 1 - epsilon]
            neg_pair_ = hh[i][label != label[i]]

            neg_pair = neg_pair_[neg_pair_ + self.margin > pos_pair_]

            pos_pair = pos_pair_
            if len(neg_pair) < 1:
                continue
            pos_loss = torch.clamp(0.2 * torch.pow(pos_pair, 2) - 0.7 * pos_pair + 0.5, min=0)

            neg_pair = max(neg_pair)
            neg_loss = torch.clamp(0.9 * torch.pow(neg_pair, 2) - 0.4 * neg_pair + 0.03, min=0)
            loss.append(pos_loss + neg_loss)

        if len(loss) == 0:
            # return torch.zeros([], requires_grad=True)
            return sim_mat.mean() - sim_mat.mean() + 0.0

        loss = sum(loss) / size
        return loss

    def forward(self, output):
        scores = output["sim"]
        label = torch.arange(scores.shape[0])
        loss = self.polyloss(scores, label)
        return loss

