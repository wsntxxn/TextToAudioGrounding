from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import ignite.metrics as metrics
from ignite.engine.engine import Engine

from models.utils import generate_length_mask


class BinaryCrossEntropyLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, output: Dict):
        score = output["score"] # [N, T]
        length = output["length"] # [N]
        label = output["label"] # [N, T]
        loss = F.binary_cross_entropy(score, label, reduction="none") # [N, T]
        mask = generate_length_mask(length).to(score.device)
        loss *= mask
        loss = loss.sum() / mask.sum()
        return loss


class Loss(metrics.Loss):

    def update(self, output: Dict) -> None:
        # score: [bs, max_len]
        # label: [bs, max_len]
        # length: [bs]
        length = output["length"]
        average_loss = self._loss_fn(output).detach()

        if len(average_loss.shape) != 0:
            raise ValueError("loss_fn did not return the average loss.")

        n = torch.sum(torch.as_tensor(length))
        self._sum += average_loss.to(self._device) * n
        self._num_examples += n

    @torch.no_grad()
    def iteration_completed(self, engine: Engine) -> None:
        output = self._output_transform(engine.state.output)
        self.update(output)
