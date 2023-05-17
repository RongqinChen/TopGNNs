import torch
from torch import nn
from torch import Tensor
from torch.nn.functional import one_hot


class CLCrossEntropyLoss(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        # num_samples, num_class = targets.size(0), preds.size(1)
        num_samples = targets.size(0)
        if preds.size(0) == num_samples * 2:
            device = targets.device
            one_hot_a = one_hot(targets)
            one_hot_b = 1 - one_hot(targets)
            double_one_hot = torch.concat(
                (one_hot_a, one_hot_b)).to(device, torch.float32)

            bce = self.bce(preds, double_one_hot) * 2
            preds_a, preds_b = torch.split(preds, (num_samples, num_samples))
            ce_a = self.ce(preds_a, targets)
            ce_b = self.ce(-preds_b, targets)
            loss = (bce + ce_a + ce_b) / 4.

        else:
            loss = self.ce(preds, targets)

        return loss


class CLBCEWithLogitsLoss(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        # num_samples, num_class = targets.size(0), preds.size(1)
        num_samples = targets.size(0)
        if preds.size(0) == num_samples * 2:
            preds_a, preds_b = torch.split(preds, (num_samples, num_samples))
            ce_a = self.bce(preds_a, targets)
            ce_b = self.bce(-preds_b, targets)
            loss = (ce_a + ce_b) / 2.
        else:
            loss = self.bce(preds, targets)

        return loss
