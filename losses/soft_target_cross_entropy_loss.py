import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftTargetCrossEntropyLoss(nn.Module):

    def __init__(self, reduction: str = 'mean') -> None:
        super(SoftTargetCrossEntropyLoss, self).__init__()
        self.reduction = reduction

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        cross_entropy = torch.sum(-target * F.log_softmax(pred, dim=-1), dim=-1)
        if self.reduction == 'mean':
            return cross_entropy.mean()
        elif self.reduction == 'sum':
            return cross_entropy.sum()
        elif self.reduction == 'none':
            return cross_entropy
        else:
            raise ValueError('{} is not a valid value for reduction'.format(self.reduction))
