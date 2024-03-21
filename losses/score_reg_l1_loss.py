import torch
import torch.nn as nn
import torch.nn.functional as F


class ScoreRegL1Loss(nn.Module):

    def __init__(self, reduction: str = 'mean') -> None:
        super(ScoreRegL1Loss, self).__init__()
        self.loss = nn.L1Loss(reduction=reduction)

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        if pred.shape[0] == 0:
            return 0
        return self.loss(pred, target)
