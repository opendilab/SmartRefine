from typing import Any, Callable, Optional

import torch
from torchmetrics import Metric


class MR(Metric):

    def __init__(self,
                 miss_threshold: float = 2.0,
                 **kwargs,) -> None:
        super(MR, self).__init__(**kwargs)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        self.miss_threshold = miss_threshold

    def update(self,
               pred: torch.Tensor,
               target: torch.Tensor) -> None:
        self.sum += (torch.norm(pred[:, -1] - target[:, -1], p=2, dim=-1) > self.miss_threshold).sum()
        self.count += pred.size(0)

    def compute(self) -> torch.Tensor:
        return self.sum / self.count
