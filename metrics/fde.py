from typing import Any, Callable, Optional

import torch
from torchmetrics import Metric


class FDE(Metric):

    def __init__(self,
                 **kwargs) -> None:
        super(FDE, self).__init__(**kwargs)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self,
               pred: torch.Tensor,
               target: torch.Tensor=None) -> None:
        if target is not None:
            self.sum += torch.norm(pred[:, -1] - target[:, -1], p=2, dim=-1).sum()
            self.count += pred.size(0)
        else:
            self.sum += pred.sum()
            self.count += pred.size(0)

    def compute(self) -> torch.Tensor:
        return self.sum / self.count
