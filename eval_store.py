from argparse import ArgumentParser

import pytorch_lightning as pl
from torch_geometric.data import DataLoader

from datasets import ArgoverseV1Dataset
from models.hivt import HiVT
import logging
logger = logging.getLogger(__name__)
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, NamedTuple, Any, Union, Optional
import os
import pickle
from tqdm import tqdm


def compute_ade(forecasted_trajectories, gt_trajectory):
    """Compute the average displacement error for a set of K predicted trajectories (for the same actor).

    Args:
        forecasted_trajectories: (K, N, 2) predicted trajectories, each N timestamps in length.
        gt_trajectory: (N, 2) ground truth trajectory.

    Returns:
        (K,) Average displacement error for each of the predicted trajectories.
    """
    # displacement_errors = np.mean(np.linalg.norm(forecasted_trajectories - gt_trajectory, axis=-1), 1)
    displacement_errors = np.sqrt(np.sum((forecasted_trajectories - gt_trajectory)**2, -1))
    ade = np.mean(displacement_errors, axis=-1)
    return ade


def compute_fde(forecasted_trajectories, gt_trajectory):
    """Compute the final displacement error for a set of K predicted trajectories (for the same actor).

    Args:
        forecasted_trajectories: (K, N, 2) predicted trajectories, each N timestamps in length.
        gt_trajectory: (N, 2) ground truth trajectory, FDE will be evaluated against true position at index `N-1`.

    Returns:
        (K,) Final displacement error for each of the predicted trajectories.
    """
    # Compute final displacement error for all K trajectories
    error_vector = forecasted_trajectories - gt_trajectory
    fde_vector = error_vector[:, -1]
    fde = np.linalg.norm(fde_vector, axis=-1)
    return fde


class Metric:
    def __init__(self):
        self.values = []

    def accumulate(self, value):
        if value is not None:
            self.values.append(value)

    def get_mean(self):
        if len(self.values) > 0:
            return np.mean(self.values)
        else:
            return 0.0

    def get_sum(self):
        return np.sum(self.values)


class PredictionMetrics:
    def __init__(self):
        self.minADE = Metric()
        self.minFDE = Metric()
        self.MR = Metric()
        self.brier_minFDE = Metric()

    def serialize(self) -> Dict[str, Any]:
        return dict(
            minADE=float(self.minADE.get_mean()),
            minFDE=float(self.minFDE.get_mean()),
            MR=float(self.MR.get_mean()),
            brier_minFDE=float(self.brier_minFDE.get_mean()),
        )


if __name__ == '__main__':

    #! set split first.
    split='train'
    
    #! prepare your model, dataloader configuration here.
    model = None
    dataloader = None

    processed_dir = './p1/'
    model.to("cuda")
    model.eval()
    metrics = PredictionMetrics()
    for data in tqdm(dataloader):
        data.to("cuda")
        with torch.no_grad():
            #! infer your model here and output trajectory and embeddings.
            pred_trajectory = None  # [K, N, T, 2]
            embeds = None           # [K, N, -1]

            file_names = None       # data ids
            gt_eval = None    # ground-truth: [N, T, 2]

        embeds = embeds.transpose(0,1).detach().cpu().numpy()
        pred_trajectory = pred_trajectory.detach().cpu().numpy()
        gt_eval = gt_eval.detach().cpu().numpy()
        for i in range(gt_trajectory.shape[0]):
            forecasted_trajectories = pred_trajectory[i][:, :, :]
            gt_trajectory = gt_eval[i][:,:]
            #! make sure the file name is the same with original id in dataset.
            raw_file_name = file_names[i]

            #! dict to store..
            dict_data = {
                'traj': torch.from_numpy(pred_trajectory[i].copy().astype(np.float32)),
                'embed': torch.from_numpy(embeds[i].copy().astype(np.float32)),
            }
            with open(os.path.join(processed_dir, split, f'{raw_file_name}.pkl'), 'wb') as handle:
                pickle.dump(dict_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

            assert forecasted_trajectories.shape == (6, 30, 2)
            assert gt_trajectory.shape == (30, 2)

            fde = compute_fde(forecasted_trajectories, gt_trajectory)
            idx = fde.argmin()
            ade = compute_ade(forecasted_trajectories[idx], gt_trajectory)

            metrics.minADE.accumulate(ade.min())
            metrics.minFDE.accumulate(fde.min())
            metrics.MR.accumulate(fde.min() > 2.0)
    import json
    print('Metrics:')
    print(json.dumps(metrics.serialize(), indent=4))
