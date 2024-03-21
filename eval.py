from argparse import ArgumentParser

import pytorch_lightning as pl
from torch_geometric.data import DataLoader

from datasets import ArgoverseV1Dataset
from models.refine import Refine
import logging
logger = logging.getLogger(__name__)
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

if __name__ == '__main__':
    pl.seed_everything(2024)

    parser = ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--ckpt_dir', type=str, required=True)
    parser = Refine.add_model_specific_args(parser)
    args = parser.parse_args()

    trainer = pl.Trainer.from_argparse_args(args)
    ckpt_dir=args.ckpt_dir+'checkpoints/'
    ckpt_paths = [ckpt_dir+p for p in os.listdir(ckpt_dir) if p.endswith('ckpt')]
    ckpt_paths.sort()
    ckpt_path = ckpt_paths[-1]

    model = Refine.load_from_checkpoint(checkpoint_path=ckpt_path)
    model.eval()
    val_dataset = ArgoverseV1Dataset(data_root=args.data_root, p1_root='', split='val', local_radius=model.hparams.local_radius)
    dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=args.pin_memory, persistent_workers=args.persistent_workers)

    trainer.validate(model, dataloader)
