from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from datamodules import ArgoverseV1DataModule
from models.refine import Refine

import torch
import os

parser = ArgumentParser()
parser.add_argument('--data_root', type=str, required=True)
parser.add_argument('--p1_root', type=str, required=True)
parser.add_argument('--train_batch_size', type=int, default=4)
parser.add_argument('--val_batch_size', type=int, default=4)
parser.add_argument('--shuffle', type=bool, default=True)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--seed', type=int, default=2024)
parser.add_argument('--pin_memory', type=bool, default=True)
parser.add_argument('--persistent_workers', type=bool, default=True)
parser.add_argument('--prefetch_factor', type=int, default=4)
parser.add_argument('--max_epochs', type=int, default=64)
parser.add_argument('--monitor', type=str, default='val_minFDE', choices=['val_minADE', 'val_minFDE', 'val_minMR'])
parser.add_argument('--save_top_k', type=int, default=5)
parser.add_argument('--exp_name', type=str, required=True)
parser.add_argument('--gpus', type=int, default=1)
parser = Refine.add_model_specific_args(parser)
args = parser.parse_args()
if args.num_workers == 0:
    args.persistent_workers = False
args.accelerator='auto'
if args.gpus > 1:
    args.strategy="ddp_find_unused_parameters_false"

pl.seed_everything(args.seed)
model_checkpoint = ModelCheckpoint(monitor=args.monitor, save_top_k=args.save_top_k, mode='min')
#! base dir for loogging
base_dir="./"
trainer = pl.Trainer.from_argparse_args(args, callbacks=[model_checkpoint],
                    default_root_dir=base_dir+args.exp_name)
model = Refine(**vars(args))
datamodule = ArgoverseV1DataModule.from_argparse_args(args)
trainer.fit(model, datamodule)
