import time
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
from pathlib import Path
import os
import json
import copy
import argparse
import random

# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.distributed import init_process_group, destroy_process_group

from ..common.model_utils import get_model, ddp_setup
from ..models.base import Trainer


def run_train(rank, world_size, cfg):
    if cfg.train.deterministic:
        torch.manual_seed(cfg.train.random_seed)
        torch.cuda.manual_seed(cfg.train.random_seed)
        torch.cuda.manual_seed_all(cfg.train.random_seed)
        np.random.seed(cfg.train.random_seed)
        random.seed(cfg.train.random_seed)

    os.makedirs(cfg.output_dir, exist_ok=True)

    run_ddp = (rank != "cpu") and cfg.ddp
    if run_ddp:
        ddp_setup(rank, world_size)

    model = get_model(cfg)
    trainer = Trainer(model, rank, world_size, cfg)
    trainer.train_start()
    print(trainer.model)
    print(f"\nModel parameters [GPU{trainer.device}]:")
    print(f"{round(sum(p.numel() for p in trainer.model.parameters() if p.requires_grad) / 1e6, 2)}M")

    for e in range(trainer.current_epoch, cfg.train.max_epochs):
        tick = time.time()
        trainer.train()

        trainer.train_epoch_start(e)
        if run_ddp:
            trainer.train_sampler.set_epoch(e)
        for batch_idx, batch in enumerate(trainer.train_dataloader):
            loss = trainer.training_step(batch, batch_idx)
            trainer.optimizer.zero_grad()
            loss.backward()
            trainer.clip_grad_value_()
            trainer.optimizer.step()
            if cfg.optim.lr_scheduler._target_ != "ReduceLROnPlateau":
                trainer.scheduler.step()
            trainer.train_step_end(e)

        trainer.train_epoch_end(e)

        if e % cfg.logging.check_val_every_n_epoch == 0:

            trainer.eval()
            trainer.val_epoch_start(e)
            if run_ddp:
                trainer.val_sampler.set_epoch(e)

            with torch.no_grad():
                outs = []
                for val_batch_idx, val_batch in enumerate(trainer.val_dataloader):
                    val_out = trainer.validation_step(val_batch, val_batch_idx)
                    outs.append(val_out.detach())
                    trainer.val_step_end(e)

            trainer.val_epoch_end(e)

            if cfg.optim.lr_scheduler._target_ == "ReduceLROnPlateau":
                trainer.scheduler.step(torch.mean(torch.stack([x for x in outs])))

        trainer.train_val_epoch_end(e)
        print(f"\tTraining time: {time.time() - tick} s")

        if trainer.early_stopping(e):
            break

    trainer.train_end(e)
    # destroy_process_group()
