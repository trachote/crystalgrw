from typing import Any, Dict

import numpy as np
import math
import copy
import json
import os
import glob
import sys
from omegaconf import DictConfig, OmegaConf

import torch
from torch import sqrt
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch_scatter import scatter

from ..common.datamodule import CrystDataModule
from ..common.data_utils import (
    EPSILON, cart_to_frac_coords, mard, lengths_angles_to_volume,
    frac_to_cart_coords, min_distance_sqr_pbc)


# from ..common.model_utils import get_model


class BaseModel(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.hparams = cfg.model
        self.hparams.data = cfg.data

    def kld_reparam(self, hidden):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        mu = self.fc_mu(hidden)
        log_var = self.fc_var(hidden)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = eps * std + mu
        return mu, log_var, z

    def gaussian_kld(self, q_mu, q_logvar, p_mu, p_logvar, batch_idx, is_logvar=False, reduce="sum"):
        """
        KLD(q || p)
        """
        if not is_logvar:
            q_logvar = 2 * torch.log(q_logvar)
            p_logvar = 2 * torch.log(p_logvar)
        kld = 0.5 * (p_logvar - q_logvar - 1 \
                     + (torch.exp(q_logvar) + (q_mu - p_mu) ** 2) / torch.exp(p_logvar))
        #         kld = torch.log(p_sigma / q_sigma) \
        #               + 0.5 * (q_sigma**2 + (q_mu - p_mu)**2) / (p_sigma**2) \
        #               - 0.5
        return scatter(kld, batch_idx, dim=0, reduce=reduce)

    def generate_rand_init(self, pred_composition_per_atom, pred_lengths,
                           pred_angles, num_atoms, batch):
        rand_frac_coords = torch.rand(num_atoms.sum(), 3,
                                      device=num_atoms.device)
        pred_composition_per_atom = F.softmax(pred_composition_per_atom,
                                              dim=-1)
        rand_atom_types = self.sample_composition(
            pred_composition_per_atom, num_atoms)
        return rand_frac_coords, rand_atom_types

    def l2_loss(self, output, target, batch_idx=None, norm=False):
        if norm:
            factor = target.shape[-1]
        else:
            factor = 1.

        loss = torch.sum((output - target) ** 2 / factor, dim=1)
        if batch_idx is not None:
            loss = scatter(loss, batch_idx, reduce="mean").mean()
        else:
            loss = loss.mean()
        return loss

    def num_atom_loss(self, pred_num_atoms, num_atoms):
        return F.cross_entropy(pred_num_atoms, num_atoms)

    def lattice_loss(self, pred_lengths_and_angles, batch):
        self.lattice_scaler.match_device(pred_lengths_and_angles)
        if self.hparams.data.lattice_scale_method == "scale_length":
            target_lengths = batch.lengths / \
                             batch.num_atoms.view(-1, 1).float() ** (1 / 3)
        target_lengths_and_angles = torch.cat(
            [target_lengths, batch.angles], dim=-1)
        target_lengths_and_angles = self.lattice_scaler.transform(
            target_lengths_and_angles)
        return F.mse_loss(pred_lengths_and_angles, target_lengths_and_angles)

    def composition_loss(self, pred_composition_per_atom, target_atom_types, batch_idx):
        target_atom_types = target_atom_types - 1
        loss = F.cross_entropy(pred_composition_per_atom,
                               target_atom_types, reduction="none")
        return scatter(loss, batch_idx, reduce="mean").mean()

    def coord_loss(self, pred_coord_diff, noisy_coords,
                   alpha_t, beta_t, batch, loss_type="coord"):
        alpha_t, beta_t = alpha_t.unsqueeze(-1), beta_t.unsqueeze(-1)
        noisy_coords = frac_to_cart_coords(noisy_coords, batch.lengths,
                                           batch.angles, batch.num_atoms)
        noisy_coords = self.add_mean(noisy_coords, recenter=self.recenter)
        target_coords = frac_to_cart_coords(batch.frac_coords, batch.lengths, batch.angles, batch.num_atoms)
        _, target_coord_diff = min_distance_sqr_pbc(target_coords, noisy_coords,
                                                    batch.lengths, batch.angles,
                                                    batch.num_atoms, noisy_coords.device,
                                                    return_vector=True)
        target_coord_diff = (target_coord_diff - (sqrt(alpha_t) - 1) * target_coords)
        target_coord_diff = self.coords(target_coord_diff, batch, input_type="cart")

        loss_per_atom = torch.sum(
            (target_coord_diff - pred_coord_diff) ** 2, dim=1)
        return scatter(loss_per_atom, batch.batch, reduce="mean").mean()

    def type_loss(self, pred_atom_types, target_atom_types,
                  batch_idx):
        target_atom_types = target_atom_types - 1
        loss = F.cross_entropy(
            pred_atom_types, target_atom_types, reduction="none")
        # rescale loss according to noise
        loss = loss
        return scatter(loss, batch_idx, reduce="mean").mean()

    def kld_loss(self, mu, log_var):
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        return kld_loss

    def get_condition(self, labels, num_atoms):
        if self.control_fn is not None:
            uncond_prob = torch.zeros(labels.size(0)).to(num_atoms.device) + self.uncond_prob
            cond_mask = torch.bernoulli(1. - uncond_prob).bool().to(num_atoms.device).repeat_interleave(num_atoms)
            condition = torch.zeros(num_atoms.sum(0), self.cfg.controller.hidden_dim).to(num_atoms.device)
            condition[cond_mask] = self.control_fn(labels, num_atoms)[cond_mask]
            return condition
        else:
            return None

    def control_score(self, labels, natoms, guidance_strength, **kwargs):
        if self.control_fn is None:
            raise NotImplementedError(
                "This model has not been trained "
                "with a control function.")

        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor([labels] * natoms.size(0),
                                  device=natoms.device).view(natoms.size(0), -1)
        else:
            labels = labels.view(natoms.size(0), -1)

        condition = self.control_fn(labels, natoms)
        uncondition = torch.zeros_like(condition)

        cond_scores = self.score_fn(cond_feat=condition,
                                    natoms=natoms,
                                    **kwargs)
        uncond_scores = self.score_fn(cond_feat=uncondition,
                                      natoms=natoms,
                                      **kwargs)

        scores = {}
        for f in uncond_scores:
            scores[f] = ((1 + guidance_strength) * cond_scores[f]
                         - guidance_strength * uncond_scores[f])
        return scores


class Trainer:
    def __init__(self, model, rank, world_size, cfg, training=True, load_data=True) -> None:
        super().__init__()
        self.cfg = cfg
        self.hparams = cfg.model
        self.device = rank
        self.hparams.data = cfg.data
        self.current_epoch = 0
        self.logs = {"train": [], "val": [], "test": []}
        self.train_checkpoint_path = None
        self.val_checkpoint_path = None
        self.min_val_loss = float("inf")
        self.min_val_epoch = 0
        self.train_log = None
        self.val_log = None
        self.test_log = None
        self.datamodule = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None
        self.train_sampler = None
        self.val_sampler = None
        self.test_sampler = None

        self.run_ddp = (rank != "cpu") and cfg.ddp
        self.model = model.to(self.device)
        self.model_name = self.model.model_name
        self.init(training, load_data)
        if self.run_ddp:
            self.model = DDP(self.model, device_ids=[rank])

        if not training:
            self.model.eval()

    def init(self, training=True, load_data=True):
        if training:
            # load_data = True
            self.init_optimizer()
            self.init_scheduler()

        checkpoint = self.load_checkpoint(training)

        if not checkpoint:
            print("No checkpoint: Save hparams.yaml")
            with open(self.cfg.output_dir + "/hparams.yaml", "w") as f:
                f.write(OmegaConf.to_yaml(cfg=self.cfg))

        if load_data:
            self.init_datamodule()
            self.init_dataloader(training)

    def init_optimizer(self):
        print(f"Instantiating Optimizer <{self.cfg.optim.optimizer._target_}>")
        # if self.cfg.optim.optimizer._target_=="AdamW":
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           **{k: v for k, v in self.cfg.optim.optimizer.items() if k != "_target_"})

    def init_scheduler(self):
        print(f"Instantiating LR Scheduler <{self.cfg.optim.lr_scheduler._target_}>")
        print(f"LR Scheduler: ", self.cfg.optim.use_lr_scheduler)
        # if self.cfg.optim.optimizer.optimizer._target_=="ReduceLROnPlateau":
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, **{k: v for k, v in
                                                                                       self.cfg.optim.lr_scheduler.items()
                                                                                       if k != "_target_"})

    def init_datamodule(self):
        # if self.cfg.data.datamodule._target_ == "CrystDataModule":
        print(f"Set up datamodule")
        self.datamodule = CrystDataModule(**{k: v for k, v in self.cfg.data.datamodule.items() if k != "_target_"},
                                          run_ddp=self.run_ddp,
                                          dataset=self.cfg.data.datamodule._target_)

    def init_dataloader(self, training):
        assert self.datamodule is not None
        self.datamodule.setup(training)
        if not training:
            self.test_dataloader, self.test_sampler = self.datamodule.test_dataloader()[0]
        else:
            self.train_dataloader, self.train_sampler = self.datamodule.train_dataloader()
            self.val_dataloader, self.val_sampler = self.datamodule.val_dataloader()[0]

    def main_process(self):
        """
        Check if the current process is the main process.
        """
        return dist.get_rank() == 0 if self.run_ddp else True

    def train_start(self):
        print(">>>TRAINING START<<<")
        pass

    def train_end(self, e):
        print(">>>TRAINING END<<<")
        self.logging(e)
        if self.main_process():
            self.train_checkpoint_path = self.save_checkpoint(
                model_checkpoint_path=self.train_checkpoint_path,
                suffix="train"
            )

    def clip_grad_value_(self):
        torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=0.5)

    def train_step_end(self, e):
        # for examination
        pass

    def val_step_end(self, e):
        # for examination
        pass

    def train_epoch_start(self, e):
        self.clear_log_dict()
        self.train_log = None
        self.val_log = None
        self.test_log = None

    def val_epoch_start(self, e):
        pass

    def train_epoch_end(self, e):
        log_dict = {"epoch": e}
        log_dict.update(
            {k: np.mean([x[k].item() if torch.is_tensor(x[k]) else x[k] for x in self.logs["train"]]) for k in
             self.logs["train"][0].keys()})

        with open(self.cfg.output_dir + "/train_metrics.json", "a") as f:
            f.write(json.dumps({k: v for k, v in log_dict.items()}))
            f.write("\r\n")

        self.train_log = log_dict

    def val_epoch_end(self, e):
        log_dict = {"epoch": e}
        log_dict.update({k: np.mean([x[k].item() if torch.is_tensor(x[k]) else x[k] for x in self.logs["val"]]) for k in
                         self.logs["val"][0].keys()})

        with open(self.cfg.output_dir + "/val_metrics.json", "a") as f:
            f.write(json.dumps({k: v for k, v in log_dict.items()}))
            f.write("\r\n")

        self.val_log = log_dict

        if self.val_log["val_loss"] < self.min_val_loss:
            self.min_val_loss = self.val_log["val_loss"]
            self.min_val_epoch = e
            if self.main_process():
                self.val_checkpoint_path = self.save_checkpoint(
                    model_checkpoint_path=self.val_checkpoint_path,
                    suffix="val",
                )

    def test_epoch_end(self):
        log_dict = {}
        log_dict.update(
            {k: np.mean([x[k].item() if torch.is_tensor(x[k]) else x[k] for x in self.logs["test"]]) for k in
             self.logs["test"][0].keys()})

        with open(self.cfg.output_dir + "/test_metrics.json", "a") as f:
            f.write(json.dumps({k: v for k, v in log_dict.items()}))
            f.write("\r\n")

        self.test_log = log_dict

    def train_val_epoch_end(self, e):

        if e % self.cfg.logging.log_freq_every_n_epoch == 0:
            self.logging(e)

        if (e % self.cfg.checkpoint_freq_every_n_epoch) and self.main_process():
            self.train_checkpoint_path = self.save_checkpoint(
                model_checkpoint_path=self.train_checkpoint_path,
                suffix="train",
            )

        self.current_epoch += 1

    def load_checkpoint(self, training, load_val_for_train=False):
        train_ckpts = list(glob.glob(f"{self.cfg.output_dir}/*={self.model_name}=train.ckpt"))
        val_ckpts = list(glob.glob(f"{self.cfg.output_dir}/*={self.model_name}=val.ckpt"))
        ckpts = train_ckpts + val_ckpts
        print("All checkpoints:", ckpts)
        checkpoint = False

        if training and not load_val_for_train:
            ckpts = train_ckpts + val_ckpts
        elif len(val_ckpts) == 0:
            print("No VAL checkpoints exist, use a TRAIN checkpoint instead.")
            ckpts = train_ckpts
        else:
            ckpts = val_ckpts

        if len(ckpts) > 0:
            ckpt_epochs = np.array([int(ck.split("/")[-1].split(".")[0].split("=")[1]) for ck in ckpts])
            ckpt_path = str(ckpts[ckpt_epochs.argsort()[-1]])

            print(f">>>>> Load the model from a checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=torch.device(self.device))
            self.current_epoch = ckpt["epoch"] + 1
            self.model.load_state_dict(ckpt["model_state_dict"])

            try:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)
            except:
                print("ERROR: Loading state dict")

            if self.cfg.optim.use_lr_scheduler:
                try:
                    self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                except:
                    print("ERROR: Loading scheduler")

            self.train_checkpoint_path = f"{self.cfg.output_dir}/epoch={ckpt['epoch']}={self.model_name}=train.ckpt"

            if val_ckpts:
                if ckpt_path.split("/")[-1].split("=")[-1][:-5] == "train":
                    val_epochs = np.array([int(ck.split("/")[-1].split(".")[0].split("=")[1]) for ck in val_ckpts])
                    val_path = str(val_ckpts[val_epochs.argsort()[-1]])
                    print(f">>>>> Load the val model from a checkpoint: {val_path}")
                    ckpt = torch.load(val_path, map_location=torch.device(self.device))
            else:
                print("No VAL checkpoints exist, set the last TRAIN checkpoint as the min val")

            self.min_val_epoch = ckpt["epoch"]
            self.min_val_loss = torch.tensor(ckpt["val_loss"])

            print("min val epoch: ", self.min_val_epoch)
            print("min val loss: ", self.min_val_loss.item())

            self.val_checkpoint_path = f"{self.cfg.output_dir}/epoch={ckpt['epoch']}={self.model_name}=val.ckpt"
            checkpoint = True
        else:
            print(f">>>>> New Training")

        return checkpoint

    def save_checkpoint(self, model_checkpoint_path, suffix="val", logs={}):
        model_checkpoint = {
            "model_state_dict":
                copy.deepcopy(self.model.module.state_dict())
                if self.run_ddp else copy.deepcopy(self.model.state_dict()),
            "optimizer_state_dict": copy.deepcopy(self.optimizer.state_dict()),
            "scheduler_state_dict": copy.deepcopy(self.scheduler.state_dict()),
            "epoch": self.current_epoch,
            "train_loss": self.train_log["train_loss"],
            "val_loss": self.val_log["val_loss"] if self.val_log else None
        }

        model_checkpoint.update(logs)

        new_model_checkpoint_path = f"{self.cfg.output_dir}/epoch={self.current_epoch}={self.model_name}={suffix}.ckpt"

        # if new_model_checkpoint_path != model_checkpoint_path:
        #     if model_checkpoint_path and os.path.exists(model_checkpoint_path):
        #         os.remove(model_checkpoint_path)
        #     print("Save model checkpoint: ", new_model_checkpoint_path)
        #     print("\tmodel checkpoint train loss: ", model_checkpoint["train_loss"])
        #     print("\tmodel checkpoint val loss: ", model_checkpoint["val_loss"])
        #     torch.save(model_checkpoint, new_model_checkpoint_path)

        print("Save model checkpoint: ", new_model_checkpoint_path)
        print("\tmodel checkpoint train loss: ", model_checkpoint["train_loss"])
        print("\tmodel checkpoint val loss: ", model_checkpoint["val_loss"])
        torch.save(model_checkpoint, new_model_checkpoint_path)

        ckpts = list(glob.glob(f"{self.cfg.output_dir}/*={self.model_name}={suffix}.ckpt"))
        for ckpt in ckpts:
            if (new_model_checkpoint_path != ckpt) and os.path.exists(ckpt) and len(ckpts) > 1:
                os.remove(ckpt)

        return new_model_checkpoint_path

    def early_stopping(self, e):
        if e - self.min_val_epoch > self.cfg.data.early_stopping_patience_epoch:
            print("Early stopping")
            return True

        return False

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        outputs = self.model(batch.to(self.device), training=True)
        log_dict, loss = self.compute_stats(batch, outputs, prefix="train")
        self.log_dict(
            log_dict,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            prefix="train"
        )
        if math.isnan(loss.item()):
            sys.exit()
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        outputs = self.model(batch.to(self.device), training=False)
        log_dict, loss = self.compute_stats(batch, outputs, prefix="val")
        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            prefix="val"
        )
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        outputs = self.model(batch.to(self.device), training=False)
        log_dict, loss = self.compute_stats(batch, outputs, prefix="test")
        self.log_dict(
            log_dict,
            prefix="test"
        )
        return loss

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def compute_stats(self, batch, outputs, prefix):
        losses = outputs["losses"]
        loss = 0
        log_dict = {}

        for k in losses:
            if hasattr(self.hparams, "cost_" + k):
                cost = getattr(self.hparams, "cost_" + k)
            else:
                cost = 1
            loss += cost * losses[k]
            log_dict.update({f"{prefix}_{k}_loss": losses[k]})

        log_dict.update({f"{prefix}_loss": loss})

        if prefix != "train":
            # evaluate num_atom prediction.
            if "pred_num_atoms" in outputs.keys():
                pred_num_atoms = outputs["pred_num_atoms"].argmax(dim=-1)
                num_atom_accuracy = (pred_num_atoms == batch.num_atoms).sum() / batch.num_graphs
                log_dict.update({f"{prefix}_natom_accuracy": num_atom_accuracy})

            if "pred_atom_types" in outputs.keys():
                if outputs["pred_atom_types"] is not None:
                    if batch.batch is None:
                        batch.batch = torch.arange(
                            batch.num_graphs, device=batch.num_atoms.device
                        ).repeat_interleave(batch.num_atoms)
                    # evaluate atom type prediction.
                    pred_atom_types = outputs["pred_atom_types"]
                    target_atom_types = outputs["target_atom_types"]
                    type_accuracy = pred_atom_types.argmax(
                        dim=-1) == (target_atom_types - 1)
                    type_accuracy = scatter(type_accuracy.float(
                    ), batch.batch, dim=0, reduce="mean").mean()
                    log_dict.update({f"{prefix}_type_accuracy": type_accuracy})

        return log_dict, loss

    def logging(self, e):
        print(f"Epoch {e:5d} [GPU{self.device}]:")
        print(f"\tTrain Loss:{self.train_log['train_loss']}")
        if self.val_log:
            print(f"\tVal Loss:{self.val_log['val_loss']}")
        print(f"\tLR:{self.optimizer.param_groups[0]['lr']}")

    def log_dict(self, log_dict, prefix, on_step=False, on_epoch=False, prog_bar=False):
        self.logs[prefix].append(log_dict)

    def clear_log_dict(self):
        for x in self.logs:
            self.logs[x] = []
        self.train_log = []
        self.val_log = []
