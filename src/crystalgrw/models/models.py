import numpy as np
from tqdm import tqdm
from functools import partial

import torch
import torch.nn as nn
from torch.nn import functional as F

from ..common.data_utils import (
    lattice_params_to_matrix_torch,
    lattice_params_from_matrix,
)

from ..gnn.embeddings import MAX_ATOMIC_NUM
from .base import BaseModel


def build_mlp(in_dim, hidden_dim, fc_num_layers, out_dim):
    mods = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for i in range(fc_num_layers - 1):
        mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    mods += [nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*mods)


class CrystalGRW(BaseModel):
    def __init__(self, encoder, sde_fn, score_fn, control_fn, cfg, **kwarg):
        super().__init__(cfg)
        self.cfg = cfg
        self.hparams = cfg.model
        self.hparams.data = cfg.data
        self.hparams.algo = "crystalgrw"
        self.model_name = "crystalgrw"
        self.logs = {'train': [], 'val': [], 'test': []}
        self.T = self.hparams.num_noise_level
        self.vae = cfg.model.vae

        self.encoder = encoder
        self.sde_fn = sde_fn
        self.score_fn = score_fn
        self.control_fn = control_fn

        if hasattr(self.hparams, "uncond_prob"):
            self.uncond_prob = self.hparams.uncond_prob

        if self.vae:
            self.fc_mu = nn.Linear(self.hparams.latent_dim,
                                   self.hparams.latent_dim)
            self.fc_var = nn.Linear(self.hparams.latent_dim,
                                    self.hparams.latent_dim)

    def forward(self, batch, *args, **kwargs):
        if batch.batch is None:
            batch_idx = torch.arange(
                batch.num_graphs, device=batch.num_atoms.device
            ).repeat_interleave(batch.num_atoms)
        else:
            batch_idx = batch.batch
        atom_types = batch.atom_types
        natoms = batch.num_atoms
        frac_coords = batch.frac_coords
        lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)

        # Encoder
        if self.encoder is not None:
            z = self.encoder(batch)
            if self.vae:
                mu, log_var, z = self.kld_reparam(z)
        else:
            z = None

        # Corrupt features
        t = torch.randint(1, self.T, size=(natoms.size(0),),
                          device=natoms.device) / self.T
        x_0 = {"frac_coords": frac_coords,
               "lattices": lattices,
               "atom_types": atom_types,
               }

        x_t, x_inv = self.sde_fn(x_0, t, natoms)

        # Embed conditions
        if self.control_fn is not None:
            try:
                condition = self.get_condition(batch.y, natoms)
            except Exception as e:
                raise Exception(e)
        else:
            condition = None

        # Get scores
        scores = self.score_fn(**x_t, t=t, z=z,
                               natoms=natoms,
                               cond_feat=condition,
                               batch=batch_idx,
                               noisy_atom_types=x_t["atom_types"],
                               )

        # Compute losses
        losses = {f: 0 for f in x_0}

        for f in scores:
            if (f == "frac_coords") and self.hparams.corrupt_coords:
                if self.hparams.loss_type == "Varadhan":
                    x_inv["frac_coords"] = (
                            x_inv["frac_coords"] /
                            t.repeat_interleave(natoms, dim=0).unsqueeze(-1)
                    )
                losses["frac_coords"] += self.l2_loss(
                    scores["frac_coords"], x_inv["frac_coords"], batch_idx, norm=True
                )
            elif (f == "lattices") and self.hparams.corrupt_lattices:
                if self.hparams.loss_type == "Varadhan":
                    x_inv["lattices"] = (
                            x_inv["lattices"].view(-1, 9) / t.unsqueeze(-1)
                    )
                losses["lattices"] = self.l2_loss(
                    scores["lattices"].view(-1, 9), x_inv["lattices"], norm=True
                )
            elif (f == "atom_types") and self.hparams.corrupt_types:
                losses["atom_types"] = self.l2_loss(
                    scores["atom_types"],
                    F.one_hot(atom_types - 1, num_classes=MAX_ATOMIC_NUM),
                    batch_idx, norm=True
                )

            elif (f == "energy") and self.score_fn.regress_energy:
                if hasattr(batch, "energy"):
                    losses["energy"] = self.l2_loss(
                        scores["energy"], batch.energy, norm=True
                    )
                else:
                    losses["energy"] = 0

        if self.vae and (self.encoder is not None):
            losses["kld"] = self.kld_loss(mu, log_var)
        else:
            losses["kld"] = 0

        return {
            "losses": losses,
            "pred_atom_types": scores["atom_types"] if "atom_types" in scores else None,
            "target_atom_types": atom_types,
        }
