import numpy as np
import math
from copy import copy
from omegaconf import ListConfig
from functools import partial

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Uniform

from torch_scatter import scatter_mean
from ..gnn.embeddings import MAX_ATOMIC_NUM
from ..common.data_utils import lattice_params_from_matrix
from ..common.manifolds import Torus3d, Euclid3d, Hypercube

from ..common import DTYPE
from numpy import pi as PI


class BaseGRW(nn.Module):
    def __init__(self):
        super().__init__()
        self.forward_algo = None
        self.manifolds = {}
        self.b_m = {}
        self.sample_type_method = None

    @staticmethod
    def expand_dim(x, target, num_atoms=None):
        size0 = len(x.shape)
        size1 = len(target.shape)
        if size0 < size1:
            for _ in range(size1 - size0):
                x = x.unsqueeze(-1)
        if num_atoms is not None:
            x = x.repeat_interleave(num_atoms, dim=0)
        return x

    def get_params(self, x, n, b_s, tan_vec, score=0.):
        b = 0  # for a compact manifold
        b = -b + score
        b_0, b_f = b_s
        if isinstance(b, torch.Tensor):
            b = self.expand_dim(b, tan_vec)
        if isinstance(b_f, ListConfig):
            b_f = torch.zeros_like(tan_vec)
            for i, b_fi in enumerate(b_f):
                b_f[:, i] += b_fi
        if self.forward_algo == "skip":
            sigma = (b_0 * n + 0.5 * (b_f - b_0) * (n ** 2)).sqrt()
        else:
            sigma = (b_0 + (b_f - b_0) * n).sqrt()
        return b, sigma

    def random_walk(self, x, t, n, manifold, b_s, score, num_atoms=None, coeff=1):
        tan_vec = manifold.get_tangent(x)
        eps = torch.randn_like(x)
        eps = self.expand_dim(eps, tan_vec)
        manifold.eps = eps
        t = self.expand_dim(t, tan_vec, num_atoms) * coeff
        n = self.expand_dim(n, tan_vec, num_atoms)
        b, sigma = self.get_params(x, n, b_s, tan_vec, score)
        W = t * b + t.sqrt() * sigma * eps
        W = W * tan_vec
        return manifold.exp(W, x).detach()

    def grw(self, x_0, T, N,
            num_atoms,
            score_fn=None,
            stack_data=False,
            adaptive_timestep=1,
            progress_bar=None,
            ):
        t = (T / N)
        x_t = copy(x_0)
        x_all = []

        for k in range(N):
            if stack_data:
                x_all.append(copy(x_t))

            if score_fn is not None:
                n = T - k * t
                p = t * (N - k) ** adaptive_timestep
                scores = score_fn(t=n, **self.retransform(x_t))
                scores = self.convert_score(x_t, scores, n, num_atoms)
            else:
                if (N == 1) and (self.forward_algo == "skip"):
                    n = T
                    p = torch.ones_like(T)
                else:
                    n = k * t
                    p = t
                scores = {f: 0 for f in x_t}

            for f in self.manifolds:
                x_t[f] = self.random_walk(x_t[f], p, n,
                                          self.manifolds[f], self.b_m[f], scores[f],
                                          num_atoms=num_atoms if f != "lattices" else None)

            if progress_bar:
                progress_bar.update(1)

        if stack_data:
            x_all.append(copy(x_t))
            return x_all
        else:
            return x_t

    def pushforward(self, x_0, x_t):
        x_inv = {}
        for f in self.manifolds:
            x_inv[f] = self.manifolds[f].log(x_0[f], x_t[f])
        return x_inv

    def convert_score(self, x, scores, t, num_atoms):
        if "atom_types" in self.manifolds:
            f = "atom_types"
            scores[f] = self.manifolds[f].simp_to_hpc(scores[f].squeeze(1))
            scores[f] = (self.manifolds[f].log(scores[f], x[f]) /
                         t.repeat_interleave(num_atoms, dim=0).unsqueeze(-1))
        if "lattices" in self.manifolds:
            scores["lattices"] = scores["lattices"].view(-1, 3, 3)
        return scores

    def transform(self, x):
        if "atom_types" in self.manifolds:
            f = "atom_types"
            x[f] = F.one_hot(x[f] - 1,
                             num_classes=MAX_ATOMIC_NUM
                             ).to(DTYPE)
            x[f] = self.manifolds[f].simp_to_hpc(x[f])
        return x

    def retransform(self, x, sample_type_method, embed_noisy_types=False):
        x = copy(x)
        if "atom_types" in self.manifolds:
            f = "atom_types"
            if embed_noisy_types:
                x["noisy_atom_types"] = x[f].clone()
            if sample_type_method == "force_atom_types":
                pass
            else:
                x[f] = self.manifolds[f].simp_from_hpc(x[f])
                if sample_type_method == "multinomial":
                    x[f] = x[f].multinomial(num_samples=1).squeeze(1) + 1
                elif sample_type_method == "argmax":
                    x[f] = x[f].argmax(dim=-1) + 1
        return x

    def forward(self, x_0, t, num_atoms, N=None, score_fn=None,
                stack_data=False, adaptive_timestep=1, progress_bar=None,
                sample_type_method="multinomial", embed_noisy_types=False):
        if (score_fn is None) and (self.forward_algo == "skip"):
            N = 1
        elif N is None:
            N = self.N

        self.retransform = partial(
            self.retransform,
            sample_type_method=sample_type_method,
            embed_noisy_types=embed_noisy_types
        )

        x_0 = self.transform(x_0)
        x_t = self.grw(x_0, T=t, N=N,
                       num_atoms=num_atoms,
                       score_fn=score_fn,
                       stack_data=stack_data,
                       adaptive_timestep=adaptive_timestep,
                       progress_bar=progress_bar,
                       )

        if not stack_data:
            x_inv = self.pushforward(x_0, x_t)
            x_t = self.retransform(x_t)
        else:
            x_inv = [self.pushforward(x_0, x) for x in x_t]
            x_t = [self.retransform(x) for x in x_t]
        return x_t, x_inv


class GRW(BaseGRW):
    def __init__(self, max_time, timesteps, corrupt_coords=True, corrupt_lattices=True,
                 corrupt_types=True, forward_algo="skip", **kwargs):
        super().__init__()
        self.T = max_time
        self.N = timesteps
        self.forward_algo = forward_algo
        self.corrupt_coords = corrupt_coords
        self.corrupt_lattices = corrupt_lattices
        self.corrupt_types = corrupt_types
        self.manifolds = self.get_manifolds()
        self.b_m = self.get_b_params(**kwargs)

    def get_manifolds(self):
        manifolds = {}
        if self.corrupt_coords:
            manifolds["frac_coords"] = Torus3d()

        if self.corrupt_lattices:
            manifolds["lattices"] = Euclid3d()

        if self.corrupt_types:
            manifolds["atom_types"] = Hypercube(MAX_ATOMIC_NUM - 1)

        return manifolds

    def get_b_params(self, b0_coord=1e-4, bf_coord=1, b0_lattice=1e-3, bf_lattice=20,
                     b0_type=1e-6, bf_type=5, **kwargs):
        b_m = {}
        if self.corrupt_coords:
            b_m["frac_coords"] = (b0_coord, bf_coord)

        if self.corrupt_lattices:
            b_m["lattices"] = (b0_lattice, bf_lattice)

        if self.corrupt_types:
            b_m["atom_types"] = (b0_type, bf_type)

        return b_m
