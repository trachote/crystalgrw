import random
from typing import Optional, Sequence, Tuple, Any, List
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

import torch
from torch.utils.data import Dataset, DistributedSampler
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader, DataListLoader

from .data_utils import get_scaler_from_data_list
from . import dataset as get_dataset

def worker_init_fn(id: int):
    """
    DataLoaders workers init function.

    Initialize the numpy.random seed correctly for each worker, so that
    random augmentations between workers and/or epochs are not identical.

    If a global seed is set, the augmentations are deterministic.

    https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    """
    uint64_seed = torch.initial_seed()
    ss = np.random.SeedSequence([uint64_seed])
    # More than 128 bits (4 32-bit words) would be overkill.
    np.random.seed(ss.generate_state(4))
    random.seed(uint64_seed)


class CrystDataModule:
    def __init__(
        self,
        datasets: DictConfig,
        num_workers: DictConfig,
        batch_size: DictConfig,
        dataset: str,
        scaler_path=None,
        training=True,
        run_ddp=False,
    ):
        super().__init__()
        self.datasets = datasets
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.dataset = getattr(get_dataset, dataset)
        self.run_ddp = run_ddp

        self.train_dataset: Optional[Dataset] = None
        self.val_datasets: Optional[Sequence[Dataset]] = None
        self.test_datasets: Optional[Sequence[Dataset]] = None

        # self.get_scaler(scaler_path)

    def prepare_data(self) -> None:
        # download only
        pass

    def get_scaler(self, scaler_path):
        # Load once to compute property scaler
        if scaler_path is None:
            train_dataset = self.dataset(**self.datasets.train)

            self.lattice_scaler = get_scaler_from_data_list(
                train_dataset.cached_data,
                key='scaled_lattice')
            if train_dataset.prop is not None:
                self.scaler = [get_scaler_from_data_list(train_dataset.cached_data,key=prop) for prop in train_dataset.prop]
            else:
                self.scaler = np.array([1])
            # self.scaler = get_scaler_from_data_list(
            #     train_dataset.cached_data,
            #     key=train_dataset.prop)
        else:
            self.lattice_scaler = torch.load(
                Path(scaler_path) / 'lattice_scaler.pt')
            self.scaler = torch.load(Path(scaler_path) / 'prop_scaler.pt')

    def setup(self, training=True):
        """
        construct datasets.
        """
        if training:
            self.train_dataset = self.dataset(**self.datasets.train)
            self.val_datasets = [
                self.dataset(**dataset_cfg)
                for dataset_cfg in self.datasets.val
            ]
            # self.train_dataset.lattice_scaler = self.lattice_scaler
            # self.train_dataset.scaler = self.scaler
            # for val_dataset in self.val_datasets:
            #     val_dataset.lattice_scaler = self.lattice_scaler
            #     val_dataset.scaler = self.scaler

        else:
            self.test_datasets = [
                self.dataset(**dataset_cfg)
                for dataset_cfg in self.datasets.test
            ]
            # for test_dataset in self.test_datasets:
            #     test_dataset.lattice_scaler = self.scaler
            #     test_dataset.scaler = self.scaler

    def train_dataloader(self) -> tuple[DataLoader, DistributedSampler[Any]]:
        sampler = DistributedSampler(self.train_dataset) if self.run_ddp else None
        shuffle = True if not self.run_ddp else False
        return DataLoader(
            self.train_dataset,
            shuffle=shuffle,
            batch_size=self.batch_size.train,
            sampler=sampler,
            num_workers=self.num_workers.train,
            worker_init_fn=worker_init_fn,
        ), sampler

    def val_dataloader(self) -> list[tuple[DataLoader, Any]]:
        samplers = [
            DistributedSampler(dataset, shuffle=False)
            if self.run_ddp else None for dataset in self.val_datasets
        ]
        return [
            (DataLoader(
                dataset,
                # shuffle=False,
                batch_size=self.batch_size.val,
                sampler=sampler,
                num_workers=self.num_workers.val,
                worker_init_fn=worker_init_fn,
            ), sampler)
            for dataset, sampler in zip(self.val_datasets, samplers)
        ]

    def test_dataloader(self) -> list[tuple[DataLoader, Any]]:
        samplers = [
            DistributedSampler(dataset, shuffle=False)
            if self.run_ddp else None for dataset in self.test_datasets
        ]
        return [
            (DataLoader(
                dataset,
                # shuffle=False,
                batch_size=self.batch_size.test,
                sampler=DistributedSampler(dataset, shuffle=False),
                num_workers=self.num_workers.test,
                worker_init_fn=worker_init_fn,
            ), sampler)
            for dataset, sampler in zip(self.test_datasets, samplers)
        ]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.datasets=}, "
            f"{self.num_workers=}, "
            f"{self.batch_size=})"
        )

