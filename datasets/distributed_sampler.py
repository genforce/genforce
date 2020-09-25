# python3.7
"""Contains the distributed data sampler.

This file is mostly borrowed from `torch/utils/data/distributed.py`.

However, sometimes, initialize the data loader and data sampler can be time
consuming (since it will load a large amount of data at one time). To avoid
re-initializing the data loader again and again, we modified the sampler to
support loading the data for only one time and then repeating the data loader.
Please use the class member `repeat` to control how many times you want the
data load to repeat. After `repeat` times, the data will be re-loaded.

NOTE: The number of repeat times should not be very large, especially when there
are too many samples in the dataset. We recommend to set `repeat = 500` for
datasets with ~50K samples.
"""

# pylint: disable=line-too-long

import math
from typing import TypeVar, Optional, Iterator

import torch
from torch.utils.data import Sampler, Dataset
import torch.distributed as dist


T_co = TypeVar('T_co', covariant=True)


class DistributedSampler(Sampler):
    r"""Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class:`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`rank` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.
        current_iter (int, optional): Number of current iteration. Default: ``0``.
        repeat (int, optional): Repeating number of the whole dataloader. Default: ``1000``.

    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    """

    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False, current_iter: int = 0,
                 repeat: int = 1000) -> None:
        super().__init__(None)
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.iter = current_iter
        self.drop_last = drop_last

        # NOTE: self.dataset_length is `repeat X len(self.dataset)`
        self.repeat = repeat
        self.dataset_length = len(self.dataset) * self.repeat

        if self.drop_last and self.dataset_length % self.num_replicas != 0:
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (self.dataset_length - self.num_replicas) / self.num_replicas
            )
        else:
            self.num_samples = math.ceil(self.dataset_length / self.num_replicas)


        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.__generate_indices__()

    def __generate_indices__(self) -> None:
        g = torch.Generator()
        indices_bank = []
        for iter_ in range(self.iter, self.iter + self.repeat):
            g.manual_seed(self.seed + iter_)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
            indices_bank.extend(indices)
        self.indices = indices_bank

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on iter and seed
            indices = self.indices
        else:
            indices = list(range(self.dataset_length))

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            indices += indices[:(self.total_size - len(indices))]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def __reset__(self, iteration: int) -> None:
        self.iter = iteration
        self.__generate_indices__()

# pylint: enable=line-too-long
