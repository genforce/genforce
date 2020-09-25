# python3.7
"""Contains the class of data loader."""

import argparse

from torch.utils.data import DataLoader
from .distributed_sampler import DistributedSampler
from .datasets import BaseDataset


__all__ = ['IterDataLoader']


class IterDataLoader(object):
    """Iteration-based data loader."""

    def __init__(self,
                 dataset,
                 batch_size,
                 shuffle=True,
                 num_workers=1,
                 current_iter=0,
                 repeat=1):
        """Initializes the data loader.

        Args:
            dataset: The dataset to load data from.
            batch_size: The batch size on each GPU.
            shuffle: Whether to shuffle the data. (default: True)
            num_workers: Number of data workers for each GPU. (default: 1)
            current_iter: The current number of iterations. (default: 0)
            repeat: The repeating number of the whole dataloader. (default: 1)
        """
        self._dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self._dataloader = None
        self.iter_loader = None
        self._iter = current_iter
        self.repeat = repeat
        self.build_dataloader()

    def build_dataloader(self):
        """Builds data loader."""
        dist_sampler = DistributedSampler(self._dataset,
                                          shuffle=self.shuffle,
                                          current_iter=self._iter,
                                          repeat=self.repeat)

        self._dataloader = DataLoader(self._dataset,
                                      batch_size=self.batch_size,
                                      shuffle=(dist_sampler is None),
                                      num_workers=self.num_workers,
                                      drop_last=self.shuffle,
                                      pin_memory=True,
                                      sampler=dist_sampler)
        self.iter_loader = iter(self._dataloader)


    def overwrite_param(self, batch_size=None, resolution=None):
        """Overwrites some parameters for progressive training."""
        if (not batch_size) and (not resolution):
            return
        if (batch_size == self.batch_size) and (
                resolution == self.dataset.resolution):
            return
        if batch_size:
            self.batch_size = batch_size
        if resolution:
            self._dataset.resolution = resolution
        self.build_dataloader()

    @property
    def iter(self):
        """Returns the current iteration."""
        return self._iter

    @property
    def dataset(self):
        """Returns the dataset."""
        return self._dataset

    @property
    def dataloader(self):
        """Returns the data loader."""
        return self._dataloader

    def __next__(self):
        try:
            data = next(self.iter_loader)
            self._iter += 1
        except StopIteration:
            self._dataloader.sampler.__reset__(self._iter)
            self.iter_loader = iter(self._dataloader)
            data = next(self.iter_loader)
            self._iter += 1
        return data

    def __len__(self):
        return len(self._dataloader)


def dataloader_test(root_dir, test_num=10):
    """Tests data loader."""
    res = 2
    bs = 2
    dataset = BaseDataset(root_dir=root_dir, resolution=res)
    dataloader = IterDataLoader(dataset=dataset,
                                batch_size=bs,
                                shuffle=False)
    for _ in range(test_num):
        data_batch = next(dataloader)
        image = data_batch['image']
        assert image.shape == (bs, 3, res, res)
        res *= 2
        bs += 1
        dataloader.overwrite_param(batch_size=bs, resolution=res)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Data Loader.')
    parser.add_argument('root_dir', type=str,
                        help='Root directory of the dataset.')
    parser.add_argument('--test_num', type=int, default=10,
                        help='Number of tests. (default: %(default)s)')
    args = parser.parse_args()
    dataloader_test(args.root_dir, args.test_num)
