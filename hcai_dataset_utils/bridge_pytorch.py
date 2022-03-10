from typing import Iterator

from torch.utils.data.dataset import T_co, IterableDataset

import tensorflow as tf

from hcai_dataset_utils.dataset_iterable import DatasetIterable


class BridgePyTorch(IterableDataset):
    """
    Takes a dataset iterable and pytorch transforms
    can be fed directly to a pytorch DataLoader
    """

    def __init__(self, ds_generic: DatasetIterable):
        self._ds = ds_generic
        self._transforms = {}
        self._global_transform = None

    def __iter__(self) -> Iterator[T_co]:
        return self

    def __next__(self):
        data = self._ds.__next__()
        for field in self._transforms.keys():
            data[field] = self._transforms[field](data[field])
        if self._global_transform is not None:
            data = self._global_transform(data)
        return data

    def __getitem__(self, index) -> T_co:
        return self._ds.__getitem__(index)

    def apply_global_transform(self, transform):
        self._global_transform = transform

    def apply_transform(self, field_index, transform):
        self._transforms[field_index] = transform

