from abc import ABC

from hcai_dataset_utils.dataset_iterable import DatasetIterable


class DatasetIndexed(ABC, DatasetIterable):

    def __getitem__(self, item):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()
