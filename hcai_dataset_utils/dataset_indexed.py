from abc import ABC

from hcai_dataset_utils.dataset_iterable import DatasetIterable


class DatasetIndexed(ABC, DatasetIterable):
    """
    For Datasets which are not only iterable, but also support random access and length
    """

    def __getitem__(self, item):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()
