from abc import ABC


class DatasetIterable(ABC):
    """
    Abstract superclass for iterable Datasets, which can be fed into the Bridge classes for tensorflow and pytorch
    """

    def __init__(self, *args, split: str = "train", **kwargs):
        if split not in ["test", "train", "val"]:
            split = "train"
        self.split = split

    def __iter__(self):
        raise NotImplementedError()

    def __next__(self):
        raise NotImplementedError()

    def get_output_info(self):
        raise NotImplementedError()
