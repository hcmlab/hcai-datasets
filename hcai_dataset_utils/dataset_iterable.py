from abc import ABC


class DatasetIterable(ABC):

    def __init__(self, split: str):
        if split not in ["test", "train", "val"]:
            split = "train"
        self.split = split

    def __iter__(self):
        raise NotImplementedError()

    def __next__(self):
        raise NotImplementedError()

    def get_output_info(self):
        raise NotImplementedError()
