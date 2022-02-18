from abc import ABC


class DatasetIterable(ABC):

    def __init__(self, split: str):
        self.split = str

    def __iter__(self):
        raise NotImplementedError()

    def __next__(self):
        raise NotImplementedError()

    def get_output_types(self):
        raise NotImplementedError()
