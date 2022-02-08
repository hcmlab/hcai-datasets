from abc import ABC


class GenericDataset(ABC):
    def __iter__(self):
        pass

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass
