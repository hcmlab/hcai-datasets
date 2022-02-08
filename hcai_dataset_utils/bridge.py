from typing import Iterator

from torch.utils.data.dataset import T_co, IterableDataset

import tensorflow as tf

from hcai_dataset_utils.generic_dataset import GenericDataset


class BridgePyTorch(IterableDataset):
    def __init__(self, ds_generic: GenericDataset):
        self._ds = ds_generic

    def __iter__(self) -> Iterator[T_co]:
        return self._ds.__iter__()

    def __getitem__(self, index) -> T_co:
        return self._ds.__getitem__(index)


class BridgeKeras(tf.keras.utils.Sequence):
    def __init__(self, ds_generic: GenericDataset):
        self._ds = ds_generic

    def __len__(self):
        return len(self.indices) // self.batch_size

    def __getitem__(self, index):
        index = self.index[index * self.batch_size : (index + 1) * self.batch_size]
        batch = [self.indices[k] for k in index]

        X, y = self.__get_data(batch)
        return X, y

    def on_epoch_end(self):
        self.index = np.arange(len(self.indices))
        if self.shuffle == True:
            np.random.shuffle(self.index)
