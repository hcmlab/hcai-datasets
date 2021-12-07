from tensorflow.python.data import Dataset
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data.dataset import T_co
import torch
import tensorflow as tf
import numpy as np


class PyTorchDatasetWrapper(TorchDataset):
    def __init__(
        self,
        dataset: Dataset,
        load_to_device: bool = False,
    ):
        self._original_ds_reference = dataset
        self._iter = None
        self._output_dict = False

        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")

        if load_to_device:
            self._load_to_device()
        else:
            self._tensor_representation = None
            self._len = tf.data.experimental.cardinality(dataset).numpy()
            if self._len < 0:
                self._find_length()

    def _load_to_device(self):

        self._len = 0
        # process dict type sets
        if isinstance(self._original_ds_reference.element_spec, dict):
            self._output_dict = True
            self._tensor_representation = {}
            for k in self._original_ds_reference.element_spec.keys():
                if self._original_ds_reference.element_spec[k].dtype != tf.string:
                    self._tensor_representation[k] = []

            for data in self._original_ds_reference:
                for k in self._tensor_representation.keys():
                    as_tensor = torch.tensor(data[k].numpy()).to(self._device)
                    self._tensor_representation[k].append(as_tensor)
                self._len = self._len + 1
        # process tuple sets, typically x,y for classifiers
        else:
            self._tensor_representation = None
            for data in self._original_ds_reference:
                if self._tensor_representation is None:
                    self._tensor_representation = [[] for i in data]
                for i, field in enumerate(data):
                    as_tensor = torch.tensor(field.numpy()).to(self._device)
                    self._tensor_representation[i].append(as_tensor)
                self._len = self._len + 1

    def _find_length(self):
        self._len = 0
        for i in self._original_ds_reference:
            self._len = self._len + 1

    def __getitem__(self, index) -> T_co:
        # preprocessed as gpu tensor
        if self._tensor_representation is not None:
            if self._output_dict:
                r = {}
                for k in self._tensor_representation.keys():
                    r[k] = self._tensor_representation[k][index]
                return r
            else:
                return [i[index] for i in self._tensor_representation]

        # return dynamically from tf pipeline
        if index == 0 or self._iter is None:
            self._iter = iter(self._original_ds_reference)
        data = self._iter.get_next()

        if self._output_dict:
            r = {}
            for k in data.keys():
                # todo catch string types
                r[k] = torch.tensor(data[index]).to(self._device)
        else:
            return [torch.tensor(field.numpy()).to(self._device) for field in data]

    def __len__(self):
        return self._len
