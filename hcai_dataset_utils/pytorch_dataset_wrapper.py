from torch.utils.data import Dataset as TorchDataset
from torch.utils.data.dataset import T_co
import tensorflow as tf


class PyTorchDatasetWrapper(TorchDataset):
    """
    Adapts an non-batched, finite tensorflow dataset into a pytorch dataset for use with a pytorch dataloader.
    This class can process both dict and tuple output datasets.
    For performance reasons, the wrapper will batch the set internally,
    but will still return single data points as numpy arrays for the dataloader.
    """

    def __init__(self, dataset: tf.data.Dataset, internal_cache_size: int = 32):
        """
        :param dataset: any unbatched, finite tensorflow dataset
        :param internal_cache_size: the size of the internal batch cache
        """
        self._original_ds_reference = dataset
        self._output_dict = False
        self._iter = None
        self._batch_cache = None
        self._batch_cache_size = internal_cache_size
        self._internal_cache_size = 32
        self._configure()

    def _configure(self):

        # is the set dict or tuple based?
        if isinstance(self._original_ds_reference.element_spec, dict):
            self._output_dict = True

        self._original_ds_reference = self._original_ds_reference.batch(
            batch_size=self._internal_cache_size
        )

        # iteratively find total item count, since tf.cardinality is unreliably in this context
        self._len = 0
        self._first_member = 0
        for i, b in enumerate(self._original_ds_reference):
            if self._output_dict and i == 0:
                self._first_member = list(b.keys())[0]
            self._len = self._len + tf.shape(b[self._first_member])[0].numpy()

    def __getitem__(self, index) -> T_co:

        # on epoch start (or error), create tf iterator
        if index == 0 or self._iter is None:
            self._iter = iter(self._original_ds_reference)

        # fetch a new batch if current batch is exhausted
        if (
            index % self._internal_cache_size == 0
            or index % self._internal_cache_size >= self._batch_cache_size
        ):
            if self._output_dict:
                batch = self._iter.get_next()
                self._batch_cache = {k: batch[k].numpy() for k in batch.keys()}
                self._batch_cache_size = tf.shape(
                    self._batch_cache[self._first_member]
                )[0].numpy()
            else:
                self._batch_cache = [field.numpy() for field in self._iter.get_next()]
                self._batch_cache_size = tf.shape(self._batch_cache[0])[0].numpy()

        # return as dict or list of numpy arrays
        if self._output_dict:
            return {
                k: self._batch_cache[k][index % self._batch_cache_size]
                for k in self._batch_cache.keys()
            }
        else:
            return [
                field[index % self._batch_cache_size] for field in self._batch_cache
            ]

    def __len__(self):
        return self._len
