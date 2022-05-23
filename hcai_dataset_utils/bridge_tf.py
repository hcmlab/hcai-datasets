import tensorflow as tf
import numpy as np

from hcai_dataset_utils.dataset_iterable import DatasetIterable


class BridgeTensorflow:
    """
    Provides a helper function to make a tf Dataset directly from a DatasetIterable
    """

    TYPE_MAPPING = {
        np.str: tf.string,
        np.int: tf.int32,
        np.int8: tf.int8,
        np.int32: tf.int32,
        np.int64: tf.int64,
        np.uint: tf.uint32,
        np.uint8: tf.uint8,
        np.uint32: tf.uint32,
        np.uint64: tf.uint64,
        np.float: tf.float32,
        np.float32: tf.float32,
        np.float64: tf.float32,
    }

    @staticmethod
    def make(ds: DatasetIterable):
        iter = ds.__iter__

        info = ds.get_output_info()
        output_types = {
            **{k: BridgeTensorflow.TYPE_MAPPING[info[k]["dtype"]] for k in info.keys()}
        }

        return tf.data.Dataset.from_generator(iter, output_types=output_types)
