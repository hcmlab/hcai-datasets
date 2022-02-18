import tensorflow as tf
import numpy as np

from hcai_dataset_utils.dataset_iterable import DatasetIterable


class BridgeTensorflow:
    TYPE_MAPPING = { #TODO complete using docs
        np.str: tf.string,
        np.int: tf.int32,
        np.int32: tf.int32,
        np.int64: tf.int64,
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
