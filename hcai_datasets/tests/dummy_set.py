from hcai_dataset_utils.dataset_iterable import DatasetIterable
import numpy as np


class DummySetIterable(DatasetIterable):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._iter = self._yield_samples([
            ("a", 1, 1.0, np.array([1, 1, 1])),
            ("b", 2, 2.0, np.array([2, 2, 2])),
            ("c", 3, 3.0, np.array([3, 3, 3])),
            ("d", 4, 4.0, np.array([4, 4, 4])),
            ("e", 5, 5.0, np.array([5, 5, 5]))
        ])

    def __iter__(self):
        return self._iter

    def __next__(self):
        return self._iter.__next__()

    def get_output_info(self):
        return {
            "field_string": {"dtype": np.str, "shape": (1,)},
            "field_integer": {"dtype": np.int64, "shape": (1,)},
            "field_float": {"dtype": np.float, "shape": (1,)},
            "field_array": {"dtype": np.uint32, "shape": (3,)},
        }

    def _yield_samples(self, data):
        for sample in data:
            yield {
                "field_string": sample[0],
                "field_integer": sample[1],
                "field_float": sample[2],
                "field_array": sample[3]
            }
