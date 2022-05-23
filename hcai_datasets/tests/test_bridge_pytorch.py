
from unittest import TestCase

from hcai_dataset_utils.dataset_iterable import DatasetIterable


class TestBridgePyTorch(TestCase):

    class DummySet(DatasetIterable):

        def __iter__(self):
            pass

        def __next__(self):
            pass

        def get_output_info(self):
            pass

    def test_bridge(self):



