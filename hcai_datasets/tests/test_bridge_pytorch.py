from unittest import TestCase

import torch
from torch.utils.data import DataLoader

from hcai_dataset_utils.bridge_pytorch import BridgePyTorch
from hcai_datasets.tests.dummy_set import DummySetIterable


class TestBridgePyTorch(TestCase):

    def test_bridge(self):
        dataset = DummySetIterable()

        loader = DataLoader(BridgePyTorch(dataset))

        for sample in loader:
            self.assertEqual(sample["field_float"].dtype, torch.float64)
            self.assertEqual(sample["field_integer"].dtype, torch.int64)
            self.assertEqual(sample["field_array"].shape[1], 3)
