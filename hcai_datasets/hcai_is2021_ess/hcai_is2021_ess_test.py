"""hcai_is2021_ess dataset."""

import tensorflow_datasets as tfds
from . import hcai_is2021_ess


class HcaiIs2021EssTest(tfds.testing.DatasetBuilderTestCase):
  """Tests for hcai_is2021_ess dataset."""
  # TODO(hcai_is2021_ess):
  DATASET_CLASS = hcai_is2021_ess.HcaiIs2021Ess
  SPLITS = {
      'train': 3,  # Number of fake train example
      'test': 1,  # Number of fake test example
  }

  # If you are calling `download/download_and_extract` with a dict, like:
  #   dl_manager.download({'some_key': 'http://a.org/out.txt', ...})
  # then the tests needs to provide the fake output paths relative to the
  # fake data directory
  # DL_EXTRACT_RESULT = {'some_key': 'output_file1.txt', ...}


if __name__ == '__main__':
  tfds.testing.test_main()
