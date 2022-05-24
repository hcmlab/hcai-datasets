"""hcai_ckplus dataset."""
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
from tensorflow_datasets.core.splits import Split
from pathlib import Path
from hcai_dataset_utils.statistics import Statistics
from hcai_datasets.hcai_ckplus.hcai_ckplus_iterable import HcaiCkplusIterable

_DESCRIPTION = """
The Extended Cohn-Kanade (CK+) dataset contains 593 video sequences from a total of 123 different subjects, ranging from 18 to 50 years of age with a variety of genders and heritage.
"""

_CITATION = """
@inproceedings{lucey2010extended,
  title={The extended cohn-kanade dataset (ck+): A complete dataset for action unit and emotion-specified expression},
  author={Lucey, Patrick and Cohn, Jeffrey F and Kanade, Takeo and Saragih, Jason and Ambadar, Zara and Matthews, Iain},
  booktitle={2010 ieee computer society conference on computer vision and pattern recognition-workshops},
  pages={94--101},
  year={2010},
  organization={IEEE}
}
"""


class HcaiCkplus(tfds.core.GeneratorBasedBuilder, HcaiCkplusIterable, Statistics):
    """DatasetBuilder for hcai_ckplus dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def __init__(self, *args, dataset_dir, **kwargs):
        tfds.core.GeneratorBasedBuilder.__init__(self, *args, **kwargs)
        HcaiCkplusIterable.__init__(self, *args, dataset_dir=dataset_dir, **kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            metadata=tfds.core.MetadataDict({}),
            features=tfds.features.FeaturesDict(
                {
                    "index": tf.string,
                    "image": tfds.features.Image(shape=(None, None, 3)),
                    # 0=neutral, 1=anger, 2=contempt, 3=disgust, 4=fear, 5=happy, 6=sadness, 7=surprise)
                    "label": tfds.features.ClassLabel(names=self.LABELS),
                    "rel_file_path": tf.string,
                }
            ),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=("image", "label"),  # Set to `None` to disable
            homepage="https://dataset-homepage/",
            citation=_CITATION,
        )

    def _populate_meta_data(self, data):
        df = pd.DataFrame(data, columns=["file_name", "emotion"])
        df = df.drop(columns="file_name")
        self._populate_stats(df)

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""

        self._populate_meta_data(self.samples)
        return {Split.TRAIN: self._generate_examples(self.samples)}

    def _generate_examples(self, files):
        for sample in self._yield_samples(files):
            yield sample["index"], sample
