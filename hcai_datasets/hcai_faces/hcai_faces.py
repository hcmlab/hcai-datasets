"""hcai_faces dataset."""

import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
from tensorflow_datasets.core.splits import Split
from pathlib import Path
from hcai_dataset_utils.statistics import Statistics
from hcai_datasets.hcai_faces.hcai_faces_iterable import HcaiFacesIterable

_DESCRIPTION = """
FACES is a set of images of naturalistic faces of 171 young (n = 58), middle-aged (n = 56), and older (n = 57) women and 
men displaying each of six facial expressions: neutrality, sadness, disgust, fear, anger, and happiness. 
The FACES database was developed between 2005 and 2007 by Natalie C. Ebner, Michaela Riediger, 
and Ulman Lindenberger at the Center for Lifespan Psychology, Max Planck Institute for Human Development, Berlin, Germany.
"""

_CITATION = """
@article{ebner2010faces,
  title={FACES—A database of facial expressions in young, middle-aged, and older women and men: Development and validation},
  author={Ebner, Natalie C and Riediger, Michaela and Lindenberger, Ulman},
  journal={Behavior research methods},
  volume={42},
  number={1},
  pages={351--362},
  year={2010},
  publisher={Springer}
}
"""


class HcaiFaces(tfds.core.GeneratorBasedBuilder, HcaiFacesIterable, Statistics):
    """DatasetBuilder for hcai_faces dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def __init__(self, dataset_dir, *args, **kwargs):
        HcaiFacesIterable.__init__(self, *args, dataset_dir=dataset_dir, **kwargs)
        tfds.core.GeneratorBasedBuilder.__init__(self, *args, **kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            metadata=tfds.core.MetadataDict({}),
            features=tfds.features.FeaturesDict(
                {
                    # These are the features of your dataset like images, labels ...
                    "index": tf.string,
                    "image": tfds.features.Image(shape=(3543, 2835, 3)),
                    "id": tf.int64,
                    "age": tfds.features.ClassLabel(names=["y", "o", "m"]),
                    "gender": tfds.features.ClassLabel(names=["m", "f"]),
                    "emotion": tfds.features.ClassLabel(
                        names=["a", "d", "f", "h", "n", "s"]
                    ),
                    "set": tfds.features.ClassLabel(names=["a", "b"]),
                    "rel_file_path": tf.string,
                }
            ),
            supervised_keys=("image", "emotion"),
            homepage="https://dataset-homepage/",
            citation=_CITATION,
        )

    def _populate_meta_data(self, data):
        df = pd.DataFrame(data, columns=["id", "age", "gender", "emotion", "set"])
        self._populate_stats(df)

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        self._populate_meta_data([data for file, data in self.parsed])

        return {
            Split.TRAIN: self._generate_examples(self.parsed),
        }

    def _generate_examples(self, files):
        for sample in self.yield_samples(files):
            yield sample["index"], sample
