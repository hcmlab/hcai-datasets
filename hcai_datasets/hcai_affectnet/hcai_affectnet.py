"""hcai_affectnet dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf
import pandas as pd
import json
import numpy as np
from pathlib import Path
from tensorflow_datasets.core.splits import Split
from hcai_dataset_utils.statistics import Statistics
from hcai_datasets.hcai_affectnet.hcai_affectnet_iterable import HcaiAffectnetIterable

_DESCRIPTION = """
Affectnet is a dataset that has been crawled from the internet and annotated with respect to affective classes as well as valence and arousal.
The annotations also include automatically extracted facial landmarks.
Overall the dataset consists of roughly 1Million images. Half of the images are manually annotated where the other half has been annotated automatically.
Since there is currently no official test set available the validation set is used for testing and a split of the training set is used for validation.
The number of images in test is only 5499 since one corrupt image has been deleted
"""

_CITATION = """
@article{mollahosseini2017affectnet,
  title={Affectnet: A database for facial expression, valence, and arousal computing in the wild},
  author={Mollahosseini, Ali and Hasani, Behzad and Mahoor, Mohammad H},
  journal={IEEE Transactions on Affective Computing},
  volume={10},
  number={1},
  pages={18--31},
  year={2017},
  publisher={IEEE}
}
"""

VERSION = tfds.core.Version("1.0.0")
RELEASE_NOTES = {
    "1.0.0": "Initial release.",
}


class HcaiAffectnetConfig(tfds.core.BuilderConfig, HcaiAffectnetIterable):
    """BuilderConfig for HcaiAffectnetConfig."""

    def __init__(
        self,
        *,
        include_auto=False,
        ignore_duplicate=True,
        ignore_broken=True,
        ignore_unsupported_format=True,
        ignore_lists=None,
        **kwargs
    ):
        """BuilderConfig for HcaiAffectnetConfig.
        Args:
          ignore_duplicate: bool. Flag to determine whether the duplicated files in the dataset should be included. Only affects the training set.
          ignore_broken: bool. Flag to determine whether the broken files in the dataset should be included.Affects all sets.
          ignore_wrong_format:  bool. Flag to determine whether files that are not in tensorflow compatible encoding should be ignored. Affects all sets.
          ignore_lists: list. Custom ignore lists for additional configurations.
          include_auto: bool. Flag to determine whether the automatically annotated files should be included in the dataset.
          **kwargs: keyword arguments forwarded to super.
        """
        super(HcaiAffectnetConfig, self).__init__(version=VERSION, **kwargs)
        if ignore_lists is None:
            ignore_lists = []

        if ignore_duplicate:
            ignore_lists.append("affect_net_ignore_list_duplicates.json")

        if ignore_broken:
            ignore_lists.append("affect_net_broken_files_ignore_list.json")

        if ignore_unsupported_format:
            ignore_lists.append("affect_net_ignore_list_wrong_format.json")

        self.include_auto = include_auto
        self.ignore_lists = ignore_lists


class HcaiAffectnet(tfds.core.GeneratorBasedBuilder, HcaiAffectnetIterable, Statistics):
    """DatasetBuilder for hcai_affectnet dataset."""

    BUILDER_CONFIGS = [
        HcaiAffectnetConfig(
            name="default",
            include_auto=False,
            ignore_duplicate=True,
            ignore_unsupported_format=True,
        ),
        HcaiAffectnetConfig(
            name="inc_auto",
            include_auto=True,
            ignore_duplicate=True,
            ignore_unsupported_format=True,
        ),
        HcaiAffectnetConfig(
            name="emo_net",
            include_auto=False,
            ignore_duplicate=True,
            ignore_unsupported_format=True,
            ignore_lists=['affect_net_emo_net_ignore_list.json']
        ),
    ]

    def __init__(self, *args, dataset_dir, **kwargs):
        tfds.core.GeneratorBasedBuilder.__init__(self, *args, **kwargs)
        if self.builder_config is None:
            HcaiAffectnetIterable.__init__(self, *args, dataset_dir=dataset_dir, **kwargs)
        else:
            HcaiAffectnetIterable.__init__(
                self, *args,
                dataset_dir=dataset_dir,
                include_auto=self.builder_config.include_auto,
                ignore_lists=self.builder_config.ignore_lists,
                **kwargs)

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
                    "image": tfds.features.Image(shape=(None, None, 3)),
                    "expression": tfds.features.ClassLabel(names=self.LABELS),
                    "arousal": tf.float32,
                    "valence": tf.float32,
                    "facial_landmarks": tfds.features.Tensor(
                        shape=(68, 2), dtype=tf.float32
                    ),
                    "rel_file_path": tf.string,
                    #"face_bbox": tf.tuple(),
                }
            ),
            supervised_keys=(
                "image",
                "expression",
            ),
            homepage="http://mohammadmahoor.com/affectnet/",
            citation=_CITATION,
        )

    def _populate_meta_data(self, data: pd.DataFrame):
        """
        Updating the metadata dictionary
        Args:
            data: the dataframe that will be used for calulating the metadata
        """
        for split_name, df in data:
            # Filter only necessary columns
            df_filtered = df.filter(["expression", "arousal", "valence"])

            # Replacing -2 values with nan for the statistics
            df_filtered = df_filtered.applymap(lambda x: np.nan if x == -2 else x)

            # Converting integer to type object to enable correct description
            convert_dict = {"expression": "object"}
            df_filtered = df_filtered.astype(convert_dict)
            self._populate_stats(df_filtered, split_name)

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""

        data = [(Split.TRAIN, self.train_df), (Split.TEST, self.test_df)]

        self._populate_meta_data(data)

        return {
            split_name: self._generate_examples(split_data)
            for split_name, split_data in data
        }

    def _generate_examples(self, split_data):
        for sample in self._yield_examples(split_data):
            yield sample["index"], sample