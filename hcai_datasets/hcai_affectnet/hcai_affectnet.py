"""hcai_affectnet dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf
import pandas as pd
import json
import numpy as np
from pathlib import Path

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

VERSION = tfds.core.Version("4.3.0")
RELEASE_NOTES = {
    "1.0.0": "Initial release.",
}


class HcaiAffectnetConfig(tfds.core.BuilderConfig):
    """BuilderConfig for HcaiAffectnetConfig."""

    def __init__(
        self, *, include_auto=False, ignore_duplicate=True, ignore_lists=None, **kwargs
    ):
        """BuilderConfig for HcaiAffectnetConfig.
        Args:
          include_auto: bool. Flag to determine whether the automatically annotated files should be included in the dataset.
          include_auto: bool. Flag to determine whether the duplicated files in the dataset should be included.
          **kwargs: keyword arguments forwarded to super.
        """
        super(HcaiAffectnetConfig, self).__init__(version=VERSION, **kwargs)
        if ignore_lists is None:
            ignore_lists = []

        if ignore_duplicate:
            ignore_lists.append("affect_net_ignore_list_duplicates.json")
        self.include_auto = include_auto
        self.ignore_lists = ignore_lists


class HcaiAffectnet(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for hcai_affectnet dataset."""

    BUILDER_CONFIGS = [
        HcaiAffectnetConfig(name="default", include_auto=False, ignore_duplicate=True),
        HcaiAffectnetConfig(name="inc_auto", include_auto=True, ignore_duplicate=True),
    ]

    IMAGE_FOLDER_COL = "image_folder"

    def __init__(self, *, dataset_dir=None, **kwargs):
        self.dataset_dir = dataset_dir
        self.labels = [
            "neutral",
            "happy",
            "sad",
            "suprise",
            "fear",
            "disgust",
            "anger",
            "contempt",
            "none",
            "uncertain",
            "non-face",
        ]
        super(HcaiAffectnet, self).__init__(**kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    # These are the features of your dataset like images, labels ...
                    "image": tfds.features.Image(shape=(None, None, 3)),
                    "expression": tfds.features.ClassLabel(names=self.labels),
                    "arousal": tf.float32,
                    "valence": tf.float32,
                    "facial_landmarks": tfds.features.Tensor(
                        shape=(68, 2), dtype=tf.float32
                    ),
                    #'face_bbox': tfds.features.BBox(),
                }
            ),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=(
                "image",
                "expression",
            ),  # Set to `None` to disable
            homepage="https://dataset-homepage/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        print("Loading Labels...")

        train_csv_path = (
            Path(self.dataset_dir) / "Manually_Annotated_file_lists" / "training.csv"
        )
        test_csv_path = (
            Path(self.dataset_dir) / "Manually_Annotated_file_lists" / "validation.csv"
        )

        train_csv_path_auto = (
            Path(self.dataset_dir)
            / "Automatically_Annotated_file_lists"
            / "automatically_annotated.csv"
        )

        train_df = pd.read_csv(train_csv_path, index_col=0)
        test_df = pd.read_csv(test_csv_path, index_col=0)
        train_df[self.IMAGE_FOLDER_COL] = ["Manually_Annotated_Images"] * len(train_df)
        test_df[self.IMAGE_FOLDER_COL] = ["Manually_Annotated_Images"] * len(test_df)


        # append automatic labels if not ignored:
        if self.builder_config.include_auto:
            train_df_auto = pd.read_csv(train_csv_path_auto, index_col=0)
            train_df_auto[self.IMAGE_FOLDER_COL] = ["Automatically_Annotated_Images"]
            train_df = pd.concat([train_df, train_df_auto])

        len_train = len(train_df)
        len_test = len(test_df)
        print(
            "...loaded {} images for train\n...loaded {} images for test".format(
                len_train, len_test
            )
        )

        # removing labels that are specified in the ignore-lists
        print("Apply filtering...")

        filter_list_path = Path(__file__).parent / "Ignore_Lists"
        for filter_list in self._builder_config.ignore_lists:
            with open(filter_list_path / filter_list) as json_file:
                filter = json.load(json_file)
                train_df.drop(filter, errors="ignore", inplace=True)
                test_df.drop(filter, errors="ignore", inplace=True)
        print(
            "...dropped {} images from train\n... dropped {} images from test".format(
                len_train - len(train_df), len_test - len(test_df)
            )
        )

        # split train in validation an train
        print("Splitting validation set")
        val_df = train_df.sample(frac=0.2, random_state=1337)
        train_df = train_df.drop(val_df.index)
        print(
            "Final set sizes: \nTrain {}\nVal {}\n Test{}".format(
                len(train_df), len(val_df), len(test_df)
            )
        )

        return {
            "train": self._generate_examples(train_df),
            "val": self._generate_examples(val_df),
            "test": self._generate_examples(test_df),
        }

    def _generate_examples(self, label_df):
        for index, row in label_df.iterrows():
            yield index, {
                "image": Path(self.dataset_dir) / row[self.IMAGE_FOLDER_COL] / index,
                "expression": row["expression"],
                "arousal": row["arousal"],
                "valence": row["valence"],
                "facial_landmarks": np.fromstring(
                    row["facial_landmarks"], sep=";", dtype=np.float32
                ).reshape((68, 2)),
            }
