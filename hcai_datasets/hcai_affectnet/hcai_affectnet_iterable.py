import json

import pandas as pd

from pathlib import Path

from hcai_dataset_utils.dataset_iterable import DatasetIterable

import numpy as np


class HcaiAffectnetIterable(DatasetIterable):

    IMAGE_FOLDER_COL = "image_folder"

    LABELS = [
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

    def __init__(
            self,
            *args,
            dataset_dir=None,
            include_auto=False,
            ignore_duplicate=True,
            ignore_broken=True,
            ignore_unsupported_format=True,
            ignore_lists=None,
            **kwargs
    ):
        """
        Args:
          ignore_duplicate: bool. Flag to determine whether the duplicated files in the dataset should be included. Only affects the training set.
          ignore_broken: bool. Flag to determine whether the broken files in the dataset should be included.Affects all sets.
          ignore_wrong_format:  bool. Flag to determine whether files that are not in tensorflow compatible encoding should be ignored. Affects all sets.
          ignore_lists: list. Custom ignore lists for additional configurations.
          include_auto: bool. Flag to determine whether the automatically annotated files should be included in the dataset.
          **kwargs: keyword arguments forwarded to super.
        """
        super(HcaiAffectnetIterable, self).__init__(*args, **kwargs)
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

        self.dataset_dir = dataset_dir
        self._load_meta_data()

        if self.split == "train":
            self._iter = self._yield_examples(self.train_df)
        else:
            self._iter = self._yield_examples(self.test_df)

    def __iter__(self):
        return self._iter

    def __next__(self):
        return self._iter.__next__()

    def get_output_info(self):
        return {
            "index": {"dtype": np.str, "shape": (1,)},
            "image": {"dtype": np.str, "shape": (1,)},
            "expression": {"dtype": np.uint8, "shape": (1,)},
            "arousal": {"dtype": np.float32, "shape": (1,)},
            "valence": {"dtype": np.float32, "shape": (1,)},
            "facial_landmarks": {"dtype": np.uint32, "shape": (68, 2)},
            "rel_file_path": {"dtype": np.str, "shape": (1,)},
        }

    def _yield_examples(self, label_df):
        for index, row in label_df.iterrows():
            landmarks = np.fromstring(
                row["facial_landmarks"], sep=";", dtype=np.float32
            ).reshape((68, 2))

            yield {
                "index": index,
                "image": str(Path(self.dataset_dir) / row[self.IMAGE_FOLDER_COL] / index),
                "expression": row["expression"],
                "arousal": row["arousal"],
                "valence": row["valence"],
                "facial_landmarks": landmarks,
                "rel_file_path": row[self.IMAGE_FOLDER_COL] + "/" + index
            }

    def _load_meta_data(self):

        print("Loading Labels...")
        train_csv_path = (
                Path(self.dataset_dir) / "Manually_Annotated_file_lists" / "training.csv"
        )

        # The official test set has not been released in this version so we use the validation set as test set
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
        if self.include_auto:
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
        print("Applying ignore lists...")

        filter_list_path = Path(__file__).parent / "Ignore_Lists"
        for filter_list in self.ignore_lists:
            print("... {}:".format(filter_list))
            with open(filter_list_path / filter_list) as json_file:
                filter = json.load(json_file)
                train_df.drop(filter, errors="ignore", inplace=True)
                test_df.drop(filter, errors="ignore", inplace=True)
            print(
                "\t...dropped {} images from train\n\t... dropped {} images from test".format(
                    len_train - len(train_df), len_test - len(test_df)
                )
            )
            len_train = len(train_df)
            len_test = len(test_df)

        print(
            "Final set sizes: \nTrain {}\nTest {}".format(len(train_df), len(test_df))
        )

        self.train_df = train_df
        self.test_df = test_df
