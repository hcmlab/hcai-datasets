import pickle
import shutil
from pathlib import Path

import pandas as pd
import numpy as np
from PIL import Image
from torch.distributions import Transform
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
import torch
from torchvision import io


class AffectNetPyTorch(Dataset):
    def __init__(
        self, dataset_dir, split="train", transform=None, filters=None, cache_dir=None
    ):
        self._dataset_dir = dataset_dir
        self._split = split
        self._transform = transform
        self._filters = filters
        self._cache_dir = cache_dir
        if cache_dir is not None:
            if self._find_cache():
                self._load_cached()
            else:
                self._index_data_from_source()
                self._build_cache()
        else:
            self._index_data_from_source()

        if self._filters is not None:
            self._filter_indices()

    def _find_cache(self):
        return (Path(self._cache_dir) / (self._split + ".dat")).is_file()

    def _load_cached(self):

        with open(Path(self._cache_dir) / (self._split + ".dat"), "rb") as file:
            self._indices = pickle.load(file)

        self._dataset_dir = Path(self._cache_dir) / self._split

    def _build_cache(self):

        split_path = Path(self._cache_dir) / self._split
        ds_dir = Path(self._dataset_dir)

        print("building cache...")
        n = len(self._indices)

        for i, sample in enumerate(self._indices):
            t = split_path / sample["rel_file_path"]
            t.parent.mkdir(exist_ok=True, parents=True)
            shutil.copy(ds_dir / sample["rel_file_path"], t)
            if i % 100 == 0:
                print(f"[{i}|{n}]")

        with open(Path(self._cache_dir) / (self._split + ".dat"), "wb") as file:
            pickle.dump(self._indices, file)

        self._dataset_dir = split_path

    def _filter_indices(self):
        i = []
        for s in self._indices:
            retain = True
            for fn in self._filters:
                if not fn(s):
                    retain = False
                    break
            if retain:
                i.append(s)
        self._indices = i

    def _index_data_from_source(self):

        if self._split == "train":
            filepath = (
                Path(self._dataset_dir)
                / "Manually_Annotated_file_lists"
                / "training.csv"
            )
            filepath_auto = (
                Path(self._dataset_dir)
                / "Automatically_Annotated_file_lists"
                / "training.csv"
            )
        else:
            filepath = (
                Path(self._dataset_dir)
                / "Manually_Annotated_file_lists"
                / "validation.csv"
            )
            filepath_auto = None

        self._indices = []
        self._parse_index_file(filepath, "Manually_Annotated_Images", True)
        if filepath_auto is not None:
            self._parse_index_file(
                filepath_auto, "Automatically_Annotated_Images", False
            )

    def _parse_index_file(self, filepath, imdir, manual):

        df = pd.read_csv(filepath)
        for row in df.iloc:
            landmarks = np.fromstring(
                row.facial_landmarks, sep=";", dtype=np.float32
            ).reshape((68, 2))

            prep = {
                "rel_file_path": imdir + "/" + row.iloc[0],
                "expression": row.expression,
                "arousal": row.arousal,
                "valence": row.valence,
                "facial_landmarks": landmarks,
                "manual": manual
                # "face_bbox": tfds.features.BBox(ymin=ymin, xmin=xmin, ymax=ymax, xmax=xmax)
            }
            self._indices.append(prep)

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, index) -> T_co:

        row = self._indices[index]
        im = Image.open(Path(self._dataset_dir) / row["rel_file_path"])

        if self._transform is not None:
            im = self._transform(im)

        return {
            "image": im,
            "expression": row["expression"],
            "arousal": row["arousal"],
            "valence": row["valence"],
            "facial_landmarks": row["facial_landmarks"],
        }
