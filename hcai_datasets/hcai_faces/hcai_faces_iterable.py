from pathlib import Path

from hcai_dataset_utils.dataset_iterable import DatasetIterable
import numpy as np


class HcaiFacesIterable(DatasetIterable):

    def __init__(self, *args, dataset_dir=None, **kwargs):
        super(HcaiFacesIterable, self).__init__(*args, **kwargs)
        self.dataset_dir = Path(dataset_dir) / "bilder"
        self._generate_index()
        self._iter = self.yield_samples(self.parsed)

    def __iter__(self):
        return self._iter

    def __next__(self):
        return self._iter.__next__()

    def get_output_info(self):
        return {
            "index": {"dtype": np.uint32, "shape": (1,)},
            "image": {"dtype": np.str, "shape": (1,)},
            "id": {"dtype": np.uint32, "shape": (1,)},
            "age": {"dtype": np.uint32, "shape": (1,)},
            "gender": {"dtype": np.str, "shape": (1,)},
            "emotion": {"dtype": np.uint32, "shape": (1,)},
            "set": {"dtype": np.str, "shape": (1,)},
            "rel_file_path": {"dtype": np.str, "shape": (1,)},
        }

    def _generate_index(self):
        f_names = list(self.dataset_dir.glob("*.jpg"))
        parsed = [n.stem.split("_") for n in f_names]
        self.parsed = zip(f_names, parsed)

    def yield_samples(self, files):
        """Yields examples.
        # Labels are parese using filenames:
        # "ID_ageGroup_gender_emotion_pictureSet.jpg"""
        for f, p in files:
            id, age, gender, emotion, set = p

            yield {
                "index": str(f),
                "image": f,
                "id": int(id),
                "age": age,
                "gender": gender,
                "emotion": emotion,
                "set": set,
                "rel_file_path": str(f.relative_to(f.parents[1])),
            }
