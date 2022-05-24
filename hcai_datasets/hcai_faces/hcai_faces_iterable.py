from pathlib import Path

from hcai_dataset_utils.dataset_iterable import DatasetIterable
import numpy as np


class HcaiFacesIterable(DatasetIterable):

    CLASSES_AGE = ["y", "o", "m"]
    CLASSES_GENDER = ["m", "f"]
    CLASSES_EMOTION = ["a", "d", "f", "h", "n", "s"]
    CLASSES_SET = ["a", "b"]

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
            "index": {"dtype": np.str, "shape": (1,)},
            "image": {"dtype": np.str, "shape": (1,)},
            "id": {"dtype": np.uint32, "shape": (1,)},
            "age": {"dtype": np.uint8, "shape": (1,)},
            "gender": {"dtype": np.uint8, "shape": (1,)},
            "emotion": {"dtype": np.uint8, "shape": (1,)},
            "set": {"dtype": np.uint8, "shape": (1,)},
            "rel_file_path": {"dtype": np.str, "shape": (1,)},
        }

    def _generate_index(self):
        f_names = list(self.dataset_dir.glob("*.jpg"))
        parsed = [n.stem.split("_") for n in f_names]
        self.parsed = list(zip(f_names, parsed))

    def yield_samples(self, files):
        """Yields examples.
        # Labels are parsed using filenames:
        # "ID_ageGroup_gender_emotion_pictureSet.jpg"""
        for f, p in files:
            id, age, gender, emotion, set = p

            yield {
                "index": str(f),
                "image": str(f),
                "id": int(id),
                "age": self.CLASSES_AGE.index(age),
                "gender": self.CLASSES_GENDER.index(gender),
                "emotion": self.CLASSES_EMOTION.index(emotion),
                "set": self.CLASSES_SET.index(set),
                "rel_file_path": str(f.relative_to(f.parents[1])),
            }
