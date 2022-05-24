from pathlib import Path

from hcai_dataset_utils.dataset_iterable import DatasetIterable
import numpy as np


class HcaiCkplusIterable(DatasetIterable):
    LABELS = [
        "neutral",
        "anger",
        "contempt",
        "disgust",
        "fear",
        "happy",
        "sadness",
        "suprise",
    ]

    def __init__(self, *args, dataset_dir=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_dir = Path(dataset_dir)
        self._build_index()
        self._iter = self._yield_samples(self.samples)

    def __iter__(self):
        return self._iter

    def __next__(self):
        return self._iter.__next__()

    def get_output_info(self):
        return {
            "index": {"dtype": np.str, "shape": (1,)},
            "image": {"dtype": np.str, "shape": (1,)},
            "label": {"dtype": np.uint8, "shape": (1,)},
            "rel_file_path": {"dtype": np.str, "shape": (1,)},
        }

    def _build_index(self):

        emo_anno_files = list(self.dataset_dir.glob("Emotion/**/**/*.txt"))
        samples = []

        for ef in emo_anno_files:
            with open(ef, "r") as f:
                emotion = self.LABELS[int(float(f.readline().strip()))]
                fn_img = ef.stem.replace("_emotion", ".png")
                samples.append((fn_img, emotion))

                # add neutral images
                fn_img_neut = ef.stem.rsplit("_", 2)[0] + "_00000001.png"
                samples.append((fn_img_neut, "neutral"))
        self.samples = samples

    def _yield_samples(self, files):
        """Yields examples."""

        for f, e in files:
            rel_path = (Path(*f.split("_")[:-1]) / f).resolve()
            yield {
                "index": str(f),
                "image": str(self.dataset_dir / "cohn-kanade-images" / rel_path),
                "label": self.LABELS.index(e),
                "rel_file_path": str(rel_path),
            }
