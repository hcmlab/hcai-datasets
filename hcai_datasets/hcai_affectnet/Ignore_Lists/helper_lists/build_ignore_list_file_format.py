"""
https://www.tensorflow.org/datasets/api_docs/python/tfds/features/Image
Tensorflow datasets only allow encoding of images in the format of BMP, JPEG, and PNG (and GIFs with multiple frames).
We ignore the rest.
"""

import os
import argparse
from pathlib import Path
import pandas as pd
import json


def build_ignore_list(corpus_dir, out_path_il):

    valid_ext = [".bmp", ".jpg", ".jpeg", ".png"]
    csv_train_auto = (
        Path(corpus_dir)
        / "Automatically_Annotated_file_list"
        / "automatically_annotated.csv"
    )
    csv_train_man = Path(corpus_dir) / "Manually_Annotated_file_lists" / "training.csv"
    csv_val_man = Path(corpus_dir) / "Manually_Annotated_file_lists" / "validation.csv"
    ignore_list = []

    for csv_file in [csv_val_man, csv_train_auto, csv_train_man]:
        df = pd.read_csv(csv_file, usecols=[0])
        s = df["#subDirectory_filePath"]
        s_keep = s[s.apply(lambda x: os.path.splitext(x)[1].lower() in valid_ext)]
        s_rmv = s[s.apply(lambda x: os.path.splitext(x)[1].lower() not in valid_ext)]

        assert len(s_keep) + len(s_rmv) == len(s)
        ignore_list.extend( s_rmv.tolist() )
        print("")

    with open(out_path_il, "w") as fp:
        json.dump(ignore_list, fp, sort_keys=True, indent=4)


if __name__ == "__main__":
    # Parse Arguments
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument("--out_path_il", help="Path to store the final ignore list")
    my_parser.add_argument(
        "--corpus_dir", help="Path to the affect net corpus directory"
    )
    args = my_parser.parse_args()

    build_ignore_list(args.corpus_dir, args.out_path_il)
