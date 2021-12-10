"""
This file creates an ignore list that reproduces the test set for the paper Estimation of continuous valence and arousal levels from faces in naturalistic conditions.
In this paper the authors cleaned the test set by manually removing labels they deemed incorrect.
https://github.com/face-analysis/emonet
"""

import pickle
import pandas
import json
import argparse
from pathlib import Path

def build_ignore_list(corpus_dir, out_path_il, test_pickle):

    # Loading emo net test set
    with open(test_pickle, "rb") as f:
        test_data_clean = list(pickle.load(f).keys())

    # Loading original affect net validation set -> validation set ist used as official test set in affectnet
    val_data_orig_path = Path(corpus_dir) / 'Manually_Annotated_file_lists' / 'validation.csv'
    val_data_orig = pandas.read_csv(val_data_orig_path)
    val_data_orig = val_data_orig["#subDirectory_filePath"].to_list()

    # Removing all labels that are in both lists to create ignore list
    ignore_list = [x for x in val_data_orig if x not in test_data_clean]

    # Saving
    with open(out_path_il, "w") as fp:
        json.dump(ignore_list, fp, sort_keys=True, indent=4)


if __name__ == "__main__":
    # Parse Arguments
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument("--out_path_il", help="Path to store the final ignore list")
    my_parser.add_argument(
        "--test_pickle",
        help="Path to the emo_net_test_fullpath.pkl of the emonet repository",
    )
    my_parser.add_argument(
        "--corpus_dir", help="Path to the affect net corpus directory"
    )
    args = my_parser.parse_args()

    build_ignore_list(args.corpus_dir, args.out_path_il, args.test_pickle)
