"""
This file creates an ignore list that reproduces the test set for the paper Estimation of continuous valence and arousal levels from faces in naturalistic conditions.
In this paper the authors cleaned the test set by manually removing labels they deemed incorrect.
https://github.com/face-analysis/emonet
THIS CODE DOES NOT PERFORM THE RESPECTIVE CLASSMAPPINGS FOR 8 to 5 CLASSES OR CLASS FILTERING!
"""

import pickle
import pandas
import json
import argparse
from pathlib import Path
import math


def build_ignore_list(corpus_dir, out_path_il, test_pickle):

    # Loading emo net test set
    with open(test_pickle, "rb") as f:
        test_data_clean = pickle.load(f)

        len_orig = len(test_data_clean.keys())

        # Removing manually cleaned labels
        test_data_clean = dict(
            filter(lambda elem: elem[1]["expression_correct"], test_data_clean.items())
        )

        len_man = len(test_data_clean.keys())

        # Removing automatically cleaned labels
        def auto_filter(expression, valence, arousal):
            intensity = math.sqrt(valence ** 2 + arousal ** 2)
            if expression == 0 and intensity >= 0.2:
                return False
            elif expression == 1 and (valence <= 0 or intensity <= 0.2):
                return False
            elif expression == 2 and (valence >= 0 or intensity <= 0.2):
                return False
            elif expression == 3 and (arousal <= 0 or intensity <= 0.2):
                return False
            elif expression == 4 and (
                not (arousal >= 0 and valence <= 0) or intensity <= 0.2
            ):
                return False
            elif expression == 5 and (valence >= 0 or intensity <= 0.3):
                return False
            elif expression == 6 and (arousal <= 0 or intensity <= 0.2):
                return False
            elif expression == 7 and (valence >= 0 or intensity <= 0.2):
                return False
            else:
                return True

        test_data_clean = dict(
            filter(
                lambda elem: auto_filter(
                    int(elem[1]["expression"]),
                    float(elem[1]["valence"]),
                    float(elem[1]["arousal"]),
                ),
                test_data_clean.items(),
            )
        )

        len_auto = len(test_data_clean.keys())
        print(
            "Original length: {} \nDropped manual filter: {} \nDropped auto filter {} \nFinal length {}".format(
                len_orig, len_orig - len_man, len_man - len_auto, len_auto
            )
        )

        test_data_clean = list(test_data_clean.keys())

    # Loading original affect net validation set -> validation set ist used as official test set in affectnet
    val_data_orig_path = (
        Path(corpus_dir) / "Manually_Annotated_file_lists" / "validation.csv"
    )
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
