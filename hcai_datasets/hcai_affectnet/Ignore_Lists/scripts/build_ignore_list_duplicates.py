import os
import json
from PIL import Image, ImageStat
import random
import argparse
from pathlib import Path
import csv

random.seed(1337)

def flatten(t):
    return [item for sublist in t for item in sublist]

def hash_image(image_path):
    img = Image.open(image_path).resize((8, 8), Image.LANCZOS).convert(mode="L")
    mean = ImageStat.Stat(img).mean[0]
    return sum((1 if p > mean else 0) << i for i, p in enumerate(img.getdata()))


def build_ignore_list(out_path_dpl, out_path_il, corpus_path):

    ignore_list = []
    duplicate_dict = {}
    counter = 0

    if os.path.exists(out_path_dpl):
        duplicate_dict = json.load(open(os.path.join(out_path_dpl)))

    else:
        image_folder_path = Path(corpus_path) / 'Manually_Annotated_Images'
        exclude_extensions = ["db"]
        for root, dir, filenames in os.walk(image_folder_path):
            for f in filenames:
                if f.split(".")[-1] in exclude_extensions:
                    continue
                if counter % 1000 == 0:
                    print("hashing image {}".format(counter))
                counter += 1
                path = os.path.join(root, f)
                hash = hash_image(path)
                if hash not in duplicate_dict.keys():
                    duplicate_dict[hash] = [root.split(os.sep)[-1] + "/" + f]
                else:
                    duplicate_dict[hash].append(root.split(os.sep)[-1] + "/" + f)

        with open(out_path_dpl, "w") as fp:
            json.dump(duplicate_dict, fp, sort_keys=True, indent=4)
            exit()

    dupl = [x for x in duplicate_dict.values() if len(x) > 1]

    # Do not touch validation samples
    print('n duplicates: {}'.format(len(flatten(dupl))))
    validation_samples = []
    print('Removing validation samples')
    with open(Path(corpus_path) / 'Manually_Annotated_file_lists' / 'validation.csv', 'r') as validation_file:
        labels = csv.reader(validation_file, delimiter=',')
        validation_samples = [x[0] for x in list(labels)]

    dupl = [list(filter(lambda x: x not in validation_samples, d)) for d in dupl]
    print('n duplicates: {}'.format(len(flatten(dupl))))


    # Keeping one random image
    for img_list in dupl:
        random.shuffle(img_list)
        for i in range(len(img_list) - 1):
            ignore_list.append(img_list[i])
    with open(out_path_il, "w") as fp:
        json.dump(ignore_list, fp, sort_keys=True, indent=4)


if __name__ == "__main__":
    # Parse Arguments
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument("--out_path_dpl", help="Path to store the caluclated hashes")
    my_parser.add_argument("--out_path_il", help="Path to store the final ignore list")
    my_parser.add_argument(
        "--coprpus_dir", help="Path to the affect net corpus directory"
    )
    args = my_parser.parse_args()

    build_ignore_list(args.out_path_dpl, args.out_path_il, args.coprpus_dir)
