import os
import hcai_datasets
import tensorflow_datasets as tfds
import warnings
from matplotlib import pyplot as plt

warnings.simplefilter("ignore")

import time

start = time.time()

# Load Data
ds, ds_info = tfds.load(
    "hcai_nova_dynamic",
    split="dynamic_split",
    with_info=True,
    as_supervised=False,
    data_dir=".",
    shuffle_files=False,
    read_config=tfds.ReadConfig(shuffle_seed=1337),
    builder_kwargs={
        # Database Config
        "db_config_path": "C:\\Users\\valentin\\PycharmProjects\\hcai_datasets\\local\\nova_db.cfg",
        "db_config_dict": None,
        # Dataset Config
        "dataset": "kassel_therapie_korpus",
        "nova_data_dir": os.path.join("Z:\\Korpora\\nova\\data"),
        "sessions": ["OPD_101_No"],
        "roles": ["patient", "therapist"],
        "schemes": [],  # ["transcript"],
        "annotator": "system",
        "data_streams": ["video_resized"],
        # Sample Config
        "frame_size": 0.04,
        "left_context": 0,
        "right_context": 0,
        # "start": "0s",
        # "end": "300s",
        "flatten_samples": False,
        # Additional Config
        "lazy_loading": False,
        "clear_cache": True,
    },
)

end = time.time()
print("Elapsed time {}".format(end - start))

data_it = ds.as_numpy_iterator()
data_list = list(data_it)
data_list.sort(key=lambda x: int(x["frame"].decode("utf-8").split("_")[0]))
