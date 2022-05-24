from configparser import ConfigParser

from hcai_dataset_utils.bridge_tf import BridgeTensorflow

import tensorflow as tf

from hcai_datasets.hcai_affectnet.hcai_affectnet_iterable import HcaiAffectnetIterable

if __name__ == "__main__":

    config = ConfigParser()
    config.read("config.ini")

    iterable = HcaiAffectnetIterable(
        dataset_dir=config["directories"]["data_dir"] + "/Affectnet",
        split="train"
    )
    dataset = BridgeTensorflow.make(iterable)

    for i, sample in enumerate(dataset):
        if i > 0:
            break
        print(sample)

    # cast to supervised tuples
    dataset = dataset.map(lambda s: (s["image"], s["expression"]))
    # open files, resize images, one-hot vectors
    dataset = dataset.map(lambda x, y: (
        tf.image.resize(
            tf.image.decode_image(tf.io.read_file(x), channels=3, dtype=tf.uint8, expand_animations=False),
            size=[224, 224]
        ),
        tf.one_hot(y, depth=11)
    ))
    # batch
    dataset = dataset.batch(32, drop_remainder=True)

    for i, sample in enumerate(dataset):
        if i > 0:
            break
        print(sample[0].shape, sample[1].shape)

    efficientnet = tf.keras.applications.EfficientNetB0(
        include_top=True,
        weights=None,
        classes=11,
        classifier_activation="softmax"
    )
    efficientnet.compile(optimizer="adam", loss="categorical_crossentropy")
    efficientnet.fit(dataset, epochs=1)
