from configparser import ConfigParser

from hcai_dataset_utils.bridge_tf import BridgeTensorflow

import tensorflow as tf

from hcai_datasets.hcai_faces.hcai_faces_iterable import HcaiFacesIterable

if __name__ == "__main__":

    config = ConfigParser()
    config.read("config.ini")

    iterable = HcaiFacesIterable(
        dataset_dir=config["directories"]["data_dir"] + "/FACES",
        split="test"
    )
    dataset = BridgeTensorflow.make(iterable)

    for i, sample in enumerate(dataset):
        if i > 0:
            break
        print(sample)

    # cast to supervised tuples
    dataset = dataset.map(lambda s: (s["image"], s["emotion"]))
    # open files, resize images
    dataset = dataset.map(lambda x, y: (
        tf.image.resize(
            tf.image.decode_image(tf.io.read_file(x), channels=3, dtype=tf.uint8, expand_animations=False),
            size=[224, 224]
        ), y
    ))
    # dataset = dataset.map(lambda x, y: (tf.ensure_shape(x, [224, 224, 3]), y))
    # batch
    dataset = dataset.batch(32, drop_remainder=True)

    for i, sample in enumerate(dataset):
        if i > 0:
            break
        print(sample[0].shape, sample[1].shape)

    efficientnet = tf.keras.applications.EfficientNetB0(
        include_top=True,
        weights=None,
        classes=6,
        classifier_activation="softmax"
    )
    efficientnet.compile(optimizer="adam", loss="categorical_crossentropy", metrics="accuracy")
    efficientnet.fit(dataset, epochs=1)
