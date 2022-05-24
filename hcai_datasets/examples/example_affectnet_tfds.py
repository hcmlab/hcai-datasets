from configparser import ConfigParser

import tensorflow as tf
import tensorflow_datasets as tfds
import hcai_datasets

if __name__ == "__main__":
    config = ConfigParser()
    config.read("config.ini")

    dataset, dataset_info = tfds.load(
        'hcai_affectnet',
        split='train',
        with_info=True,
        as_supervised=True,
        builder_kwargs={'dataset_dir': config["directories"]["data_dir"] + "/Affectnet"}
    )

    # resize images, one-hot vectors
    dataset = dataset.map(lambda x, y: (
        tf.image.resize(x, size=[224, 224]),
        tf.one_hot(y, depth=11)
    ))
    # batch
    dataset = dataset.batch(32, drop_remainder=True)

    efficientnet = tf.keras.applications.EfficientNetB0(
        include_top=True,
        weights=None,
        classes=11,
        classifier_activation="softmax"
    )
    efficientnet.compile(optimizer="adam", loss="categorical_crossentropy")
    efficientnet.fit(dataset, epochs=1)
