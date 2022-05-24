from pathlib import Path
from hcai_dataset_utils.bridge_tf import BridgeTensorflow
import tensorflow as tf
from hcai_datasets.hcai_nova_dynamic.hcai_nova_dynamic_iterable import HcaiNovaDynamicIterable

if __name__ == "__main__":

    iterable = HcaiNovaDynamicIterable(
        db_config_path="../../local/nova_db.cfg",
        db_config_dict=None,
        # Dataset Config
        dataset="affect-net",
        nova_data_dir=Path("Z:\\nova\\data"),
        sessions=[f"{i}_man_eval" for i in range(8)],
        roles=["session"],
        schemes=["emotion_categorical"],  # [""],
        # schemes=["neuthappysad"],
        annotator="gold",
        # annotator="system",
        data_streams=["video"],
        # Sample Config
        frame_size=0.04,
        left_context=0,
        right_context=0,
        # start = "0s",
        # end = "300s",
        flatten_samples=False,
        # Additional Config
        lazy_loading=False
    )
    dataset = BridgeTensorflow.make(iterable)

    for i, sample in enumerate(dataset):
        if i > 0:
            break
        print(sample["session.video"].shape)
        print(sample["session.emotion_categorical"].shape)

    # cast to supervised tuples
    dataset = dataset.map(lambda s: (s["session.video"], s["session.emotion_categorical"]))
    # resize images, one-hot vectors
    dataset = dataset.map(lambda x, y: (
        tf.image.resize(tf.ensure_shape(tf.squeeze(x), [300, 300, 3]), [224, 224]),
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
