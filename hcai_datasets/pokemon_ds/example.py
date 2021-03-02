from pokemon_ds import PokemonDs
import tensorflow_datasets as tfds
import tensorflow as tf

def pp(x, y):
    img = x.numpy()
    return img, y

## Load Data
ds, ds_info = tfds.load(
    'pokemon_ds',
    split='train',
    with_info=True,
    as_supervised=True,
)


ds = ds.map(lambda x, y: (tf.py_function(func=pp, inp=[x, y], Tout=[tf.float32, tf.int64])))
img = next(ds.as_numpy_iterator())[0]