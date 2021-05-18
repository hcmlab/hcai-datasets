import os
import tensorflow as tf
import tensorflow_datasets as tfds
import hcai_faces
from matplotlib import pyplot as plt

def pp(x, y):
  img = x.numpy()
  return img, y


## Load Data
ds, ds_info = tfds.load(
  'hcai_faces',
  split='train',
  with_info=True,
  as_supervised=True,
  builder_kwargs={'dataset_dir': os.path.join('\\\\137.250.171.12', 'Korpora', 'FACES', 'bilder')}
)

ds = ds.map(lambda x, y: (tf.py_function(func=pp, inp=[x, y], Tout=[tf.float32, tf.int64])))
img, label = next(ds.as_numpy_iterator())

plt.imshow(img / 255.)
plt.show()