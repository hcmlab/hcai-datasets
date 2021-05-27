### Description
This repository contains code to make datasets stored on th corpora network drive of the chair compatible with the [tensorflow dataset api](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) .

### Currently available Datasets

| Dataset       | Status        | Url  |
| :------------- |:-------------:| :-----|
| audioset      | ❌              | https://research.google.com/audioset/ |
| ckplus        | ✅             | http://www.iainm.com/publications/Lucey2010-The-Extended/paper.pdf |
| faces         | ✅             |    https://faces.mpdl.mpg.de/imeji/ |
| is2021_ess    | ❌             |    -|
| librispeech   | ❌              |    https://www.openslr.org/12 |
| nova_dynamic   | ✅              |    https://github.com/hcmlab/nova |


### Example Usage

```python
import os
import tensorflow as tf
import tensorflow_datasets as tfds
import hcai_datasets
from matplotlib import pyplot as plt

# Preprocessing function
def preprocess(x, y):
  img = x.numpy()
  return img, y

# Creating a dataset
ds, ds_info = tfds.load(
  'hcai_example_dataset',
  split='train',
  with_info=True,
  as_supervised=True,
  builder_kwargs={'dataset_dir': os.path.join('path', 'to', 'directory')}
)

# Input output mapping
ds = ds.map(lambda x, y: (tf.py_function(func=preprocess, inp=[x, y], Tout=[tf.float32, tf.int64])))

# Manually iterate over dataset
img, label = next(ds.as_numpy_iterator())

# Visualize
plt.imshow(img / 255.)
plt.show()
```