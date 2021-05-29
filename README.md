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

### Example Usage Nova Dynamic Data
```python
import os
import hcai_datasets
import tensorflow_datasets as tfds

## Load Data
ds, ds_info = tfds.load(
  'hcai_nova_dynamic',
  split='dynamic_split',
  with_info=True,
  as_supervised=True,
  builder_kwargs={
    # Database Config
    'db_config_path': 'db.cfg',
    'db_config_dict': None,

    # Dataset Config
    'dataset': '<dataset_name>',
    'nova_data_dir': os.path.join('C:', 'Nova', 'Data'),
    'sessions': ['<session_name>'],
    'roles': ['<role_one>', '<role_two>'],
    'schemes': ['<label_scheme_one'],
    'annotator': '<annotator_id>',
    'data_streams': ['<stream_name>'],

    # Sample Config
    'frame_step': 1,
    'left_context': 0,
    'right_context': 0,
    'start': None,
    'end': None,
    #'flatten_samples': False, 
    'supervised_keys': ['<role_one>.<stream_name>', '<scheme_two>'],

    # Additional Config
    'clear_cache' : True
  }
)

data_it = ds.as_numpy_iterator()
ex_data = next(data_it)
print(ex_data)
```