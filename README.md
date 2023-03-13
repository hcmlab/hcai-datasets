## Description
This repository contains code to make datasets stored on the corpora network drive of the chair.
You can use this project to easily create or reuse a data loader that is universally compatible with either plain python code or tensorflow / pytorch. 
Also you this code can be used to dynamically create a dataloader for a Nova database to directly work with Nova Datasets in Python. 

Compatible with the [tensorflow dataset api](https://www.tensorflow.org/api_docs/python/tf/data/Dataset).
Pytorch Dataset [is also supported](https://pytorch.org/vision/stable/datasets.html).
 
## Installation Information

For efficient data loading we rely on the [decord](https://github.com/dmlc/decord) library. 
Decord ist not available as prebuild binary for non x86 architectures. 
If you want to install the project on other architecture you will need to compile it yourself. 

## Currently available Datasets

| Dataset       | Status        | Url  |
| :------------- |:-------------:| :-----|
| ckplus        | ✅             | http://www.iainm.com/publications/Lucey2010-The-Extended/paper.pdf |
| affectnet     | ✅             | http://mohammadmahoor.com/affectnet/ |
| faces         | ✅             |    https://faces.mpdl.mpg.de/imeji/ |
| nova_dynamic  | ✅             |    https://github.com/hcmlab/nova |
| audioset      | ❌             | https://research.google.com/audioset/ |
| is2021_ess    | ❌             |    -|
| librispeech   | ❌             |    https://www.openslr.org/12 |


## Architecture

![uml diagram](image/architecture.png)

Dataset implementations are split into two parts.\

Data access is handled by a generic python iterable, implemented by the DatasetIterable interface.\
The access class is then extended by an API class, which implements tfds.core.GeneratorBasedBuilder.
This results in the dataset being available by the Tensorflow Datasets API, and enables features 
such as local caching.

The iterables themselves can also be used as-is, either in PyTorch native DataGenerators by wrapping them in
the utility class BridgePyTorch, or as tensorflow-native Datasets by passing them to BridgeTensorflow.

The benefits of this setup are that a pytorch application can be served without installing or loading 
tensorflow, and vice versa, since the stack up to the adapters does not involve tf or pytorch. 
Also, when using tf, caching can be used or discarded by using tfds or the plain tensorflow Dataset
provided by the bridge.


### Dynamic Dataset usage with Nova Example 

To use the hcai_datasets repository with Nova you can use the `HcaiNovaDynamicIterable`class from the `hcai_datasets.hcai_nova_dynamic.hcai_nova_dynamic_iterable` module to create an iterator for a specific data configuration. 
This readme assumes that you are already familiar with the terminology and the general concept of the NOVA annotation tool / database.
The constructor of the class takes the following arguments as input: 

`db_config_path`: `string` path to a configfile with the nova database config. the config file looks like this:

```
[DB]
ip = 127.0.0.1
port = 37317
user = my_user
password = my_password
```

`db_config_dict`: `string` dictionary with the nova database config. can be used instead of db_config_path. if both are specified db_config_dict is used.

`dataset`: `string` the name of the dataset. Same as the entry in the Nova db.

`nova_data_dir`: `string` the directory to look for data. same as the directory specified in the nova gui. 

`sessions`: `list` list of sessions that should be loaded. must match the session names in nova.

`annotator`: `string` the name of the annotator that labeld the session. must match annotator names in nova.

`schemes`: `list` list of the annotation schemes to fetch.

`roles`: `list` list of roles for which the annotation should be loaded.

`data_streams`: `list` list datastreams for which the annotation should be loaded. must match stream names in nova.

`start`: `string | int | float` start time_ms. use if only a specific chunk of a session should be retrieved. can be passed as String (e.g. '1s' or '1ms'), Int (interpreted as milliseconds) or Float (interpreted as seconds).

`end`: `string | int | float` optional end time_ms. use if only a specific chunk of a session should be retrieved. can be passed as String (e.g. '1s' or '1ms'), Int (interpreted as milliseconds) or Float (interpreted as seconds).

`left_context`: `string | int | float` additional data to pass to the classifier on the left side of the frame. can be passed as String (e.g. '1s' or '1ms'), Int (interpreted as milliseconds) or Float (interpreted as seconds).

`right_context`: `string | int | float` additional data to pass to the classifier on the left side of the frame. can be passed as String (e.g. '1s' or '1ms'), Int (interpreted as milliseconds) or Float (interpreted as seconds).

`frame_size`: `string | int | float` the framesize to look at. the matching annotation will be calculated as majority vote from all annotations that are overlapping with the timeframe. can be passed as String (e.g. '1s' or '1ms'), Int (interpreted as milliseconds) or Float (interpreted as seconds).

`stride`: `string | int | float`  how much a frame is moved to calculate the next sample. equals framesize by default. can be passed as String (e.g. '1s' or '1ms'), Int (interpreted as milliseconds) or Float (interpreted as seconds).

`flatten_samples`: `bool` if set to `True` samples with the same annotation scheme but from different roles will be treated as separate samples. only <scheme> is used for the keys.  

`add_rest_class`: `bool` when set to True an additional rest class will be added to the end the label list


```python

from pathlib import Path
from hcai_dataset_utils.bridge_tf import BridgeTensorflow
import tensorflow as tf
from hcai_datasets.hcai_nova_dynamic.hcai_nova_dynamic_iterable import HcaiNovaDynamicIterable

ds_iter = HcaiNovaDynamicIterable(
    db_config_path="./nova_db.cfg",
    db_config_dict=None,
    dataset="affect-net",
    nova_data_dir=Path("./nova/data"),
    sessions=[f"{i}_man_eval" for i in range(8)],
    roles=["session"],
    schemes=["emotion_categorical"],
    annotator="gold",
    data_streams=["video"],
    frame_size=0.04,
    left_context=0,
    right_context=0,
    start = "0s",
    end = "3000ms",
    flatten_samples=False,
)

for sample in ds_iter:
    print(sample)
```

## Pytorch Example

The BridePyTorch module can be used to create a Pytorch DataLoader directly from the Dataset iterable. 

```python
from torch.utils.data import DataLoader
from hcai_dataset_utils.bridge_pytorch import BridgePyTorch
from hcai_datasets.hcai_affectnet.hcai_affectnet_iterable import HcaiAffectnetIterable

ds_iter = HcaiAffectnetIterable(
    dataset_dir="path/to/data_sets/AffectNet",
    split="test"
)
dataloader = DataLoader(BridgePyTorch(ds_iter))

for sample in dataloader:
    print(sample)
```


## Tensorflow Example

The BridgeTensorflow module can be used to create a Pytorch DataLoader directly from the Dataset iterable. 

```python
from hcai_dataset_utils.bridge_tf import BridgeTensorflow
from hcai_datasets.hcai_affectnet.hcai_affectnet_iterable import HcaiAffectnetIterable

ds_iter = HcaiAffectnetIterable(
    dataset_dir="path/to/data_sets/AffectNet",
    split="test"
)

tf_dataset = BridgeTensorflow.make(ds_iter)

for sample in tf_dataset:
    print(sample)
```


## Tensorflow Dataset API (DEPRECATED)

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
