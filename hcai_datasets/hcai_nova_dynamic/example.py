import os
import hcai_datasets
import tensorflow_datasets as tfds

## Load Data
ds, ds_info = tfds.load(
  'hcai_nova_dynamic',
  split='dynamic_split',
  with_info=True,
  as_supervised=False,
  builder_kwargs={
    'db_config_path': 'db.cfg',
    'db_config_dict': None,
    'dataset': 'DFG_A1_A2b',
    'nova_data_dir': os.path.join('\\\\137.250.171.12', 'Korpora', 'nova', 'data'),
    'sessions': ['NP001'],
    'roles': ['caretaker', 'infant'],
    'schemes': ['IEng'],
    'annotator': 'gold',
    'data_streams': ['video'],
    'frame_step': 5,
    'left_context': 1,
    'right_context': 1,
    'start': 0,
    'end': 200
  }
)

data_it = ds.as_numpy_iterator()
ex_data = next(data_it)
print(ex_data)