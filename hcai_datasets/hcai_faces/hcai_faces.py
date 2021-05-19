"""hcai_faces dataset."""

import os

import tensorflow as tf
import tensorflow_datasets as tfds

# TODO(hcai_faces): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(hcai_faces): BibTeX citation
_CITATION = """
"""


class HcaiFaces(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for hcai_faces dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
    '1.0.0': 'Initial release.',
  }

  def __init__(self, *, dataset_dir=None, **kwargs):
    super(HcaiFaces, self).__init__(**kwargs)
    self.dataset_dir = os.path.join(dataset_dir, 'bilder')

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return tfds.core.DatasetInfo(
      builder=self,
      description=_DESCRIPTION,
      features=tfds.features.FeaturesDict({
        # These are the features of your dataset like images, labels ...
        'image': tfds.features.Image(shape=(None, None, 3)),
        'id': tf.int64,
        'age': tfds.features.ClassLabel(names=['y', 'o', 'm']),
        'gender': tfds.features.ClassLabel(names=['m', 'f']),
        'emotion': tfds.features.ClassLabel(names=['a', 'd', 'f', 'h', 'n', 's']),
        'set': tfds.features.ClassLabel(names=['a', 'b']),
      }),
      # If there's a common (input, target) tuple from the
      # features, specify them here. They'll be used if
      # `as_supervised=True` in `builder.as_dataset`.
      supervised_keys=('image', 'emotion'),  # Set to `None` to disable
      homepage='https://dataset-homepage/',
      citation=_CITATION,
    )

  def _standard_splits(self):
    """Returns the standard splits for the dataset.
    Since faces has no predefined splits the function applies a 70-15-15 split ratio to create them.
    Splits are sorted by filename and contain unique subjects and equal gender distribution."""

    f_names = os.listdir(self.dataset_dir)
    f_names_unique = []

    for f in f_names:
      if f.endswith('jpg'):
        f_names_unique.append((f.split('_')[1], f.split('_')[0]))

    f_names_unique = set(f_names_unique)

    # sorting unique subject ids into age categories
    old = sorted([x[1] for x in f_names_unique if x[0] == 'o'])
    middle = sorted([x[1] for x in f_names_unique if x[0] == 'm'])
    young = sorted([x[1] for x in f_names_unique if x[0] == 'y'])

    # building splits using 70,15,15 % distribution evenly over all age groups
    f = 0
    t = 0.7
    train_ids = old[int(len(old) * f): int(len(old) * t)] + \
                middle[int(len(middle) * f): int(len(middle) * t)] + \
                young[int(len(young) * f): int(len(young) * t)]
    f = 0.7
    t = 0.85
    val_ids = old[int(len(old) * f): int(len(old) * t)] + \
              middle[int(len(middle) * f): int(len(middle) * t)] + \
              young[int(len(young) * f): int(len(young) * t)]
    f = 0.85
    t = 1
    test_ids = old[int(len(old) * f): int(len(old) * t)] + \
               middle[int(len(middle) * f): int(len(middle) * t)] + \
               young[int(len(young) * f): int(len(young) * t)]

    # getting filenames for each split
    train = [f for x in train_ids for f in f_names if f.startswith(x)]
    val = [f for x in val_ids for f in f_names if f.startswith(x)]
    test = [f for x in test_ids for f in f_names if f.startswith(x)]

    return train, val, test

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    train, val, test = self._standard_splits()
    return {
      'train': self._generate_examples(train),
      'val': self._generate_examples(val),
      'test': self._generate_examples(test),
    }

  def _generate_examples(self, files):
    """Yields examples."""
    for f in files:
      # names are composed as the following:
      # "ID_ageGroup_gender_emotion_pictureSet.jpg
      id, age, gender, emotion, set = os.path.splitext(f)[0].split('_')

      yield f, {
        'image': os.path.join(self.dataset_dir, f),
        'id': int(id),
        'age': age,
        'gender': gender,
        'emotion': emotion,
        'set': set
      }
