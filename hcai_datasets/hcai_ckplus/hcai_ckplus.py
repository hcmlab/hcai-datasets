"""hcai_ckplus dataset."""

import os
import glob
import tensorflow_datasets as tfds

# TODO(hcai_ckplus): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(hcai_ckplus): BibTeX citation
_CITATION = """
"""


class HcaiCkplus(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for hcai_ckplus dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
    '1.0.0': 'Initial release.',
  }

  def __init__(self, *, dataset_dir=os.path.join('\\\\137.250.171.12', 'Korpora', 'CK+'), **kwargs):
    self.dataset_dir = dataset_dir
    self.labels = ['neutral', 'anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'suprise']
    super(HcaiCkplus, self).__init__(**kwargs)

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(hcai_ckplus): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
      builder=self,
      description=_DESCRIPTION,
      features=tfds.features.FeaturesDict({
        # These are the features of your dataset like images, labels ...
        'image': tfds.features.Image(shape=(None, None, 3)),
        # 0=neutral, 1=anger, 2=contempt, 3=disgust, 4=fear, 5=happy, 6=sadness, 7=surprise)
        'label': tfds.features.ClassLabel(names=self.labels),
      }),
      # If there's a common (input, target) tuple from the
      # features, specify them here. They'll be used if
      # `as_supervised=True` in `builder.as_dataset`.
      supervised_keys=('image', 'label'),  # Set to `None` to disable
      homepage='https://dataset-homepage/',
      citation=_CITATION,
    )

  def _standard_splits(self):
    """Returns the standard splits for the dataset.
    Since clplus has no predefined splits the function applies a 70-15-15 split ratio to create them.
    Splits are sorted by filename and contain an equal emotion distribution."""

    emotion_files = glob.glob(os.path.join(self.dataset_dir, 'Emotion', '**/**/*.txt'))
    samples_classwise = {l: [] for l in self.labels}

    for ef in emotion_files:
      with open(ef, 'r') as f:
        emotion = self.labels[int(float(f.readline().strip()))]
        fn = ef.split(os.sep)[-1]
        samples_classwise[emotion].append(fn)

    train, val, test = [], [], []

    for k,v in samples_classwise.items():
      train.extend([ (x, k) for x in v[:int(len(v) * 0.7)]])
      val.extend([ (x, k) for x in v[int(len(v) * 0.7):int(len(v) * 0.85)]])
      test.extend([ (x, k) for x in v[int(len(v) * 0.85):]])

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

    for f, e in files:
      yield f, {
        'image': os.path.join(self.dataset_dir, 'cohn-kanade-images', *f.split('_')[:-2], f.replace('_emotion.txt', '.png')),
        'label': e,
      }
