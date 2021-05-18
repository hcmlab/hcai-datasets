"""pokemon_ds dataset."""

import tensorflow_datasets as tfds
import pathlib
import os

# TODO(pokemon_ds): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(pokemon_ds): BibTeX citation
_CITATION = """
Yes please
"""


class PokemonDs(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for pokemon_ds dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(pokemon_ds): Specifies the tfds.core.DatasetInfo object
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                # These are the features of your dataset like images, labels ...
                'image': tfds.features.Image(shape=(299, 299, 3)),
                'label': tfds.features.ClassLabel(names=['Blastoise', 'Charizard', 'Charmeleon', 'Ivysaur', 'Venusaur', 'Wartortle']),
            }),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=('image', 'label'),  # Set to `None` to disable
            homepage='https://hcmlab.de/',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # TODO(pokemon_ds): Downloads the data and defines the splits
        #path = dl_manager.download_and_extract('https://todo-data-url')
        path = pathlib.Path(r'./dummy_data')

        # TODO(pokemon_ds): Returns the Dict[split names, Iterator[Key, Example]]
        return {
            'train': self._generate_examples(path / 'train_imgs'),
            'validation': self._generate_examples(path / 'val_imgs'),
            'test': self._generate_examples(path / 'test_imgs')
        }

    def _generate_examples(self, path):
        """Yields examples."""
        # TODO(pokemon_ds): Yields (key, example) tuples from the dataset
        for f in path.glob('**/*.jpeg'):
            label = str(f).split(os.sep)[-2]
            key = str(f).split(os.sep)[-1]
            yield key, {
                'image': f,
                'label': label,
            }