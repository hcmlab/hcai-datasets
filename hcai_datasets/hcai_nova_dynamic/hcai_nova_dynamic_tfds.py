"""hcai_nova_dynamic dataset."""
from enum import Enum
from typing import Any

import shutil
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow_datasets.core import split_builder as split_builder_lib

from hcai_dataset_utils.bridge_tf import BridgeTensorflow
from hcai_datasets.hcai_nova_dynamic.hcai_nova_dynamic_iterable import HcaiNovaDynamicIterable
import hcai_datasets.hcai_nova_dynamic.utils.nova_types as nt
import hcai_datasets.hcai_nova_dynamic.utils.nova_data_utils as ndu
import hcai_datasets.hcai_nova_dynamic.utils.nova_anno_utils as nau

from hcai_datasets.hcai_nova_dynamic.utils.nova_utils import *

# TODO(hcai_audioset): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
The Nova Dynamic dataset can be used to retrieve the data and labels for a certain session or a certain part of a session of a nova dataset. 
This is part of the Nova CML Python backend (https://github.com/hcmlab/nova)
To specify which data to load use the following format: 

TODO: x
 
"""

# TODO(hcai_audioset): BibTeX citation
_CITATION = """
"""

class TFDataTypes(Enum):
  UNDEF = 0
  SHORT = tf.int16
  USHORT = tf.uint16
  INT = tf.int32
  UINT = tf.uint32
  LONG = tf.int64
  ULONG = tf.uint64
  FLOAT = tf.float32
  DOUBLE = tf.float64
  LDOUBLE = tf.float64
  BOOL = tf.bool


class HcaiNovaDynamic(HcaiNovaDynamicIterable, tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for hcai_nova_dynamic dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def __init__(self, clear_cache=True, *args, **kwargs):
        """
        Initialize the HcaiNovaDynamic dataset builder
        Args:
          clear_cache:  when set to True the cache will be cleared else the cached dataset will be used. make sure that dataset and sample config did not change. defaults to true.
          nova_data_dir: the directory to look for data. same as the directory specified in the nova gui.
          frame_size: the framesize to look at. the matching annotation will be calculated as majority vote from all annotations that are overlapping with the timeframe.
          left_context: additional data to pass to the classifier on the left side of the frame.
          right_context: additional data to pass to the classifier on the left side of the frame.
          stride: how much a frame is moved to calculate the next sample. equals framesize by default.
          flatten_samples: if set to True samples with the same annotation scheme but from different roles will be treated as separate samples. only <scheme> is used for the keys.
          supervised_keys: if specified the dataset can be used with "as_supervised" set to True. Should be in the format <role>.<scheme>. if flatten_samples is true <role> will be ignored.
          add_rest_class: when set to True an additional restclass will be added to the end the label list
          db_config_path: path to a configfile whith the nova database config.
          db_config_dict: dictionary with the nova database config. can be used instead of db_config_path. if both are specified db_config_dict is used.
          dataset: the name of the dataset. must match the dataset name in the nova database.
          sessions: list of sessions that should be loaded. must match the session names in nova.
          annotator: the name of the annotator that labeld the session. must match annotator names in nova.
          schemes: list of the annotation schemes to fetch
          roles: list of roles for which the annotation should be loaded.
          data_streams: list datastreams for which the annotation should be loaded. must match stream names in nova.
          start: optional start time_ms. use if only a specific chunk of a session should be retreived.
          end: optional end time_ms. use if only a specifc chunk of a session should be retreived.
          **kwargs: arguments that will be passed through to the dataset builder
        """

        HcaiNovaDynamicIterable.__init__(self, *args, **kwargs)
        tfds.core.GeneratorBasedBuilder.__init__(self)

        if clear_cache:
            try:
                shutil.rmtree(self.data_dir)
            except OSError as e:
                print("Error: %s : %s" % (self.data_dir, e.strerror))

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""

        def map_label_id(lid):
            if self.flatten_samples and not lid == 'frame':
                return split_role_key(lid)[-1]
            return lid

        features_dict = {
            # TODO: Remove frame when tfds implements option to disable shuffle
            # Adding fake framenumber label for sorting
            'frame': tf.string,
            **{map_label_id(k): self._build_tfds_label_info(k, v) for k, v in self.annos.items()},
            **{map_label_id(k): self._build_tfds_data_info(k, v) for k, v in self.data_info.items()}
        }

        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(features_dict),
            supervised_keys=self.supervised_keys,
            homepage='https://github.com/hcmlab/nova',
            citation=_CITATION,
            disable_shuffling=True
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        return {'dynamic_split': self._generate_examples()}

    def _generate_examples(self, **kwargs: Any) -> split_builder_lib.SplitGenerator:
        sample_counter = 1
        iter = self._yield_sample()
        while True:
            try:
                yield sample_counter, iter.__next__()
            except StopIteration:
                break
            sample_counter = sample_counter + 1

    def _build_tfds_label_info(self, label_key, scheme):

        scheme_type = self.anno_schemes[label_key]

        if scheme_type == nt.AnnoTypes.DISCRETE:
            return tfds.features.ClassLabel(names=list(scheme.labels.values()))
        elif scheme_type == nt.AnnoTypes.DISCRETE_POLYGON:
            return tfds.features.FeaturesDict(
                {
                    str(l): tfds.features.Sequence(
                        tfds.features.Tensor(shape=(2,), dtype=tf.float64)
                    )
                    for l in scheme.labels.keys()
                }
            )
        elif scheme_type == nt.AnnoTypes.FREE:
            return tfds.features.Sequence(tfds.features.Text())
        else:
            raise NotImplementedError(f"_build_tfds_label_info not implemented for {scheme_type}")

    def _build_tfds_data_info(self, data_key, data):

        if data.lazy_loading:
            return tfds.features.FeaturesDict(
                {
                    "frame_start": tf.dtypes.float32,
                    "frame_end": tf.dtypes.float32,
                    "file_path": tfds.features.Text(),
                }
            )

        scheme_type = self.data_schemes[data_key]

        if scheme_type == nt.DataTypes.VIDEO:
            return tfds.features.Sequence(
                tfds.features.Image(shape=data.sample_data_shape, dtype=tf.uint8)
            )
        elif scheme_type == nt.DataTypes.AUDIO:
            return tfds.features.Audio()
        elif scheme_type == nt.DataTypes.FEATURE:
            data_type = data.np_data_type
            if BridgeTensorflow.TYPE_MAPPING[data_type] is not None:
                data_type = BridgeTensorflow.TYPE_MAPPING[data_type]
            return tfds.features.Sequence(
                tfds.features.Tensor(shape=data.sample_data_shape, dtype=data_type)
            )
        else:
            raise NotImplementedError(f"_build_tfds_data_info not implemented for {scheme_type}")
