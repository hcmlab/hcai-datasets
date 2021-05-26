"""hcai_nova_dynamic dataset."""

import tensorflow_datasets as tfds
import os
import cv2
import sys
import numpy as np
from hcai_datasets.hcai_nova_dynamic.nova_db_handler import NovaDBHandler
from hcai_datasets.hcai_nova_dynamic import data_utils
from hcai_datasets.hcai_nova_dynamic.defines import DataTypes

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


class HcaiNovaDynamic(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for hcai_nova_dynamic dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
    '1.0.0': 'Initial release.',
  }

  def __init__(self, *, db_config_path=None, db_config_dict=None, dataset=None, nova_data_dir=None, sessions=None, annotator=None,
               schemes=None, roles=None, data_streams=None, start=0, end=sys.float_info.max, left_context, right_context, frame_step, **kwargs):
    """
    Initialize the HcaiNovaDynamic dataset builder
    Args:
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
    self.dataset = dataset
    self.nova_data_dir = nova_data_dir
    self.sessions = sessions
    self.roles = roles
    self.annotator = annotator
    self.left_context = left_context
    self.right_context = right_context
    self.frame_step = frame_step
    self.start = start
    self.end = end
    self.nova_db_handler = NovaDBHandler(db_config_path, db_config_dict)

    mongo_schemes = self.nova_db_handler.get_schemes(dataset=dataset, schemes=schemes)
    mongo_streams = self.nova_db_handler.get_data_streams(dataset=dataset, data_streams=data_streams)

    # infos as needed for the tensorflow dataset init and     # additional info for loading samples
    self._info_label = self._get_label_info_from_mongo_doc(mongo_schemes)
    self._info_data = self._get_data_info_from_mongo_doc(mongo_streams)

    super(HcaiNovaDynamic, self).__init__(**kwargs)

  def _info(self) -> tfds.core.DatasetInfo:
      """Returns the dataset metadata."""

      return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({**self._info_label, **{k:v['feature'] for k,v in self._info_data.items() }}),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=(None),  # Set to `None` to disable
        homepage='https://github.com/hcmlab/nova',
        citation=_CITATION
      )

  def _get_label_info_from_mongo_doc(self, mongo_schemes):
    label_info = {}
    # List of all combinations from roles and schemes that occur in the retreived data. Form is 'role.scheme'
    #roles_with_scheme = set([rs for session in self.annos.values() for rs in list(session.keys())])
    for scheme in mongo_schemes:
        # List of all role where the current scheme occures in the retrieved data.
        # label_keys = list(filter(lambda x : scheme['name'] in x,  roles_with_scheme))
        for role in self.roles:
          if scheme['type'] == 'DISCRETE':
            label_info[role + '.' + scheme['name']] = tfds.features.ClassLabel(names=[x['name'] for x in sorted(scheme['labels'], key=lambda k: k['id'])])
          else:
            raise ValueError('Invalid label type {}'.format(scheme['type']))
    return label_info

  def _get_video_resolution(self, path):
    vcap = cv2.VideoCapture(path)
    # get vcap property
    width  = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vcap.release()
    return (height, width)

  def _get_data_info_from_mongo_doc(self, mongo_data_streams):
      data_info = {}
      for data in mongo_data_streams:
        for role in self.roles:
          if data['type'] == 'video':
            # We can use any role for testing since Nova assumes that there will be a stream for every role
            role = self.roles[0]
            sample_stream_name = role + '.' + data['name'] + '.' + data['fileExt']
            sample_stream_path = os.path.join(self.nova_data_dir, self.dataset, self.sessions[0], sample_stream_name)
            res = self._get_video_resolution(sample_stream_path)
            # shape is (None, H, W, C) - We assume that we always have three channels
            data_shape = (None, ) + res + (3, )
            data_id = role + '.' + data['name']
            data_info[data_id] = {'feature' : tfds.features.Video(data_shape), 'file': sample_stream_name, 'sr': data['sr'], 'type': DataTypes.VIDEO}
          elif data['type'] == 'audio':
            pass
          else:
            raise ValueError('Invalid data type {}'.format(data['type']))

      return data_info

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    return {'dynamic_split': self._generate_examples()}

  def _get_label_for_frame(self, annotation, start, end):

    if annotation == -1:
      return -1

    else:
      annos_for_sample = list(filter(lambda x: x['from'] >= start and x['to'] <= end, annotation))

      if not annos_for_sample:
        return -1

      majority_sample_idx = np.argmax(list(map(lambda x: x['to'] - x['from'], annos_for_sample)))

      return annos_for_sample[majority_sample_idx]['id']

  def _get_data_for_frame(self, file_reader, feature_type, sr, start, end):
    start_frame = data_utils.time_to_frame(sr, start)
    end_frame = data_utils.time_to_frame(sr, end)
    if feature_type == DataTypes.VIDEO:
      return data_utils.chunk_vid(file_reader, start_frame, end_frame)

  def _generate_examples(self):
    """Yields examples."""
    # Fetching all annotations that are available for the respective schemes and roles
    for session in self.sessions:

      annotation = {}
      data = {}

      # loading annotations for the session
      for l in self._info_label.keys():
        r, s = l.split('.')
        annotation[l] = self.nova_db_handler.get_annos(self.dataset, s, session, self.annotator, r)

      # openening data reader for this session
      for d, feature_info in self._info_data.items():
        data_path = os.path.join(self.nova_data_dir, self.dataset, session, feature_info['file'])

        if not os.path.isfile(data_path):
          raise FileNotFoundError('No datastream found at {}'.format(data_path))
        file_reader = data_utils.open_file_reader(data_path, feature_info['type'])
        data[d] = file_reader

      session_info = self.nova_db_handler.get_session_info(self.dataset, session)[0]
      dur = session_info['duration']

      # current position of the last frame in seconds
      c_pos = 0

      # increase current position till we reach the first frame
      while c_pos - (self.left_context + self.frame_step) < self.start:
        c_pos += self.frame_step

      # generate samples for this session
      while c_pos + self.right_context < min(self.end, dur):
        sample_start = c_pos - (self.left_context + self.frame_step)
        sample_end = c_pos + self.right_context
        key = str(sample_start) + '_' + str(sample_end)

        labels_for_sample = [{ k: self._get_label_for_frame(v, sample_start, sample_end)} for k, v in annotation.items()]
        data_for_sample = [{ k: self._get_data_for_frame(v, self._info_data[k]['type'], self._info_data[k]['sr'], sample_start, sample_end)} for k, v in data.items()]

        #sample_dict = {
        #  'video' : np.ones( (20,288,180,3), dtype=np.uint8)
        #}
        sample_dict = {}
        for l in labels_for_sample:
          sample_dict.update(l)

        for d in data_for_sample:
          sample_dict.update(d)

        yield key, sample_dict
        c_pos += self.frame_step

      # closing file readers for this session
      for d, fr in data.items():
        data_utils.close_file_reader(fr, self._info_data[d]['feature'])