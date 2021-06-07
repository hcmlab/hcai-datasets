"""hcai_nova_dynamic dataset."""

import numpy as np
import os
import shutil
import sys
import tensorflow_datasets as tfds
import tensorflow as tf

import hcai_datasets.hcai_nova_dynamic.utils.nova_data_types as ndt
from hcai_datasets.hcai_nova_dynamic.nova_db_handler import NovaDBHandler
from hcai_datasets.hcai_nova_dynamic.utils import nova_data_utils as ndu

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

    def __init__(self, *, db_config_path=None, db_config_dict=None, dataset=None, nova_data_dir=None, sessions=None,
                 annotator=None,
                 schemes=None, roles=None, data_streams=None, start=None, end=None, left_context='0', right_context='0',
                 frame_size='1', stride=None,
                 flatten_samples=False, supervised_keys=None, clear_cache=True,
                 **kwargs):
        """
        Initialize the HcaiNovaDynamic dataset builder
        Args:
          nova_data_dir: the directory to look for data. same as the directory specified in the nova gui.
          frame_size: the framesize to look at. the matching annotation will be calculated as majority vote from all annotations that are overlapping with the timeframe.
          left_context: additional data to pass to the classifier on the left side of the frame.
          right_context: additional data to pass to the classifier on the left side of the frame.
          stride: how much a frame is moved to calculate the next sample. equals framesize by default.
          flatten_samples: if set to True samples with the same annotation scheme but from different roles will be treated as separate samples. only <scheme> is used for the keys.
          supervised_keys: if specified the dataset can be used with "as_supervised" set to True. Should be in the format <role>.<scheme>. if flatten_samples is true <role> will be ignored.
          clear_cache:  when set to True the cache will be cleared else the cached dataset will be used. make sure that dataset and sample config did not change. defaults to true.
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
        self.schemes = schemes
        self.data_streams = data_streams
        self.annotator = annotator
        self.left_context_ms = ndu.parse_time_string_to_ms(left_context)
        self.right_context_ms = ndu.parse_time_string_to_ms(right_context)
        self.frame_size_ms = ndu.parse_time_string_to_ms(frame_size)
        self.stride_ms = ndu.parse_time_string_to_ms(stride) if stride else self.frame_size_ms
        self.start_ms = ndu.parse_time_string_to_ms(start) if start else 0
        self.end_ms = ndu.parse_time_string_to_ms(end) if end else float('inf')
        self.flatten_samples = flatten_samples
        self.clear_cache = clear_cache
        self.supervised_keys = supervised_keys

        self.nova_db_handler = NovaDBHandler(db_config_path, db_config_dict)

        mongo_schemes = self.nova_db_handler.get_schemes(dataset=dataset, schemes=schemes)
        mongo_streams = self.nova_db_handler.get_data_streams(dataset=dataset, data_streams=data_streams)

        # infos as needed for the tensorflow dataset init and     # additional info for loading samples
        self._info_label = self._get_label_info_from_mongo_doc(mongo_schemes)
        self._info_data = self._get_data_info_from_mongo_doc(mongo_streams)

        # TODO: At the moment flatten_samples only works when supervised keys are provided. this is probably not necessary
        if self.flatten_samples and not self.supervised_keys:
            raise Warning('\'flatten_samples\' has been set but \'supervised_keys\' are not specified.')
        if self.flatten_samples and self.supervised_keys[0] not in self.data_streams:
            # remove <role> of supervised keys
            _, data_stream = self._split_role_key(self.supervised_keys[0])
            if not data_stream in self.data_streams:
                print('Warning: Cannot find supervised key \'{}\' in datastreams'.format(data_stream))
                raise Warning('Unknown data_stream')
            else:
                self.supervised_keys[0] = data_stream
        if self.flatten_samples and self.supervised_keys[-1] not in self.schemes:
            # remove <role> of supervised keys
            _, scheme = self._split_role_key(self.supervised_keys[1])
            if not scheme in schemes:
                print('Warning: Cannot find supervised key \'{}\' in schemes'.format(scheme))
                raise Warning('Unknown scheme')
            else:
                self.supervised_keys[1] = scheme

        super(HcaiNovaDynamic, self).__init__(**kwargs)

        if clear_cache:
            try:
                shutil.rmtree(self.data_dir)
            except OSError as e:
                print("Error: %s : %s" % (self.data_dir, e.strerror))

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""

        def map_label_id(lid):
            if self.flatten_samples and not lid == 'frame':
                return self._split_role_key(lid)[-1]
            return lid

        features_dict = {
                    **{map_label_id(k): v for k,v in self._info_label.items()},
                    **{map_label_id(k): v['feature'] for k, v in self._info_data.items()}
                }

        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(features_dict),
            supervised_keys=tuple(self.supervised_keys),
            homepage='https://github.com/hcmlab/nova',
            citation=_CITATION,
            # TODO: This option is currently disabled because it raises an error with tfds 4.3.0
            # Code should be updated once a newer version is released
            #disable_shuffling=True
        )

    def _merge_role_key(self, key, role):
        return role + '.' + key

    def _split_role_key(self, label_key):
        split = label_key.split('.')
        role = split[0]
        key = '.'.join(split[1:])
        return role, key

    def _get_label_info_from_mongo_doc(self, mongo_schemes):
        """

        Args:
          mongo_schemes:

        Returns:

        """

        label_info = {}

        # TODO: Remove when tfds implements option to disable shuffle
        # Adding fake framenumber label for sorting
        label_info['frame'] = tf.string
        # List of all combinations from roles and schemes that occur in the retrieved data. Form is 'role.scheme'.
        # If flatten_sample is True form is 'scheme'. s
        for scheme in mongo_schemes:
            for role in self.roles:
                label_id = self._merge_role_key(role=role, key=scheme['name'])

                if scheme['type'] == 'DISCRETE':
                    label_info[label_id] = tfds.features.ClassLabel(
                        names=[x['name'] for x in sorted(scheme['labels'], key=lambda k: k['id'])])
                else:
                    raise ValueError('Invalid label type {}'.format(scheme['type']))

        return label_info

    def _get_data_info_from_mongo_doc(self, mongo_data_streams):
        data_info = {}

        for data_stream in mongo_data_streams:
            for role in self.roles:
                sample_stream_name = role + '.' + data_stream['name'] + '.' + data_stream['fileExt']
                sample_stream_path = os.path.join(self.nova_data_dir, self.dataset, self.sessions[0],
                                                  sample_stream_name)
                data_id = self._merge_role_key(role=role, key=data_stream['name'])
                dtype = ndt.string_to_enum(ndt.DataTypes, data_stream['type'])

                if dtype == ndt.DataTypes.video:
                    res = ndu.get_video_resolution(sample_stream_path)
                    # shape is (None, H, W, C) - We assume that we always have three channels
                    data_shape = (None,) + res + (3,)
                    feature_connector = tfds.features.Video(data_shape)
                elif dtype == ndt.DataTypes.audio:
                    raise NotImplementedError('Audio files are not yet supported')
                elif dtype == ndt.DataTypes.feature:
                    stream = ndu.Stream().load_header(sample_stream_path)
                    data_shape = (stream.dim,)
                    data_type = stream.tftype
                    feature_connector = tfds.features.Sequence(tfds.features.Tensor(shape=data_shape, dtype=data_type))
                else:
                    raise ValueError('Invalid data type {}'.format(data_stream['type']))

                data_info[data_id] = {'feature': feature_connector, 'file': sample_stream_name, 'sr': data_stream['sr'],
                                      'type': dtype}
        return data_info

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        return {'dynamic_split': self._generate_examples()}

    def _get_label_for_frame(self, annotation, start, end):

        if annotation == -1:
            return -1

        else:

            # finding all annos that overlap with the frame
            def is_overlapping(af, at, ff, ft):

                # anno is larger than frame
                altf = af <= ff and at >= ft

                # anno overlaps frame start
                aofs = at >= ff and at <= ft

                # anno overlaps frame end
                aofe = af >= ff and af <= ft

                return altf or aofs or aofe

            annos_for_sample = list(filter(lambda x: is_overlapping(x['from'], x['to'], start, end), annotation))

            if not annos_for_sample:
                return -1

            majority_sample_idx = np.argmax(
                list(map(lambda x: min(end, x['to']) - max(start, x['from']), annos_for_sample)))

            return annos_for_sample[majority_sample_idx]['id']

    def _get_data_for_frame(self, file_reader, feature_type, sr, start, end):
        start_frame = ndu.milli_seconds_to_frame(sr, start)
        end_frame = ndu.milli_seconds_to_frame(sr, end)
        if feature_type == ndt.DataTypes.video:
            return ndu.chunk_vid(file_reader, start_frame, end_frame)
        elif feature_type == ndt.DataTypes.audio:
            raise NotImplementedError('data chunking for audio is not yet implemented')
        elif feature_type == ndt.DataTypes.feature:
            return ndu.chunk_stream(file_reader, start_frame, end_frame)

    def _get_annotation_for_session(self, session, time_in_ms=False):
        annotation = {}

        # loading annotations for the session
        for r in self.roles:
            for s in self.schemes:
                label_key = self._merge_role_key(role=r, key=s)
                annotation[label_key] = self.nova_db_handler.get_annos(self.dataset, s, session, self.annotator, r)

                if time_in_ms:
                    for d in annotation[label_key]:
                        d['from'] = int(d['from'] * 1000)
                        d['to'] = int(d['to'] * 1000)

        return annotation

    def _get_data_reader_for_session(self, session):
        data = {}

        # openening data reader for this session
        for d, feature_info in self._info_data.items():
            data_path = os.path.join(self.nova_data_dir, self.dataset, session, feature_info['file'])

            if not os.path.isfile(data_path):
                raise FileNotFoundError('No datastream found at {}'.format(data_path))
            file_reader = ndu.open_file_reader(data_path, feature_info['type'])
            data[d] = file_reader

        return data


    def _generate_examples(self):
            """Yields examples."""

            # Needed to sort the samples later and assure that the order is the same as in nova. Necessary for CML till the tfds can be returned in order.
            sample_counter = 1

            # Fetching all annotations that are available for the respective schemes and roles
            for session in self.sessions:

                # Gather all data we need for this session
                annotation = self._get_annotation_for_session(session, time_in_ms=True)

                # Converting anno times from s to ms

                data = self._get_data_reader_for_session(session)
                session_info = self.nova_db_handler.get_session_info(self.dataset, session)[0]
                dur_ms = session_info['duration'] * 1000

                if not dur_ms:
                    raise ValueError('Session {} has no duration.'.format(session))

                # Starting position of the first frame in seconds
                c_pos_ms = self.left_context_ms + self.start_ms

                # Generate samples for this session
                while c_pos_ms + self.stride_ms + self.right_context_ms <= min(self.end_ms, dur_ms):
                    sample_start_ms = c_pos_ms - self.left_context_ms
                    sample_end_ms = c_pos_ms + self.frame_size_ms + self.right_context_ms
                    key = str(sample_start_ms / 1000) + '_' + str(sample_end_ms / 1000)

                    labels_for_frame = [{k: self._get_label_for_frame(v, sample_start_ms, sample_end_ms)} for k, v in
                                        annotation.items()]
                    data_for_frame = [{k: self._get_data_for_frame(v, self._info_data[k]['type'], self._info_data[k]['sr'],
                                                                   sample_start_ms, sample_end_ms)} for k, v in data.items()]

                    sample_dict = {}

                    for l in labels_for_frame:
                        sample_dict.update(l)

                    for d in data_for_frame:
                        sample_dict.update(d)

                    if self.flatten_samples:
                        if not self.supervised_keys:
                            raise RuntimeError('Trying to \'flatten_sample\' but \'supervised_keys\' has not been set.')

                        # grouping labels and data according to roles
                        for role in self.roles:
                            data_key = self._merge_role_key(role=role, key=self.supervised_keys[0])
                            label_key = self._merge_role_key(role=role, key=self.supervised_keys[1])

                            yield key + '_' + role, {
                                #'frame': sample_counter,
                                'frame': str(sample_counter) + '_' + key + '_' + role,
                                self.supervised_keys[0]: sample_dict[data_key],
                                self.supervised_keys[1]: sample_dict[label_key]
                            }
                            sample_counter += 1
                        c_pos_ms += self.stride_ms

                    else:
                        sample_dict['frame'] = str(sample_counter) + '_' + key
                        yield key, sample_dict
                        c_pos_ms += self.stride_ms
                        sample_counter += 1

                # closing file readers for this session
                for d, fr in data.items():
                    ndu.close_file_reader(fr, self._info_data[d]['feature'])
