import cv2
import numpy as np
import tensorflow_datasets as tfds
from hcai_datasets.hcai_nova_dynamic.utils import nova_types as nt
from hcai_datasets.hcai_nova_dynamic.utils.ssi_stream_utils import Stream
from hcai_datasets.hcai_nova_dynamic.utils.nova_utils import merge_role_key, split_role_key
from typing import Union
from abc import ABC, abstractmethod

class Data(ABC):

    def __init__(self,  role: str = '', name: str = '', sr: int = 0, is_valid: bool = True, sample_data_path: str = ''):
        self.role = role
        self.name = name
        self.is_valid = is_valid
        self.sr = sr

        # Set when populate_meta_info is called
        self.sample_data_shape = None
        self.tf_data_type = None
        self.meta_loaded = False

        # Set when open_file_reader is called
        self.data_reader = None
        self.dur = None

        if sample_data_path:
            self.populate_meta_info(sample_data_path)

    def data_stream_opend(self):
        if not self.data_reader:
            print('No datastream opened for {}'.format(merge_role_key(self.role, self.name)))
            raise RuntimeError('Datastream not loaded')

    @abstractmethod
    def get_tf_info(self):
        """
        Returns the features for this datastream to create the DatasetInfo for tensorflow
        """
        ...

    @abstractmethod
    def get_chunk(self, start_frame: int, end_frame: int):
        """
        Returns a data chunk from start frame to end frame
        """
        ...

    @abstractmethod
    def open_file_reader(self, path: str):
        """
        Opens a filereader for the respective datastream
        """
        ...

    @abstractmethod
    def populate_meta_info(self, path: str):
        """
        Opens a data sample from the provided path to extract additional data that is not in the database
        """
        ...

    @abstractmethod
    def close_file_reader(path):
        """
        Closes a filereader for the respective datastream
        """
        ...

class AudioData(Data):
    pass

class VideoData(Data):
    # res = ndu.get_video_resolution(sample_stream_path)
    # shape is (None, H, W, C) - We assume that we always have three channels
    # data_shape = (None,) + res + (3,)
    # feature_connector = tfds.features.Video(data_shape)
    pass

class StreamData(Data):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_tf_info(self) -> (str, tfds.features.Sequence):
        if self.meta_loaded:
            feature_connector = tfds.features.Sequence(tfds.features.Tensor(shape=self.sample_data_shape, dtype=self.tf_data_type))
            return merge_role_key(self.role, self.name), feature_connector
        else:
            print('No datashape and type have been loaded for stream {}. Call get_meta_info() first.'.format(merge_role_key(self.role, self.name)))

    def get_chunk(self, frame_start_ms: int, frame_end_ms: int):
        try:
            self.data_stream_opend()
            start_frame = milli_seconds_to_frame(self.sr, frame_start_ms)
            end_frame = milli_seconds_to_frame(self.sr, frame_end_ms)
            return self.data_reader.data[start_frame:end_frame]
        except RuntimeError:
            print('Could not get chunk {}-{} from data stream {}'.format(start_frame, end_frame, merge_role_key(self.role, self.name)))

    def open_file_reader(self, path: str) -> bool:
        stream = Stream(path + '.stream')
        if stream:
            self.data_reader = stream
            self.dur = stream.data.shape[0] / stream.sr
            return True
        else:
            print('Could not open Stream {}'.format(str))
            return False

    def close_file_reader(path):
        return True

    def populate_meta_info(self, path: str):
        stream = Stream().load_header(path)
        self.sample_data_shape = (stream.dim,)
        self.tf_data_type = stream.tftype
        self.meta_loaded = True

def frame_to_seconds(sr: int, frame: int) -> float:
    return frame / sr

def seconds_to_frame(sr: int, time_s: float) -> int:
    return round(time_s * sr)

def milli_seconds_to_frame(sr: int, time_ms: int) -> int:
    return seconds_to_frame(sr=sr, time_s= time_ms / 1000)

def parse_time_string_to_ms(frame: Union[str, int, float]) -> int:
    # if frame is specified milliseconds as string
    if str(frame).endswith('ms'):
        try:
            return int(frame[:-2])
        except ValueError:
            raise ValueError('Invalid input format for frame in milliseconds: {}'.format(frame))
    # if frame is specified in seconds as string
    elif str(frame).endswith('s'):
        try:
            frame_s = float(frame[:-1])
            return int(frame_s * 1000)
        except ValueError:
            raise ValueError('Invalid input format for frame in seconds: {}'.format(frame))
    # if type is float we assume the input will be seconds
    elif isinstance(frame, float) or '.' in str(frame):
        try:
            print('WARNING: Automatically inferred type for frame {} is float.'.format(frame))
            return int(1000 * float(frame))
        except ValueError:
            raise ValueError('Invalid input format for frame: {}'.format(frame))
    # if type is int we assume the input will be milliseconds
    elif isinstance(frame, int):
        try:
            print('WARNING: Automatically inferred type for frame {} is int.'.format(frame))
            return int(frame)
        except ValueError:
            raise ValueError('Invalid input format for frame: {}'.format(frame))

def chunk_vid(vcap: cv2.VideoCapture, start_frame: int, end_frame: int):
    vcap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    width = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    heigth = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    depth = 3
    length = end_frame - start_frame
    chunk = np.zeros((length, heigth, width, depth), dtype=np.uint8)

    for i in range(length):
        ret, frame = vcap.read()

        if not ret:
            raise IndexError('Video frame {} out of range'.format(i))

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        chunk[i] = frame

    return chunk

def open_file_reader(path, feature_type):
    if feature_type == nt.DataTypes.VIDEO:
        fr = cv2.VideoCapture(path)
        fps = fr.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
        frame_count = int(fr.get(cv2.CAP_PROP_FRAME_COUNT))
        dur = frame_count/fps
        return cv2.VideoCapture(path), dur
    elif feature_type == nt.DataTypes.AUDIO:
        return NotImplementedError('Filereader for audio features is not yet implemented')
    elif feature_type == nt.DataTypes.FEATURE:
        stream = Stream(path)
        return stream, stream.data.shape[0] / stream.sr

def close_file_reader(reader, feature_type):
    if feature_type == nt.DataTypes.VIDEO:
        return reader.release()

def get_video_resolution(path):
    vcap = cv2.VideoCapture(path)
    # get vcap property
    width = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vcap.release()
    return (height, width)
