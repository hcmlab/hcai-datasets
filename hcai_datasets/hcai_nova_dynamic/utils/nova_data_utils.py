import cv2
import numpy as np
import tensorflow_datasets as tfds
from hcai_datasets.hcai_nova_dynamic.utils import nova_data_types as ndt
from hcai_datasets.hcai_nova_dynamic.utils.ssi_stream_utils import Stream
from typing import Union

def frame_to_seconds(sr: int, frame: int) -> float:
    return frame / sr


def seconds_to_frame(sr: int, time_s: float) -> int:
    return round(time_s * sr)


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


def chunk_stream(stream: Stream, start_frame: int, end_frame: int):
    return stream.data[start_frame:end_frame]


def open_file_reader(path, feature_type):
    if feature_type == ndt.DataTypes.video:
        return cv2.VideoCapture(path)
    elif feature_type == ndt.DataTypes.audio:
        return NotImplementedError('Filereader for audio features is not yet implemented')
    elif feature_type == ndt.DataTypes.feature:
        return Stream(path)


def close_file_reader(reader, feature_type):
    if feature_type == ndt.DataTypes.video:
        return reader.release()


def get_video_resolution(path):
    vcap = cv2.VideoCapture(path)
    # get vcap property
    width = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vcap.release()
    return (height, width)
