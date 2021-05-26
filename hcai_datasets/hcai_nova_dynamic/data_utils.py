import cv2
import numpy as np
import tensorflow_datasets as tfds
from hcai_datasets.hcai_nova_dynamic.defines import DataTypes

def frame_to_time(sr: int, frame: int):
  return frame / sr

def time_to_frame(sr: int, time_s: float):

  # Last frame will not be included if the product is not an int!
  return int(time_s * sr)

def chunk_vid(vcap: cv2.VideoCapture, start_frame: int, end_frame: int):
  vcap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

  width = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
  heigth = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  depth = 3
  length = end_frame - start_frame

  chunk = np.zeros( (length, heigth, width, depth), dtype=np.uint8)

  for i in range(length):
    ret, frame = vcap.read()

    if not ret:
      raise IndexError('Video frame {} out of range'.format(i))

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    chunk[i] = frame

  return chunk

def open_file_reader(path, feature_type):
  if feature_type == DataTypes.VIDEO:
    return cv2.VideoCapture(path)

def close_file_reader(reader, feature_type):
  if feature_type == DataTypes.VIDEO:
    return reader.release()

if __name__ == '__main__':
  test_vid_path_in = r'Z:\nova\data\DFG_A1_A2b\NP001\caretaker.video.mp4'
  test_vid_path_out = r'chunk.mp4'
  fps = 25
  start_time_seconds = 1
  end_time_seconds = 15.5
  vcap = cv2.VideoCapture(test_vid_path_in)
  start_frame = time_to_frame(fps, start_time_seconds)
  end_frame = time_to_frame(fps, end_time_seconds)
  chunk = chunk_vid(vcap, start_frame, end_frame)

  from matplotlib import pyplot as plt
  for i, f in enumerate(chunk):
    if i % fps == 0:
      plt.imshow(f)
      plt.show()
