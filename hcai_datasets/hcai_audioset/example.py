from hcai_audioset import HcaiAudioset
import tensorflow_datasets as tfds
import tensorflow as tf
import pydub
import soundfile as sf
import numpy as np


def pp(x,y):
    file_path = bytes.decode(x.numpy())
    print(file_path)
    ext = file_path.split('.')[-1]

    a = pydub.AudioSegment.from_file(file_path, ext)
    a = a.set_frame_rate(16000)
    a = a.set_channels(1)
    a = np.array(a.get_array_of_samples())
    a = a.astype(np.int16)

    return a, y

ds, ds_info = tfds.load(
    'hcai_audioset',
    split='train',
    with_info=True,
    as_supervised=True,
    decoders={
       'audio': tfds.decode.SkipDecoding()
    }
)

ds = ds.map(lambda x,y : (tf.py_function(func=pp, inp=[x, y], Tout=[tf.int16, tf.int64])))

print('')
audio, label = next(ds.as_numpy_iterator())

sf.write('test.wav', audio, 16000, format='wav')
