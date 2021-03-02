from hcai_librispeech import HcaiLibrispeech
import tensorflow_datasets as tfds
import tensorflow as tf
import librosa
import soundfile as sf

def pp(x,y):
    file_path = bytes.decode(x.numpy())
    label = bytes.decode(y.numpy())
    audio, sr = librosa.load(file_path, dtype='float32', sr=16000, mono=True)
    return audio, label

ds, ds_info = tfds.load(
    'hcai_librispeech',
    split='dev-clean',
    with_info=True,
    as_supervised=True,
    decoders={
       'speech': tfds.decode.SkipDecoding()
    }
)

#"speech": tfds.features.Text(),
#"text": tfds.features.Text(),
#"speaker_id": tf.int64,
#"chapter_id": tf.int64,
#"id": tf.string,

ds = ds.map(lambda x,y : (tf.py_function(func=pp, inp=[x, y], Tout=[tf.float32, tf.string])))

print('')
audio, label = next(ds.as_numpy_iterator())

sf.write('test.flac', audio, 16000, format='flac')
