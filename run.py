import tensorflow as tf
from scipy.io import wavfile
import os
import scipy.signal as sps
import numpy as np
import IPython
import librosa
import sounddevice as sd
from scipy.io.wavfile import write


model_path = './Persian_Numbers_Model3.keras'
model = tf.keras.models.load_model(model_path, compile=False)
class_names = np.arange(0, 10, dtype='int')

def predict(wave):
#   wave, _ = librosa.load(wave_adr, sr=16000)
  spectrogram = tf.signal.stft(wave, frame_length = 256, frame_step = 128)
  spectrogram = tf.abs(spectrogram)
  spectrogram = spectrogram[tf.newaxis, ..., tf.newaxis]
  prediction = np.argmax(model.predict(spectrogram))
  return class_names[prediction]

duration = 2
sample_rate = 16000  

model = tf.keras.models.load_model(model_path)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

def record_audio(duration, sample_rate):
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    audio = np.squeeze(audio)
    return audio

def real_time_classification():
    while True:
        audio = record_audio(duration, sample_rate)
        label = predict(audio)
        print(f'Predicted Label: {label}')

real_time_classification()
