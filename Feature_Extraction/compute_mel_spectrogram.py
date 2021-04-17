"""
    This python code is for computing mel-scale spectrogram for audio clips

        Input: a folder containing consecutive audio clips
        Output: corresponding audiowave and mel-scale spectrogram for each audio clip

    Author: Yingbo Ma
    Data: Jan 10th, 2021
"""

import librosa
import librosa.display
import os
import numpy as np
import matplotlib.pyplot as plt

audio_folder_path = r'...' # this is the folder containing consecutive audio clips in the format of "0.wav, 1.wav, 2.wav, ..."
list = os.listdir(audio_folder_path)

for index in range(len(list)):
    audio_clip = audio_folder_path + str(index+1400) + ".wav"
    signal, sample_rate = librosa.load(audio_clip)
    librosa.display.waveplot(signal, sr=sample_rate);
    plt.show() # plot the audio waveform in time domain

    n_fft = 2048
    hop_length = 512

    n_mels = 128
    mel = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels)

    S = librosa.feature.melspectrogram(signal, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_DB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_DB, sr=sample_rate, hop_length=hop_length, x_axis='time', y_axis='mel');
    plt.colorbar(format='%+2.0f dB');
    plt.show() # plot the audio mel-scale spectrogram