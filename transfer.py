import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2

# transfer all the spectrograms to a new folder
path = r'C:\Users\sean8_q3zpzqf\Downloads\DS_10283_3336\LA\ASVspoof2019_LA_train\flac'
new_path = r'C:\Users\sean8_q3zpzqf\Downloads\DS_10283_3336\LA\ASVspoof2019_LA_train\spectrogram'
plt.clf()
plt.close()  # 關閉圖形

for i in os.listdir(path):
    y, sr = librosa.load(path+'/'+ i)

    # Convert to spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)

    # Convert to decibels
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Plot the spectrogram
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_db, origin='lower', cmap='jet', aspect='auto')
    plt.axis('off')

    # Save the spectrogram image
    plt.savefig(new_path+'/'+ i+'.png', bbox_inches='tight', pad_inches=0)
    plt.clf()
    plt.close()  # 關閉圖形


