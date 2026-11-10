"""Step 1: Explore the dataset and plot waveform + spectrogram + MFCC.
Run: python 00_explore.py
"""
import glob, os
import numpy as np
import pandas as pd
import librosa, librosa.display
import matplotlib.pyplot as plt

DATA_DIR = "data/Audio_Speech_Actors_01-24"
EMOTIONS = {'01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
            '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'}

files = glob.glob(os.path.join(DATA_DIR, "**", "*.wav"), recursive=True)
print("Total audio files found:", len(files))
assert files, "No .wav files found. Check DATA_DIR path!"

rows = []
for f in files:
    p = os.path.basename(f).split('.')[0].split('-')
    rows.append((f, EMOTIONS[p[2]], int(p[6])))
df = pd.DataFrame(rows, columns=['path', 'emotion', 'actor'])
print(df.emotion.value_counts())

# Plot waveform, mel-spectrogram and MFCC for 3 emotions
chosen = ['happy', 'sad', 'angry']
fig, axes = plt.subplots(3, 3, figsize=(15, 9))
for i, emo in enumerate(chosen):
    path = df[df.emotion == emo].path.iloc[0]
    y, sr = librosa.load(path, duration=3, offset=0.5)
    librosa.display.waveshow(y, sr=sr, ax=axes[i, 0])
    axes[i, 0].set_title(f"{emo} - waveform")
    S = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr), ref=np.max)
    librosa.display.specshow(S, sr=sr, x_axis='time', y_axis='mel', ax=axes[i, 1])
    axes[i, 1].set_title(f"{emo} - mel spectrogram")
    m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    librosa.display.specshow(m, x_axis='time', ax=axes[i, 2])
    axes[i, 2].set_title(f"{emo} - MFCC")
plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/explore.png", dpi=120)
print("Saved results/explore.png")
plt.show()