"""Step 2: Convert every audio file into an MFCC matrix and save to disk.
Run:  python 01_extract_features.py
Opt:  python 01_extract_features.py --augment   (adds noise/pitch/stretch copies of TRAIN actors only)
"""
import argparse, glob, os
import numpy as np
import librosa
from tqdm import tqdm

EMOTIONS = {'01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
            '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'}
SR = 22050
N_MFCC = 40
MAX_LEN = 130          # 3 seconds of audio -> 130 MFCC time-frames
TRAIN_ACTORS = 18      # actors 1-18 get augmented (never validation/test actors)


def get_mfcc(y, sr=SR):
    m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    m = m[:, :MAX_LEN]
    if m.shape[1] < MAX_LEN:                       # pad short clips with zeros
        m = np.pad(m, ((0, 0), (0, MAX_LEN - m.shape[1])))
    return m.T                                     # shape (130, 40) = (time, features)


def augmented_versions(y, sr):
    noise = y + 0.005 * np.random.randn(len(y))
    pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
    stretch = librosa.effects.time_stretch(y, rate=0.9)
    return [noise, pitch, stretch]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/Audio_Speech_Actors_01-24")
    ap.add_argument("--out", default="data/features.npz")
    ap.add_argument("--augment", action="store_true")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, "**", "*.wav"), recursive=True))
    assert files, f"No .wav files in {args.data_dir}"
    print(f"Found {len(files)} files")

    X, labels, actors = [], [], []
    for f in tqdm(files):
        parts = os.path.basename(f).split('.')[0].split('-')
        emotion, actor = EMOTIONS[parts[2]], int(parts[6])
        y, sr = librosa.load(f, sr=SR, duration=3, offset=0.5)
        X.append(get_mfcc(y, sr)); labels.append(emotion); actors.append(actor)
        if args.augment and actor <= TRAIN_ACTORS:
            for ya in augmented_versions(y, sr):
                X.append(get_mfcc(ya, sr)); labels.append(emotion); actors.append(actor)

    X = np.array(X, dtype=np.float32)
    np.savez_compressed(args.out, X=X, labels=np.array(labels), actors=np.array(actors))
    print("Saved", args.out, "| X shape:", X.shape)


if __name__ == "__main__":
    main()