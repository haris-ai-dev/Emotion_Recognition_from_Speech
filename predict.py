"""Step 4: Predict emotion for any .wav file (e.g. your own recording).
Run: python 03_predict.py path/to/audio.wav [--model cnn_lstm]
"""
import argparse, json
import numpy as np, librosa
from tensorflow.keras.models import load_model

N_MFCC, MAX_LEN, SR = 40, 130, 22050

ap = argparse.ArgumentParser()
ap.add_argument("wav")
ap.add_argument("--model", default="cnn_lstm")
args = ap.parse_args()

model = load_model(f"models/emotion_{args.model}.keras")
cfg = json.load(open("models/preprocess.json"))

y, sr = librosa.load(args.wav, sr=SR, duration=3, offset=0.5)
m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)[:, :MAX_LEN]
m = np.pad(m, ((0, 0), (0, MAX_LEN - m.shape[1])))
x = ((m.T - cfg["mean"]) / cfg["std"])[None].astype(np.float32)

probs = model.predict(x, verbose=0)[0]
for i in np.argsort(probs)[::-1]:
    print(f"{cfg['classes'][i]:<10} {probs[i]*100:5.1f}%")
print("\nPredicted emotion:", cfg["classes"][probs.argmax()])