"""Step 3: Train the CNN+LSTM model and evaluate it.
Run: python 02_train.py            (or: python 02_train.py --model cnn / lstm for comparison)
"""
import argparse, json, os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

ap = argparse.ArgumentParser()
ap.add_argument("--features", default="data/features.npz")
ap.add_argument("--model", choices=["cnn_lstm", "cnn", "lstm"], default="cnn_lstm")
ap.add_argument("--epochs", type=int, default=80)
args = ap.parse_args()
os.makedirs("results", exist_ok=True); os.makedirs("models", exist_ok=True)

# ---------- Load ----------
d = np.load(args.features)
X, labels, actors = d["X"], d["labels"], d["actors"]
le = LabelEncoder()
y = to_categorical(le.fit_transform(labels))
n_classes = len(le.classes_)

# ---------- Split BY ACTOR (no speaker leakage) ----------
train = actors <= 18          # actors 1-18  -> train
val = (actors >= 19) & (actors <= 20)   # actors 19-20 -> validation (early stopping)
test = actors >= 21           # actors 21-24 -> final test
X_train, y_train = X[train], y[train]
X_val, y_val = X[val], y[val]
X_test, y_test = X[test], y[test]
print(f"Train {len(X_train)} | Val {len(X_val)} | Test {len(X_test)}")

# ---------- Normalise (stats from TRAIN only) ----------
mean, std = X_train.mean(), X_train.std()
X_train, X_val, X_test = [(a - mean) / std for a in (X_train, X_val, X_test)]

# ---------- Model ----------
inp = layers.Input(shape=X_train.shape[1:])
if args.model == "cnn_lstm":
    x = layers.Conv1D(64, 5, activation='relu', padding='same')(inp)
    x = layers.BatchNormalization()(x); x = layers.MaxPooling1D(2)(x); x = layers.Dropout(0.3)(x)
    x = layers.Conv1D(128, 5, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x); x = layers.MaxPooling1D(2)(x); x = layers.Dropout(0.3)(x)
    x = layers.LSTM(64)(x)
elif args.model == "cnn":
    x = layers.Conv1D(64, 5, activation='relu', padding='same')(inp)
    x = layers.BatchNormalization()(x); x = layers.MaxPooling1D(2)(x); x = layers.Dropout(0.3)(x)
    x = layers.Conv1D(128, 5, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x); x = layers.GlobalAveragePooling1D()(x)
else:  # lstm
    x = layers.LSTM(128, return_sequences=True)(inp)
    x = layers.LSTM(64)(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(64, activation='relu')(x)
out = layers.Dense(n_classes, activation='softmax')(x)
model = models.Model(inp, out)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ---------- Train ----------
hist = model.fit(
    X_train, y_train, validation_data=(X_val, y_val),
    epochs=args.epochs, batch_size=32,
    callbacks=[EarlyStopping(patience=10, restore_best_weights=True),
               ReduceLROnPlateau(patience=4, factor=0.5, verbose=1)])

# ---------- Curves ----------
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(hist.history['loss'], label='train'); ax[0].plot(hist.history['val_loss'], label='val')
ax[0].set_title('Loss'); ax[0].legend()
ax[1].plot(hist.history['accuracy'], label='train'); ax[1].plot(hist.history['val_accuracy'], label='val')
ax[1].set_title('Accuracy'); ax[1].legend()
plt.savefig(f"results/curves_{args.model}.png", dpi=120); plt.close()

# ---------- Evaluate on unseen actors ----------
pred = model.predict(X_test).argmax(1); true = y_test.argmax(1)
acc = (pred == true).mean()
print(f"\nTEST ACCURACY ({args.model}): {acc:.4f}\n")
print(classification_report(true, pred, target_names=le.classes_))
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(true, pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted'); plt.ylabel('True'); plt.title(f'Confusion matrix - {args.model}')
plt.tight_layout(); plt.savefig(f"results/confusion_{args.model}.png", dpi=120); plt.close()

# ---------- Save ----------
model.save(f"models/emotion_{args.model}.keras")
json.dump({"mean": float(mean), "std": float(std), "classes": list(le.classes_)},
          open("models/preprocess.json", "w"))
with open("results/results.txt", "a") as f:
    f.write(f"{args.model}: test accuracy = {acc:.4f}\n")
print("Saved model, plots and results.")