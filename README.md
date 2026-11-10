# Emotion Recognition from Speech

## Dataset
RAVDESS speech audio: 2880 clips, 24 actors, 8 emotions.

## Method
- Features: 40 MFCCs x 130 time frames
- Models compared: CNN, LSTM, CNN+LSTM
- Split by actor: actors 1-18 train, 19-20 validation, 21-24 test

## Results (4 unseen actors)
| Model    | Accuracy |
|----------|----------|
| CNN      | 54.6%    |
| CNN+LSTM | 42.1%    |
| LSTM     | 34.6%    |

Best model: CNN. Chance level is 12.5%.
Accuracy is lower than many online results because they use random splits,
where the same actor appears in both train and test.

## How to run
pip install -r requirements.txt
python extract_features.py
python train.py --model cnn
python predict.py path/to/audio.wav

## Files
The results folder has confusion matrices and training curves.
