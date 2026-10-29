# Poultry Sound Classifier (healthy / unhealthy / noise)

A lightweight, CPU-only audio classifier for detecting chicken/hen/chick health
status from sound recordings, designed to run on a farm device in real time
(e.g. Raspberry Pi + USB microphone).

## Why this approach

With a few hundred clips per class, a Random Forest trained on handcrafted
audio features (MFCCs, spectral shape, energy, chroma) is more accurate and
far faster than a deep neural net trained from scratch on this amount of
data - and inference takes only a few milliseconds on CPU, which is what
makes real-time monitoring practical on a small device.

## 1. Setup

```bash
pip install -r requirements.txt
```

## 2. Prepare your data

Arrange your existing recordings like this:

```
data/
    healthy/    *.wav (or mp3/flac/ogg/m4a)
    unhealthy/  *.wav
    noise/      *.wav
```

Any clip length works - the pipeline automatically chunks longer recordings
into 3-second windows and pads shorter ones.

## 3. Train

```bash
python train_model.py --data_dir ./data --out_dir ./model
```

This prints 5-fold cross-validated precision/recall per class (a realistic
estimate of accuracy on unseen data, not just training accuracy) and saves:

- `model/poultry_classifier.joblib` - the deployable model
- `model/confusion_matrix.png` - see exactly which classes get confused
- `model/feature_importance.png` - which audio features matter most
- `model/training_report.txt`

**Read the classification report carefully.** If one class (often
"unhealthy", since sick-sound variation is usually the hardest to capture)
has noticeably lower recall, that's a signal to collect more clips for that
class specifically - not to tune the model further.

## 4. Test on a single file

```bash
python realtime_infer.py --mode file --model ./model/poultry_classifier.joblib --file some_clip.wav
```

## 5. Professional web UI (upload + microphone)

```bash
python app.py
```

Open http://127.0.0.1:7860 — upload barn recordings or connect a browser microphone for live 3-second classification.

## 6. Run live on the farm device (CLI)

```bash
python realtime_infer.py --mode live --model ./model/poultry_classifier.joblib
```

This listens continuously through the default microphone, classifies every
3-second window, and prints a timestamped result with a confidence score.
Predictions below 55% confidence are flagged as "UNCERTAIN" rather than
forced into a class - tune `CONFIDENCE_THRESHOLD` in `realtime_infer.py` if
you want it stricter or looser.

## Tips for real-world accuracy

- **Balance matters less than you'd think** - the training script already
  applies class weighting, but if one folder has 5x more files than
  another, try to even that out over time.
- **Mic placement/consistency**: if your training clips were recorded on a
  different microphone or at a different distance than where you'll deploy,
  accuracy will drop. Try to record at least some training data with the
  actual deployment mic/setup.
- **Background noise variety**: make sure your `noise` folder includes the
  actual ambient sounds of your farm (fans, feeders, wind, human voices) -
  not just silence - or the model won't learn to reject them properly.
- **Retraining**: as you collect more real-world clips (especially ones the
  model got wrong), add them to the data folders and re-run `train_model.py`.
