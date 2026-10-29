"""
realtime_infer.py

Real-time / on-device inference for the poultry sound classifier.

Two modes:

1. Live microphone monitoring (for the farm device, e.g. Raspberry Pi + USB mic):
    python realtime_infer.py --mode live --model ./model/poultry_classifier.joblib

2. One-off prediction on a file (for testing):
    python realtime_infer.py --mode file --model ./model/poultry_classifier.joblib --file clip.wav

Live mode listens continuously, classifies every WINDOW_SECONDS of audio,
and prints a rolling result. It's deliberately simple (no threading tricks)
so it stays lightweight enough for a Raspberry Pi class device.
"""

import argparse
import time
import numpy as np
import joblib

from features import extract_features, SAMPLE_RATE

WINDOW_SECONDS = 3.0
CONFIDENCE_THRESHOLD = 0.55   # below this, report "uncertain" instead of guessing


def classify_array(y, bundle):
    feats = extract_features(y, sr=SAMPLE_RATE).reshape(1, -1)
    feats_scaled = bundle["scaler"].transform(feats)
    proba = bundle["model"].predict_proba(feats_scaled)[0]
    classes = bundle["label_encoder"].classes_
    top_idx = np.argmax(proba)
    label = classes[top_idx]
    confidence = proba[top_idx]
    return label, confidence, dict(zip(classes, proba))


def run_file(args, bundle):
    import librosa
    y, _ = librosa.load(args.file, sr=SAMPLE_RATE, mono=True)
    clip_len = int(WINDOW_SECONDS * SAMPLE_RATE)
    if len(y) < clip_len:
        y = np.pad(y, (0, clip_len - len(y)))
    else:
        y = y[:clip_len]
    label, conf, proba = classify_array(y, bundle)
    print(f"Prediction: {label}  (confidence {conf:.2f})")
    print("Full distribution:", {k: round(v, 3) for k, v in proba.items()})


def run_live(args, bundle):
    try:
        import sounddevice as sd
    except ImportError:
        raise SystemExit(
            "Live mode needs the 'sounddevice' package: pip install sounddevice\n"
            "On Raspberry Pi you may also need: sudo apt-get install libportaudio2"
        )

    clip_len = int(WINDOW_SECONDS * SAMPLE_RATE)
    print(f"Listening... classifying every {WINDOW_SECONDS:.0f}s window. Ctrl+C to stop.\n")

    while True:
        audio = sd.rec(clip_len, samplerate=SAMPLE_RATE, channels=1, dtype="float32")
        sd.wait()
        y = audio.flatten()

        label, conf, proba = classify_array(y, bundle)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        if conf < CONFIDENCE_THRESHOLD:
            print(f"[{timestamp}] UNCERTAIN (top guess: {label}, conf {conf:.2f})")
        else:
            tag = {"healthy": "OK", "unhealthy": "!! ALERT !!", "noise": "-- no birds detected --"}[label]
            print(f"[{timestamp}] {label.upper():10s} conf={conf:.2f}  {tag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["live", "file"], required=True)
    ap.add_argument("--model", default="./model/poultry_classifier.joblib")
    ap.add_argument("--file", help="audio file path (required for --mode file)")
    args = ap.parse_args()

    bundle = joblib.load(args.model)

    if args.mode == "file":
        if not args.file:
            raise SystemExit("--file is required for --mode file")
        run_file(args, bundle)
    else:
        run_live(args, bundle)


if __name__ == "__main__":
    main()
