"""Validate poultry-focus gating: background ignored, birds scored."""

import random
from pathlib import Path

import librosa
import numpy as np
import joblib

from features import SAMPLE_RATE
from poultry_focus import decide_with_poultry_focus, poultry_focus_metrics
from realtime_infer import classify_array

bundle = joblib.load("model/poultry_classifier.joblib")
CLIP = int(3 * SAMPLE_RATE)


def eval_file(path):
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    if len(y) < CLIP:
        y = np.pad(y, (0, CLIP - len(y)))
    else:
        y = y[:CLIP]
    _, _, proba = classify_array(y, bundle)
    metrics = poultry_focus_metrics(y)
    label, conf, uncertain, poultry = decide_with_poultry_focus(proba, metrics)
    return label, conf, poultry, metrics["focus_score"]


def main():
    random.seed(3)
    for cls in ["healthy", "unhealthy", "noise"]:
        files = list((Path("data") / cls).glob("*.wav"))
        sample = random.sample(files, min(15, len(files)))
        correct = poultry_n = 0
        for f in sample:
            label, conf, poultry, focus = eval_file(f)
            poultry_n += int(poultry)
            # For noise class, success = not poultry / label noise
            if cls == "noise":
                correct += int(label == "noise" and not poultry)
            else:
                correct += int(label == cls and poultry)
        print(
            f"{cls}: decision_acc={correct}/{len(sample)} "
            f"poultry_flagged={poultry_n}/{len(sample)}"
        )

    farm = "freesound_community-rooster-and-chicks-87843.mp3"
    y, _ = librosa.load(farm, sr=SAMPLE_RATE, mono=True)
    # multi window
    from app import load_bundle
    import app as A

    A.bundle = load_bundle()
    result = A.classify_waveform(y, full_scan=True)
    print(
        "farm_upload",
        result["label"],
        result["confidence"],
        "poultry_windows",
        result["poultry_windows"],
        "bg_ignored",
        result["background_windows_ignored"],
    )

    try:
        import sounddevice as sd

        audio = sd.rec(CLIP, samplerate=SAMPLE_RATE, channels=1, dtype="float32")
        sd.wait()
        label, conf, uncertain, poultry, focus = eval_file_array(audio.flatten())
        print("room_mic", label, conf, "poultry", poultry, "focus", round(focus, 3), "uncertain", uncertain)
    except Exception as e:
        print("room_failed", e)


def eval_file_array(y):
    if len(y) < CLIP:
        y = np.pad(y, (0, CLIP - len(y)))
    else:
        y = y[:CLIP]
    _, _, proba = classify_array(y, bundle)
    metrics = poultry_focus_metrics(y)
    return decide_with_poultry_focus(proba, metrics) + (metrics["focus_score"],)


if __name__ == "__main__":
    main()
