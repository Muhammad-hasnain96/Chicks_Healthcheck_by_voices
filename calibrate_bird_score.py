"""Calibrate bird vocalization activity score on labeled clips."""

import random
from pathlib import Path

import librosa
import numpy as np

from features import SAMPLE_RATE


def bird_score(y, sr=SAMPLE_RATE):
    y = y.astype(np.float32)
    rms = float(np.sqrt(np.mean(y**2)) + 1e-12)
    peak = float(np.max(np.abs(y)) + 1e-12)
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    total = S.sum() + 1e-12
    bird_band = float(S[(freqs >= 1200) & (freqs <= 6500)].sum() / total)
    rumble = float(S[freqs < 400].sum() / total)
    onset = librosa.onset.onset_strength(y=y, sr=sr)
    onset_mean = float(np.mean(onset))
    onset_peak = float(np.max(onset) + 1e-12)
    score = (
        0.35 * bird_band
        + 0.20 * max(0.0, bird_band - rumble)
        + 0.20 * min(1.0, onset_mean / 3.0)
        + 0.15 * min(1.0, onset_peak / 8.0)
        + 0.10 * min(1.0, peak / 0.15)
    )
    return {
        "score": score,
        "bird_band": bird_band,
        "rumble": rumble,
        "rms": rms,
        "onset_mean": onset_mean,
        "peak": peak,
    }


def main():
    random.seed(0)
    clip = int(3 * SAMPLE_RATE)
    for cls in ["healthy", "unhealthy", "noise"]:
        files = list((Path("data") / cls).glob("*.wav"))
        sample = random.sample(files, min(20, len(files)))
        scores = []
        for f in sample:
            y, _ = librosa.load(f, sr=SAMPLE_RATE, mono=True)
            if len(y) < clip:
                y = np.pad(y, (0, clip - len(y)))
            else:
                y = y[:clip]
            scores.append(bird_score(y)["score"])
        print(
            cls,
            "mean",
            round(float(np.mean(scores)), 3),
            "p25",
            round(float(np.percentile(scores, 25)), 3),
            "p50",
            round(float(np.percentile(scores, 50)), 3),
            "p75",
            round(float(np.percentile(scores, 75)), 3),
            "p10",
            round(float(np.percentile(scores, 10)), 3),
        )

    y, _ = librosa.load(
        "freesound_community-rooster-and-chicks-87843.mp3", sr=SAMPLE_RATE, mono=True
    )
    print("farm_first3", bird_score(y[:clip]))
    print("farm_full_chunked_mean", float(np.mean([
        bird_score(y[i : i + clip])["score"]
        for i in range(0, max(1, len(y) - clip + 1), clip // 2)
    ])))

    try:
        import sounddevice as sd

        audio = sd.rec(clip, samplerate=SAMPLE_RATE, channels=1, dtype="float32")
        sd.wait()
        print("room", bird_score(audio.flatten()))
    except Exception as e:
        print("room_failed", e)


if __name__ == "__main__":
    main()
