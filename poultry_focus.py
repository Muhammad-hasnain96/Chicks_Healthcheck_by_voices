"""
Poultry vocalization focus helpers.

Goal: ignore barn/room background (fans, rumble, human talk, cats/other animals)
and only treat windows that contain hen/chick-like energy as health decisions.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt

from features import SAMPLE_RATE

_BIRD_SOS = butter(4, [1000.0, 6500.0], btype="band", fs=SAMPLE_RATE, output="sos")


def bandpass_poultry(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    return sosfilt(_BIRD_SOS, y.astype(np.float64)).astype(np.float32)


def _rms(y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y.astype(np.float64)))) + 1e-12)


def librosa_stft(y: np.ndarray, sr: int) -> np.ndarray:
    import librosa

    return np.abs(librosa.stft(y, n_fft=1024, hop_length=256))


def _pitchiness(y: np.ndarray, sr: int) -> float:
    step = max(1, sr // 4000)
    x = y[::step].astype(np.float64)
    if x.size < 64:
        return 0.0
    x = x - np.mean(x)
    denom = float(np.dot(x, x) + 1e-12)
    eff_sr = sr / step
    min_lag = max(1, int(eff_sr / 300))
    max_lag = min(len(x) - 1, int(eff_sr / 80))
    if max_lag <= min_lag:
        return 0.0
    best = 0.0
    for lag in range(min_lag, max_lag + 1):
        corr = float(np.dot(x[:-lag], x[lag:]) / denom)
        if corr > best:
            best = corr
    return float(np.clip(best, 0.0, 1.0))


def _strong_peep(metrics: dict) -> bool:
    return metrics["chirp_band"] >= 0.45 or (
        metrics["focus_score"] >= 0.90 and metrics["chirp_band"] >= 0.35
    )


def poultry_focus_metrics(y: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    import librosa

    y = np.asarray(y, dtype=np.float32)
    raw_rms = _rms(y)
    focused = bandpass_poultry(y, sr)
    focus_rms = _rms(focused)
    focus_ratio = focus_rms / raw_rms

    S = np.abs(librosa_stft(y, sr))
    freqs = np.fft.rfftfreq(1024, d=1.0 / sr)
    total = float(S.sum() + 1e-12)

    speech_band = float(S[(freqs >= 300) & (freqs <= 3000)].sum() / total)
    mammal_band = float(S[(freqs >= 250) & (freqs <= 1800)].sum() / total)
    chirp_band = float(S[(freqs >= 2500) & (freqs <= 7000)].sum() / total)
    bird_band = float(S[(freqs >= 1200) & (freqs <= 6500)].sum() / total)
    rumble = float(S[freqs < 400].sum() / total)
    band_vs_rumble = bird_band / (rumble + 0.08)
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))

    frame = max(1, int(0.02 * sr))
    hop = max(1, frame // 2)
    env = [
        _rms(focused[i : i + frame])
        for i in range(0, max(1, len(focused) - frame + 1), hop)
    ]
    env = np.asarray(env, dtype=np.float64) if env else np.zeros(1)
    burst = float(np.std(env) / (np.mean(env) + 1e-12))
    pitchiness = _pitchiness(y, sr)

    score = (
        0.35 * float(np.clip(focus_ratio, 0, 1))
        + 0.25 * float(np.clip(chirp_band * 2.0, 0, 1))
        + 0.20 * float(np.clip(band_vs_rumble / 4.0, 0, 1))
        + 0.10 * float(np.clip(burst / 2.0, 0, 1))
        + 0.10 * float(np.clip(bird_band, 0, 1))
    )
    return {
        "focus_score": float(score),
        "focus_ratio": float(focus_ratio),
        "bird_band": bird_band,
        "chirp_band": chirp_band,
        "speech_band": speech_band,
        "mammal_band": mammal_band,
        "rumble": rumble,
        "burst": burst,
        "pitchiness": pitchiness,
        "centroid": centroid,
        "raw_rms": raw_rms,
        "focus_rms": focus_rms,
    }


def looks_like_non_poultry(metrics: dict, bird_mass: float = 0.0) -> bool:
    """Human talk, cat meows, and similar non-poultry voices."""
    # Strong chick peeps are never treated as non-poultry
    if _strong_peep(metrics):
        return False

    speech = metrics["speech_band"]
    chirp = metrics["chirp_band"]
    pitch = metrics["pitchiness"]
    mammal = metrics["mammal_band"]
    centroid = metrics["centroid"]

    # Cat / mammal meow
    if mammal >= 0.40 and chirp <= 0.36 and mammal >= chirp * 1.15:
        return True
    if centroid > 0 and centroid < 2300 and mammal >= 0.38 and chirp <= 0.38:
        return True
    if speech >= 0.55 and chirp <= 0.35 and centroid < 2500:
        return True

    if bird_mass >= 0.88:
        return False

    if speech >= 0.40 and chirp <= 0.22 and speech >= chirp * 1.6:
        return True
    if pitch >= 0.38 and chirp <= 0.24 and speech >= 0.34 and bird_mass < 0.82:
        return True
    if 0.40 <= bird_mass < 0.85 and speech >= 0.36 and chirp <= 0.26:
        return True
    return False


def is_poultry_window(metrics: dict, bird_mass: float, noise_mass: float) -> bool:
    if looks_like_non_poultry(metrics, bird_mass):
        return False

    strong_peep = _strong_peep(metrics)
    if noise_mass >= bird_mass and not strong_peep:
        return False
    if noise_mass >= 0.50 and not strong_peep:
        return False

    focus = metrics["focus_score"]
    fr = metrics["focus_ratio"]
    chirp = metrics["chirp_band"]

    if strong_peep or fr >= 0.78 or chirp >= 0.45:
        return True

    if (
        bird_mass >= 0.80
        and (bird_mass - noise_mass) >= 0.40
        and chirp >= 0.18
        and metrics["mammal_band"] < 0.50
    ):
        return True

    if (
        bird_mass >= noise_mass + 0.20
        and bird_mass >= 0.62
        and chirp >= 0.30
        and focus >= 0.36
        and metrics["speech_band"] < chirp * 1.8
        and metrics["mammal_band"] < chirp * 1.5
    ):
        return True

    return False


def decide_with_poultry_focus(proba: dict, metrics: dict) -> tuple[str, float, bool, bool]:
    healthy = float(proba.get("healthy", 0.0))
    unhealthy = float(proba.get("unhealthy", 0.0))
    noise = float(proba.get("noise", 0.0))
    bird_mass = healthy + unhealthy
    top_bird = max(healthy, unhealthy)
    poultry = is_poultry_window(metrics, bird_mass, noise)
    strong_peep = _strong_peep(metrics)

    # Only demote when noise wins AND peeps are not acoustically obvious
    if poultry and noise >= top_bird and not strong_peep:
        poultry = False

    if not poultry:
        conf = max(noise, 1.0 - bird_mass)
        if looks_like_non_poultry(metrics, bird_mass):
            conf = max(conf, 0.62)
        return "noise", float(conf), conf < 0.55, False

    if healthy >= unhealthy:
        label, conf = "healthy", healthy
    else:
        label, conf = "unhealthy", unhealthy
    if strong_peep and conf < 0.55:
        conf = min(0.72, conf + 0.18)
    rel = conf / max(bird_mass, 1e-6)
    confidence = float(0.65 * conf + 0.35 * rel)
    uncertain = confidence < 0.45
    return label, confidence, uncertain, True
