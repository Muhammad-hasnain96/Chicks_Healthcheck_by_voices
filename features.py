"""
features.py
Audio feature extraction for the poultry sound classifier.
Designed to be fast and CPU-only (no deep learning), so the same
functions are used both at training time and at real-time inference time.
"""

import numpy as np
import librosa

SAMPLE_RATE = 16000        # all audio is resampled to this
CLIP_DURATION = 3.0        # seconds - long recordings get chunked into this
N_MFCC = 20


def load_and_chunk(path, sr=SAMPLE_RATE, clip_duration=CLIP_DURATION, hop_ratio=0.5):
    """
    Load an audio file and split it into fixed-length overlapping chunks.
    Short clips (<= clip_duration) are returned as a single, zero-padded chunk.
    Returns a list of 1D numpy arrays.
    """
    y, _ = librosa.load(path, sr=sr, mono=True)
    clip_len = int(clip_duration * sr)

    if len(y) <= clip_len:
        # pad short clips so all feature vectors are computed the same way
        y = np.pad(y, (0, max(0, clip_len - len(y))))
        return [y]

    hop = int(clip_len * hop_ratio)
    chunks = []
    for start in range(0, len(y) - clip_len + 1, hop):
        chunks.append(y[start:start + clip_len])
    # make sure the tail of the recording isn't dropped
    if (len(y) - clip_len) % hop != 0:
        chunks.append(y[-clip_len:])
    return chunks


def extract_features(y, sr=SAMPLE_RATE):
    """
    Turn a 1D audio array into a fixed-length feature vector.
    Combines MFCCs + spectral descriptors + energy, each summarized
    with mean and std to keep the vector compact and robust to timing shifts.
    """
    # guard against completely silent/empty chunks
    if np.max(np.abs(y)) < 1e-6:
        y = y + 1e-6

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    delta_mfcc = librosa.feature.delta(mfcc)

    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    def summarize(feat):
        return np.concatenate([feat.mean(axis=1), feat.std(axis=1)])

    vector = np.concatenate([
        summarize(mfcc),
        summarize(delta_mfcc),
        summarize(spec_centroid),
        summarize(spec_bandwidth),
        summarize(spec_rolloff),
        summarize(zcr),
        summarize(rms),
        summarize(chroma),
    ])
    return vector.astype(np.float32)


def feature_names():
    """Human-readable names matching extract_features' output order (for feature importance plots)."""
    groups = [
        ("mfcc", N_MFCC), ("dmfcc", N_MFCC), ("centroid", 1), ("bandwidth", 1),
        ("rolloff", 1), ("zcr", 1), ("rms", 1), ("chroma", 12),
    ]
    names = []
    for gname, n in groups:
        for i in range(n):
            names.append(f"{gname}{i}_mean")
        for i in range(n):
            names.append(f"{gname}{i}_std")
    return names
