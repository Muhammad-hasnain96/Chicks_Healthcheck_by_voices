import librosa
from app import load_bundle, classify_waveform
import app as A

A.bundle = load_bundle()
tests = [
    ("cat", "data/noise/pv_virtual_vibes-cat-meow-sound-383823.mp3"),
    ("chick1", "test_clips/chicks/chick1_stuffings.wav"),
    ("chick_chirp", "test_clips/chicks/chick_that_chirp.wav"),
    ("farm", "freesound_community-rooster-and-chicks-87843.mp3"),
]
for name, path in tests:
    y, _ = librosa.load(path, sr=16000, mono=True)
    r = classify_waveform(y, True)
    print(
        name,
        "->",
        r["label"],
        f"conf={r['confidence']:.2f}",
        "poultry",
        r["poultry_windows"],
        "bg",
        r["background_windows_ignored"],
        "dist",
        r["distribution"],
    )
