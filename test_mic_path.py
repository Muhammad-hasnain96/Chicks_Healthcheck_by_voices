"""Accuracy + microphone path tests for FlockTone live classify-chunk API."""

import io
import json
import random
import wave
from pathlib import Path

import librosa
import numpy as np
import urllib.request

SR = 16000
WIN = 3 * SR
API = "http://127.0.0.1:7860/api/classify-chunk"
BASE = Path(__file__).resolve().parent


def to_wav_bytes(y: np.ndarray) -> bytes:
    y = np.asarray(y, dtype=np.float32)
    if len(y) < WIN:
        y = np.pad(y, (0, WIN - len(y)))
    else:
        y = y[:WIN]
    pcm = (np.clip(y, -1, 1) * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def post_chunk(wav_bytes: bytes) -> dict:
    boundary = f"----bound{random.randint(1_000_000, 9_999_999)}"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="chunk.wav"\r\n'
        "Content-Type: audio/wav\r\n\r\n"
    ).encode() + wav_bytes + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        API,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def main():
    print("=== Mic-path accuracy (classify-chunk, resampled 16k->48k->16k) ===")
    for cls in ["healthy", "unhealthy", "noise"]:
        files = list((BASE / "data" / cls).glob("*.wav")) + list(
            (BASE / "data" / cls).glob("*.mp3")
        )
        random.seed(42)
        sample = random.sample(files, min(12, len(files)))
        correct = 0
        for f in sample:
            y, _ = librosa.load(f, sr=SR, mono=True)
            y48 = librosa.resample(y, orig_sr=SR, target_sr=48000)
            y16 = librosa.resample(y48, orig_sr=48000, target_sr=SR)
            pred = post_chunk(to_wav_bytes(y16))
            ok = pred["label"] == cls
            correct += int(ok)
            print(
                f"  [{cls}] {f.name[:42]:42s} -> {pred['label']:10s} "
                f"{pred['confidence']:.2f} ok={ok}"
            )
        print(f"  >> {cls}: {correct}/{len(sample)} = {100 * correct / len(sample):.1f}%\n")

    print("=== Farm freesound via mic-path (middle 3s) ===")
    farm = BASE / "freesound_community-rooster-and-chicks-87843.mp3"
    y, _ = librosa.load(farm, sr=SR, mono=True)
    mid = len(y) // 2
    chunk = y[max(0, mid - WIN // 2) : mid + WIN // 2]
    pred = post_chunk(to_wav_bytes(chunk))
    print(
        f"  {pred['label']} conf={pred['confidence']:.2f} "
        f"flock={pred.get('flock_detected')} dist={pred['distribution']}"
    )

    print("\n=== Real microphone capture (3 windows) ===")
    try:
        import sounddevice as sd

        print("  default device:", sd.default.device)
        for i in range(3):
            print(f"  recording window {i + 1}/3 ...")
            audio = sd.rec(WIN, samplerate=SR, channels=1, dtype="float32")
            sd.wait()
            y = audio.flatten()
            rms = float(np.sqrt(np.mean(y**2)))
            pred = post_chunk(to_wav_bytes(y))
            print(
                f"    rms={rms:.4f} -> {pred['label']} "
                f"conf={pred['confidence']:.2f} uncertain={pred['uncertain']} "
                f"flock={pred.get('flock_detected')}"
            )
    except Exception as e:
        print("  mic capture failed:", e)


if __name__ == "__main__":
    main()
