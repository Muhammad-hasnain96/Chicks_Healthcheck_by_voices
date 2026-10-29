"""Live test using this PC's default microphone through the FlockTone API."""

from __future__ import annotations

import io
import json
import random
import threading
import time
import wave
from pathlib import Path

import librosa
import numpy as np
import sounddevice as sd
import urllib.request

SR = 16000
WIN = 3 * SR
API = "http://127.0.0.1:7860/api/classify-chunk"
BASE = Path(__file__).resolve().parent
FARM = BASE / "freesound_community-rooster-and-chicks-87843.mp3"


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


def classify_mic(y: np.ndarray) -> dict:
    boundary = f"----b{random.randint(100000, 999999)}"
    data = to_wav_bytes(y)
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="mic.wav"\r\n'
        "Content-Type: audio/wav\r\n\r\n"
    ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        API,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def summarize(pred: dict) -> str:
    poultry = False
    if pred.get("windows"):
        poultry = bool(pred["windows"][0].get("poultry_detected"))
    elif pred.get("flock_detected"):
        poultry = True
    return (
        f"{pred['label']:10s} conf={pred['confidence']:.2f} "
        f"poultry={poultry} uncertain={pred['uncertain']}"
    )


def record_window(seconds: float = 3.0) -> np.ndarray:
    frames = int(seconds * SR)
    audio = sd.rec(frames, samplerate=SR, channels=1, dtype="float32")
    sd.wait()
    return audio.flatten()


def main():
    print("=== Device microphone test ===")
    print("Default devices [input, output]:", sd.default.device)
    try:
        inp = sd.query_devices(kind="input")
        print("Input device:", inp["name"])
        print("Max input channels:", inp["max_input_channels"])
    except Exception as e:
        print("Could not query input device:", e)

    # health
    health = json.loads(
        urllib.request.urlopen("http://127.0.0.1:7860/api/health", timeout=10).read().decode()
    )
    print("API:", "ready" if health.get("model_loaded") else "NOT READY")
    print()

    print("Phase A — quiet room (5 x 3s). Stay silent.")
    quiet_ok = 0
    for i in range(5):
        print(f"  recording A{i+1}/5 ...")
        y = record_window(3.0)
        rms = float(np.sqrt(np.mean(y**2)))
        pred = classify_mic(y)
        poultry = bool(pred.get("windows") and pred["windows"][0].get("poultry_detected"))
        ok = pred["label"] == "noise" and not poultry
        quiet_ok += int(ok)
        print(f"    rms={rms:.4f} -> {summarize(pred)}  [{'OK' if ok else 'MISS'}]")
    print(f"  Quiet-room score: {quiet_ok}/5\n")

    print("Phase B — play poultry clip on speakers while mic listens (4 x 3s).")
    print("  Turn speaker volume up a bit so the mic can hear the chicks.")
    farm, _ = librosa.load(FARM, sr=SR, mono=True)
    # loop farm audio under recordings
    stop = threading.Event()

    def play_loop():
        pos = 0
        while not stop.is_set():
            chunk = farm[pos : pos + SR]  # 1s chunks
            if len(chunk) < SR:
                pos = 0
                chunk = farm[:SR]
            else:
                pos += SR
                if pos >= len(farm):
                    pos = 0
            # soft pad to avoid clipping on laptop speakers
            sd.play(chunk * 0.85, SR, blocking=True)

    player = threading.Thread(target=play_loop, daemon=True)
    player.start()
    time.sleep(0.4)

    bird_hits = 0
    for i in range(4):
        print(f"  recording B{i+1}/4 (poultry playing) ...")
        y = record_window(3.0)
        rms = float(np.sqrt(np.mean(y**2)))
        pred = classify_mic(y)
        poultry = bool(pred.get("windows") and pred["windows"][0].get("poultry_detected"))
        # success if poultry detected OR healthy/unhealthy label
        ok = poultry or pred["label"] in ("healthy", "unhealthy")
        bird_hits += int(ok)
        print(f"    rms={rms:.4f} -> {summarize(pred)}  [{'HEARD' if ok else 'WEAK'}]")

    stop.set()
    sd.stop()
    time.sleep(0.3)
    print(f"  Poultry-heard score: {bird_hits}/4\n")

    print("=== MIC TEST RESULT ===")
    print(f"Quiet room ignored background: {quiet_ok}/5")
    print(f"Heard poultry from speakers:   {bird_hits}/4")
    if quiet_ok >= 4 and bird_hits >= 2:
        print("VERDICT: Your microphone path works well.")
    elif quiet_ok >= 4 and bird_hits < 2:
        print(
            "VERDICT: Mic works for silence, but did not pick up played poultry clearly."
        )
        print("         Raise speaker volume or hold mic closer, then retry in the UI.")
    else:
        print("VERDICT: Needs attention — too many false poultry flags in quiet room.")


if __name__ == "__main__":
    main()
