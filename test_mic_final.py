"""Quick end-to-end mic + farm checks."""
import json
import random
import io
import wave
from pathlib import Path

import librosa
import numpy as np
import urllib.request

SR = 16000
WIN = 3 * SR
BASE = Path(__file__).resolve().parent


def wav_bytes(y):
    y = np.asarray(y, dtype=np.float32)
    if len(y) < WIN:
        y = np.pad(y, (0, WIN - len(y)))
    else:
        y = y[:WIN]
    pcm = (np.clip(y, -1, 1) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def post(url, filename, data, content_type):
    boundary = f"----b{random.randint(100000, 999999)}"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read().decode())


def main():
    # mic-path sample accuracy
    print("Mic-path sample check (4 per class):")
    total = ok = 0
    for cls in ["healthy", "unhealthy", "noise"]:
        files = list((BASE / "data" / cls).glob("*.wav"))
        random.seed(7)
        for f in random.sample(files, 4):
            y, _ = librosa.load(f, sr=SR, mono=True)
            pred = post("http://127.0.0.1:7860/api/classify-chunk", "c.wav", wav_bytes(y), "audio/wav")
            hit = pred["label"] == cls
            total += 1
            ok += int(hit)
            print(f"  {cls:10s} -> {pred['label']:10s} {pred['confidence']:.2f} {'OK' if hit else 'MISS'}")
    print(f"Accuracy: {ok}/{total} = {100*ok/total:.1f}%\n")

    # full farm upload
    farm = BASE / "freesound_community-rooster-and-chicks-87843.mp3"
    pred = post("http://127.0.0.1:7860/api/classify", farm.name, farm.read_bytes(), "audio/mpeg")
    print(
        "Farm upload:",
        pred["label"],
        f"conf={pred['confidence']:.2f}",
        f"flock={pred.get('flock_detected')}",
        f"windows={pred.get('windows_analyzed')}",
    )

    # live mic
    import sounddevice as sd

    print("\nLive mic (2 windows):")
    for i in range(2):
        audio = sd.rec(WIN, samplerate=SR, channels=1, dtype="float32")
        sd.wait()
        y = audio.flatten()
        pred = post("http://127.0.0.1:7860/api/classify-chunk", "m.wav", wav_bytes(y), "audio/wav")
        print(
            f"  #{i+1} rms={float(np.sqrt(np.mean(y**2))):.4f} -> "
            f"{pred['label']} conf={pred['confidence']:.2f} "
            f"uncertain={pred['uncertain']} flock={pred.get('flock_detected')}"
        )


if __name__ == "__main__":
    main()
