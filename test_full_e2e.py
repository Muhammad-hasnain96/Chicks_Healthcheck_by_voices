"""Full end-to-end verification of FlockTone with poultry-focus gating."""

from __future__ import annotations

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
BASE = Path(__file__).resolve().parent
API = "http://127.0.0.1:7860"


def post_file(url: str, filename: str, data: bytes, content_type: str) -> dict:
    boundary = f"----bound{random.randint(1_000_000, 9_999_999)}"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        return json.loads(resp.read().decode())


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


def main():
    print("1) Server health")
    health = json.loads(urllib.request.urlopen(f"{API}/api/health", timeout=10).read().decode())
    print("  ", health)
    assert health.get("ok") and health.get("model_loaded"), "server/model not ready"

    print("\n2) Mic-path accuracy (12 clips/class via /api/classify-chunk)")
    totals = {}
    for cls in ["healthy", "unhealthy", "noise"]:
        files = list((BASE / "data" / cls).glob("*.wav"))
        random.seed(11)
        sample = random.sample(files, min(12, len(files)))
        correct = poultry_flags = 0
        for f in sample:
            y, _ = librosa.load(f, sr=SR, mono=True)
            # simulate browser resample path
            y48 = librosa.resample(y, orig_sr=SR, target_sr=48000)
            y16 = librosa.resample(y48, orig_sr=48000, target_sr=SR)
            pred = post_file(
                f"{API}/api/classify-chunk",
                "chunk.wav",
                to_wav_bytes(y16),
                "audio/wav",
            )
            poultry = bool(pred.get("flock_detected") or (pred.get("poultry_windows") or 0) > 0)
            # single-window responses expose poultry_detected on windows[0]
            if pred.get("windows"):
                poultry = bool(pred["windows"][0].get("poultry_detected"))
            if cls == "noise":
                ok = pred["label"] == "noise" and not poultry
            else:
                ok = pred["label"] == cls and poultry
            correct += int(ok)
            poultry_flags += int(poultry)
            print(
                f"  [{cls:9s}] {f.name[:36]:36s} -> {pred['label']:10s} "
                f"conf={pred['confidence']:.2f} poultry={poultry} {'OK' if ok else 'MISS'}"
            )
        totals[cls] = (correct, len(sample), poultry_flags)
        print(
            f"  >> {cls}: {correct}/{len(sample)} "
            f"(poultry_flagged={poultry_flags}/{len(sample)})\n"
        )

    print("3) Farm multi-chick upload")
    farm = BASE / "freesound_community-rooster-and-chicks-87843.mp3"
    farm_pred = post_file(
        f"{API}/api/classify", farm.name, farm.read_bytes(), "audio/mpeg"
    )
    print(
        "  ",
        {
            "label": farm_pred["label"],
            "confidence": farm_pred["confidence"],
            "poultry_windows": farm_pred.get("poultry_windows"),
            "background_ignored": farm_pred.get("background_windows_ignored"),
            "flock_detected": farm_pred.get("flock_detected"),
        },
    )
    farm_ok = farm_pred["label"] == "healthy" and farm_pred.get("flock_detected")
    print("  farm_ok=", farm_ok)

    print("\n4) Live microphone (3 quiet-room windows)")
    try:
        import sounddevice as sd

        room_ok = 0
        for i in range(3):
            audio = sd.rec(WIN, samplerate=SR, channels=1, dtype="float32")
            sd.wait()
            y = audio.flatten()
            pred = post_file(
                f"{API}/api/classify-chunk", "mic.wav", to_wav_bytes(y), "audio/wav"
            )
            poultry = False
            if pred.get("windows"):
                poultry = bool(pred["windows"][0].get("poultry_detected"))
            ok = pred["label"] == "noise" and not poultry
            room_ok += int(ok)
            print(
                f"  mic#{i+1} rms={float(np.sqrt(np.mean(y**2))):.4f} -> "
                f"{pred['label']} conf={pred['confidence']:.2f} "
                f"poultry={poultry} uncertain={pred['uncertain']} {'OK' if ok else 'MISS'}"
            )
        print(f"  >> room: {room_ok}/3")
    except Exception as e:
        room_ok = -1
        print("  mic failed:", e)

    print("\n=== SUMMARY ===")
    overall_correct = sum(v[0] for v in totals.values())
    overall_n = sum(v[1] for v in totals.values())
    print(f"Labeled mic-path: {overall_correct}/{overall_n} = {100*overall_correct/overall_n:.1f}%")
    print(f"Farm upload: {'PASS' if farm_ok else 'FAIL'}")
    if room_ok >= 0:
        print(f"Quiet room mic: {room_ok}/3 {'PASS' if room_ok == 3 else 'CHECK'}")
    passed = overall_correct >= int(0.85 * overall_n) and farm_ok and room_ok >= 2
    print("VERDICT:", "WORKS WELL" if passed else "NEEDS ATTENTION")


if __name__ == "__main__":
    main()
