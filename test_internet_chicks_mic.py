"""Test internet chick clips: direct API + system-player -> microphone capture."""

from __future__ import annotations

import io
import json
import random
import subprocess
import time
import wave
from pathlib import Path

import librosa
import numpy as np
import sounddevice as sd
import urllib.request

SR = 16000
WIN = 3 * SR
CLIPS_DIR = Path(__file__).resolve().parent / "test_clips" / "chicks"
EXISTING = Path(__file__).resolve().parent / "freesound_community-rooster-and-chicks-87843.mp3"


def api_base() -> str:
    for port in (7860, 7861):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=2)
            return f"http://127.0.0.1:{port}"
        except Exception:
            continue
    raise SystemExit("Server not running on 7860/7861 — start with: python app.py")


def post(url: str, filename: str, data: bytes, content_type: str) -> dict:
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


def to_wav_bytes(y: np.ndarray) -> bytes:
    y = np.nan_to_num(np.asarray(y, dtype=np.float32), nan=0.0)
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


def poultry_flag(pred: dict) -> bool:
    if pred.get("windows"):
        return bool(pred["windows"][0].get("poultry_detected"))
    return bool(pred.get("flock_detected") or (pred.get("poultry_windows") or 0) > 0)


def main():
    base = api_base()
    print("API:", base)
    print("Mic:", sd.query_devices(kind="input")["name"])

    clips = sorted(CLIPS_DIR.glob("*.wav"))
    if not clips:
        clips = sorted(CLIPS_DIR.glob("*.mp3"))
    if EXISTING.exists():
        clips.append(EXISTING)
    print("Clips:", [c.name for c in clips])

    print("\n1) Direct classify (as if clean capture)")
    direct_hits = 0
    for path in clips:
        ctype = "audio/wav" if path.suffix.lower() == ".wav" else "audio/mpeg"
        pred = post(f"{base}/api/classify", path.name, path.read_bytes(), ctype)
        ok = pred["label"] in ("healthy", "unhealthy") and (
            pred.get("flock_detected") or (pred.get("poultry_windows") or 0) > 0
        )
        direct_hits += int(ok)
        print(
            f"  {path.name[:34]:34s} -> {pred['label']:10s} "
            f"conf={pred['confidence']:.2f} poultry_win={pred.get('poultry_windows')} "
            f"[{'OK' if ok else 'MISS'}]"
        )
    print(f"  Direct: {direct_hits}/{len(clips)}")

    print("\n2) Mic-path classify-chunk (file injected as mic WAV)")
    chunk_hits = 0
    chunk_n = 0
    for path in clips:
        y, _ = librosa.load(path, sr=SR, mono=True)
        # take up to 3 windows from the clip
        for i, start in enumerate(range(0, min(len(y), WIN * 3), WIN)):
            chunk = y[start : start + WIN]
            pred = post(
                f"{base}/api/classify-chunk",
                "chunk.wav",
                to_wav_bytes(chunk),
                "audio/wav",
            )
            poultry = poultry_flag(pred)
            ok = poultry and pred["label"] in ("healthy", "unhealthy")
            chunk_hits += int(ok)
            chunk_n += 1
            print(
                f"  {path.name[:28]:28s} w{i} -> {pred['label']:10s} "
                f"conf={pred['confidence']:.2f} poultry={poultry} [{'OK' if ok else 'MISS'}]"
            )
    print(f"  Mic-path inject: {chunk_hits}/{chunk_n}")

    print("\n3) Real microphone while Windows plays chick audio")
    print("   (Raise speaker volume. Recording only — no PortAudio playback.)")
    quiet = sd.rec(WIN, samplerate=SR, channels=1, dtype="float32")
    sd.wait()
    q = post(f"{base}/api/classify-chunk", "q.wav", to_wav_bytes(quiet.flatten()), "audio/wav")
    print(f"  quiet -> {q['label']} poultry={poultry_flag(q)}")

    mic_hits = 0
    mic_n = 0
    for path in clips[:3]:  # the 3 downloaded CC0 chicks
        # Play with default Windows association (wmplayer / Photos audio / etc.)
        print(f"  playing {path.name} via system ...")
        try:
            proc = subprocess.Popen(
                ["cmd", "/c", "start", "/min", "", str(path.resolve())],
                cwd=str(path.parent),
            )
        except Exception as e:
            print("  play failed:", e)
            continue
        time.sleep(1.2)
        for i in range(2):
            rec = sd.rec(WIN, samplerate=SR, channels=1, dtype="float32")
            sd.wait()
            y = np.nan_to_num(rec.flatten(), nan=0.0)
            rms = float(np.sqrt(np.mean(np.square(y.astype(np.float64)))))
            pred = post(
                f"{base}/api/classify-chunk",
                "mic.wav",
                to_wav_bytes(y),
                "audio/wav",
            )
            poultry = poultry_flag(pred)
            ok = poultry or pred["label"] in ("healthy", "unhealthy")
            mic_hits += int(ok)
            mic_n += 1
            print(
                f"    mic {path.stem}#{i+1} rms={rms:.4f} -> {pred['label']:10s} "
                f"conf={pred['confidence']:.2f} poultry={poultry} [{'HEARD' if ok else 'WEAK'}]"
            )
        # try stop player windows
        subprocess.run(
            ["taskkill", "/IM", "Microsoft.Media.Player.exe", "/F"],
            capture_output=True,
        )
        subprocess.run(["taskkill", "/IM", "wmplayer.exe", "/F"], capture_output=True)
        time.sleep(0.4)

    print("\n=== SUMMARY ===")
    print(f"Direct file classify:     {direct_hits}/{len(clips)}")
    print(f"Mic-API inject (no air):  {chunk_hits}/{chunk_n}")
    print(f"Real mic + speakers:      {mic_hits}/{mic_n if mic_n else 0}")
    if direct_hits >= len(clips) - 1 and chunk_hits >= chunk_n * 0.5:
        print("VERDICT: Chick clips work on the model/mic-path.")
        if mic_n and mic_hits < max(1, mic_n // 3):
            print("NOTE: Laptop speakers->mic pickup was weak; use louder playback in the UI test.")
    else:
        print("VERDICT: Needs attention on chick detection.")


if __name__ == "__main__":
    main()
