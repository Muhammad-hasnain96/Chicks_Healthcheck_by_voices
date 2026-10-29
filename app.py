"""
Web UI + API for the poultry sound classifier.
Upload audio files or stream live microphone chunks from the browser.

Background (fans, rumble, empty rooms) is ignored via poultry-focus gating;
health labels are only emitted when hen/chick vocal energy is detected.
"""

import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import librosa
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from features import SAMPLE_RATE
from poultry_focus import decide_with_poultry_focus, poultry_focus_metrics
from realtime_infer import CONFIDENCE_THRESHOLD, WINDOW_SECONDS, classify_array

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "poultry_classifier.joblib"
STATIC_DIR = ROOT / "static"

bundle = None


def load_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Train first with train_model.py."
        )
    return joblib.load(MODEL_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global bundle
    bundle = load_bundle()
    yield


app = FastAPI(title="Poultry Sound Monitor", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def audio_bytes_to_waveform(data: bytes, filename: str = "audio.wav") -> np.ndarray:
    try:
        y, _ = librosa.load(io.BytesIO(data), sr=SAMPLE_RATE, mono=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not decode audio: {exc}") from exc
    if y.size == 0:
        raise HTTPException(status_code=400, detail="Empty audio")
    return y.astype(np.float32)


def iter_windows(y: np.ndarray, hop_ratio: float = 0.5):
    clip_len = int(WINDOW_SECONDS * SAMPLE_RATE)
    if len(y) <= clip_len:
        padded = np.pad(y, (0, max(0, clip_len - len(y))))
        yield 0.0, padded
        return

    hop = max(1, int(clip_len * hop_ratio))
    starts = list(range(0, len(y) - clip_len + 1, hop))
    if starts[-1] != len(y) - clip_len:
        starts.append(len(y) - clip_len)
    for start in starts:
        yield start / SAMPLE_RATE, y[start : start + clip_len]


def classify_window(chunk: np.ndarray) -> dict:
    _label, _conf, proba = classify_array(chunk, bundle)
    metrics = poultry_focus_metrics(chunk)
    label, confidence, uncertain, poultry = decide_with_poultry_focus(proba, metrics)
    return {
        "label": label,
        "confidence": round(float(confidence), 4),
        "uncertain": bool(uncertain),
        "poultry_detected": bool(poultry),
        "focus_score": round(float(metrics["focus_score"]), 4),
        "focus_ratio": round(float(metrics["focus_ratio"]), 4),
        "distribution": {k: round(float(v), 4) for k, v in proba.items()},
    }


def aggregate_flock(windows: list[dict], rms: float | None = None) -> dict:
    """
    Aggregate windows while ignoring background-only frames.

    Health is decided only from strong poultry windows. If noise dominates
    overall, or bird evidence is weak/minority, stay on noise.
    """
    classes = ["healthy", "unhealthy", "noise"]
    bird_windows = [w for w in windows if w.get("poultry_detected")]
    bg_windows = [w for w in windows if not w.get("poultry_detected")]

    all_mean = {
        c: float(np.mean([w["distribution"][c] for w in windows])) for c in classes
    }

    # Global veto: clip is mostly non-poultry and bird evidence is minority/weak
    if all_mean["noise"] >= 0.55 and len(bird_windows) <= len(bg_windows):
        bird_windows = []
    # Single weak bird window against background -> ignore unless very confident
    if len(bird_windows) == 1 and len(windows) >= 2 and all_mean["noise"] >= 0.50:
        only = bird_windows[0]
        top = max(only["distribution"]["healthy"], only["distribution"]["unhealthy"])
        if top < 0.45 or only["distribution"]["noise"] >= top:
            bird_windows = []
    # Bird windows whose own noise still dominates -> drop them
    strong_birds = []
    for w in bird_windows:
        d = w["distribution"]
        top_bird = max(d["healthy"], d["unhealthy"])
        # Keep if peep focus is high OR top bird clearly beats noise
        if w.get("focus_score", 0) >= 0.90 or (
            top_bird >= d["noise"] and (d["healthy"] + d["unhealthy"]) >= d["noise"]
        ):
            strong_birds.append(w)
    bird_windows = strong_birds

    if bird_windows:
        bird_mean = {
            c: float(np.mean([w["distribution"][c] for w in bird_windows])) for c in classes
        }
        display_dist = all_mean
        label = "healthy" if bird_mean["healthy"] >= bird_mean["unhealthy"] else "unhealthy"
        bird_mass = bird_mean["healthy"] + bird_mean["unhealthy"]
        confidence = float(
            0.5 * bird_mean[label] + 0.5 * (bird_mean[label] / max(bird_mass, 1e-6))
        )
        # Absolute veto only when bird score is tiny AND overall noise dominates hard
        if bird_mean[label] < 0.28 and all_mean["noise"] >= 0.55:
            label = "noise"
            confidence = float(max(all_mean["noise"], 0.55))
            flock_detected = False
            bird_windows = []
            uncertain = False
            focus_mean = float(np.mean([w["focus_score"] for w in windows]))
            mean_dist = display_dist
        else:
            uncertain = confidence < 0.45
            flock_detected = True
            focus_mean = float(np.mean([w["focus_score"] for w in bird_windows]))
            mean_dist = display_dist
            # boost soft peep confidence for UI clarity
            if confidence < 0.50 and focus_mean >= 0.90:
                confidence = min(0.68, confidence + 0.15)
    else:
        mean_dist = all_mean
        label = "noise"
        confidence = float(max(mean_dist["noise"], 0.55))
        uncertain = False
        flock_detected = False
        focus_mean = float(np.mean([w["focus_score"] for w in windows])) if windows else 0.0

    votes = {c: 0 for c in classes}
    for w in windows:
        votes[w["label"]] += 1

    return {
        "label": label,
        "confidence": round(float(confidence), 4),
        "uncertain": bool(uncertain),
        "threshold": CONFIDENCE_THRESHOLD,
        "distribution": {k: round(v, 4) for k, v in mean_dist.items()},
        "window_seconds": WINDOW_SECONDS,
        "flock_detected": flock_detected,
        "poultry_windows": len(bird_windows),
        "background_windows_ignored": len(bg_windows),
        "focus_score": round(focus_mean, 4),
        "votes": votes,
        "windows_analyzed": len(windows),
        "rms": None if rms is None else round(float(rms), 5),
    }


def classify_waveform(y: np.ndarray, full_scan: bool = True) -> dict:
    scan = full_scan
    if len(y) > int(WINDOW_SECONDS * SAMPLE_RATE * 1.05):
        scan = True

    wave = y if scan else y[: int(WINDOW_SECONDS * SAMPLE_RATE)]
    rms = float(np.sqrt(np.mean(np.square(wave.astype(np.float64))))) if wave.size else 0.0

    windows = []
    for t0, chunk in iter_windows(wave):
        w = classify_window(chunk)
        w["t0"] = round(float(t0), 2)
        windows.append(w)

    result = aggregate_flock(windows, rms=rms)
    result["windows"] = windows
    return result


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "model_loaded": bundle is not None,
        "sample_rate": SAMPLE_RATE,
        "window_seconds": WINDOW_SECONDS,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "poultry_focus": True,
        "model_path": str(MODEL_PATH),
        "model_mtime": MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else None,
    }


@app.post("/api/reload-model")
def reload_model():
    global bundle
    bundle = load_bundle()
    return {"ok": True, "model_mtime": MODEL_PATH.stat().st_mtime}


@app.post("/api/classify")
async def classify_upload(file: UploadFile = File(...)):
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload")
    y = audio_bytes_to_waveform(data, file.filename or "upload.wav")
    result = classify_waveform(y, full_scan=True)
    result["filename"] = file.filename
    result["duration_seconds"] = round(len(y) / SAMPLE_RATE, 2)
    return result


@app.post("/api/classify-chunk")
async def classify_chunk(file: UploadFile = File(...)):
    """Accept a short WAV/blob from the browser microphone loop."""
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty chunk")
    y = audio_bytes_to_waveform(data, file.filename or "chunk.wav")
    result = classify_waveform(y, full_scan=False)
    result["source"] = "microphone"
    return result


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run("app:app", host="127.0.0.1", port=port, reload=False)
