"""
train_model.py

Trains the poultry sound classifier (healthy / unhealthy / noise).

Expected folder layout (point --data_dir at the parent folder):

    data/
        healthy/    *.wav
        unhealthy/  *.wav
        noise/      *.wav

Usage:
    python train_model.py --data_dir ./data --out_dir ./model

Outputs:
    model/poultry_classifier.joblib   - trained model + scaler + label map
    model/confusion_matrix.png
    model/feature_importance.png
    model/training_report.txt
"""

import argparse
import os
import glob
import numpy as np
from collections import Counter

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from features import load_and_chunk, extract_features, feature_names

CLASSES = ["healthy", "unhealthy", "noise"]
AUDIO_EXTS = ("*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a")


def gather_files(data_dir, cls):
    folder = os.path.join(data_dir, cls)
    files = []
    for ext in AUDIO_EXTS:
        files.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(files)


def build_dataset(data_dir):
    X, y = [], []
    for cls in CLASSES:
        files = gather_files(data_dir, cls)
        if not files:
            print(f"  WARNING: no audio files found in {os.path.join(data_dir, cls)}")
        print(f"  {cls}: {len(files)} files")
        for f in files:
            try:
                for chunk in load_and_chunk(f):
                    X.append(extract_features(chunk))
                    y.append(cls)
            except Exception as e:
                print(f"    skipping {f}: {e}")
    return np.array(X), np.array(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./data", help="parent folder containing healthy/unhealthy/noise")
    ap.add_argument("--out_dir", default="./model", help="where to save the trained model + reports")
    ap.add_argument("--n_estimators", type=int, default=300)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading audio and extracting features (this can take a few minutes)...")
    X, y = build_dataset(args.data_dir)
    print(f"\nTotal samples after chunking: {len(X)}")
    print("Class distribution:", Counter(y))

    if len(X) == 0:
        raise SystemExit("No data found - check --data_dir and folder names (healthy/unhealthy/noise).")

    le = LabelEncoder()
    le.fit(CLASSES)
    y_enc = le.transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    class_weights = compute_class_weight("balanced", classes=np.unique(y_enc), y=y_enc)
    class_weight_dict = dict(zip(np.unique(y_enc), class_weights))

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=None,
        min_samples_leaf=2,
        class_weight=class_weight_dict,
        n_jobs=-1,
        random_state=42,
    )

    # Cross-validated predictions give an honest estimate of real-world accuracy
    print(f"\nRunning {args.folds}-fold stratified cross-validation...")
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    y_pred_cv = cross_val_predict(model, X_scaled, y_enc, cv=skf, n_jobs=-1)

    report = classification_report(y_enc, y_pred_cv, target_names=le.classes_, digits=3)
    print("\n=== Cross-validated performance ===")
    print(report)

    cm = confusion_matrix(y_enc, y_pred_cv)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(le.classes_)))
    ax.set_yticks(range(len(le.classes_)))
    ax.set_xticklabels(le.classes_, rotation=45)
    ax.set_yticklabels(le.classes_)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (cross-validated)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    # Fit final model on ALL data for deployment
    print("\nFitting final model on full dataset...")
    model.fit(X_scaled, y_enc)

    # Feature importance
    importances = model.feature_importances_
    names = feature_names()
    order = np.argsort(importances)[::-1][:20]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.barh([names[i] for i in order][::-1], importances[order][::-1])
    ax.set_title("Top 20 Feature Importances")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "feature_importance.png"), dpi=150)
    plt.close(fig)

    bundle = {
        "model": model,
        "scaler": scaler,
        "label_encoder": le,
    }
    model_path = os.path.join(args.out_dir, "poultry_classifier.joblib")
    joblib.dump(bundle, model_path)

    with open(os.path.join(args.out_dir, "training_report.txt"), "w") as f:
        f.write(f"Samples: {len(X)}\n")
        f.write(f"Class distribution: {Counter(y)}\n\n")
        f.write(report)

    print(f"\nSaved model to {model_path}")
    print(f"Saved confusion_matrix.png and feature_importance.png to {args.out_dir}")
    print("\nIf per-class precision/recall look weak for one class (check the report above),")
    print("that's usually a sign to collect more clips for that class, not to tweak the model.")


if __name__ == "__main__":
    main()
