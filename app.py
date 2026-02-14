#!/usr/bin/env python3
"""
Application Web - Classification des Genres Musicaux
=====================================================

Lance un serveur web local pour classifier des fichiers audio
via une interface web avec visualisation du pipeline etape par etape.

Usage:
    python app.py

Puis ouvrir http://localhost:5000 dans le navigateur.
"""

import sys
import io
import base64
import numpy as np
import joblib
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask, render_template, request, jsonify
from src.feature_extraction import FeatureExtractor
from src.config import Config

app = Flask(__name__, template_folder="templates", static_folder="static")

UPLOAD_FOLDER = Path(__file__).parent / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}

# Charger le modele au demarrage
MODEL_PATH = Config.MODELS_DIR / "svm.joblib"
save_dict = joblib.load(MODEL_PATH)
model = save_dict["model"]
scaler = save_dict["scaler"]
label_encoder = save_dict["label_encoder"]
extractor = FeatureExtractor()

print(f"Modele charge: {MODEL_PATH}")
print(f"Genres: {list(label_encoder.classes_)}")


def fig_to_base64(fig):
    """Convertit une figure matplotlib en base64 pour l'affichage HTML."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100,
                facecolor='#1a1a2e', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def generate_waveform_image(y, sr):
    """Genere une image du signal audio (waveform)."""
    fig, ax = plt.subplots(figsize=(8, 2.5))
    times = np.linspace(0, len(y) / sr, num=len(y))
    ax.plot(times, y, color='#7c4dff', linewidth=0.4, alpha=0.8)
    ax.fill_between(times, y, alpha=0.15, color='#7c4dff')
    ax.set_xlabel('Temps (s)', color='#aaa', fontsize=9)
    ax.set_ylabel('Amplitude', color='#aaa', fontsize=9)
    ax.set_xlim(0, len(y) / sr)
    ax.set_facecolor('#1a1a2e')
    ax.tick_params(colors='#888', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#333')
    fig.patch.set_facecolor('#1a1a2e')
    fig.tight_layout()
    return fig_to_base64(fig)


def generate_spectrogram_image(y, sr):
    """Genere une image du mel-spectrogramme."""
    fig, ax = plt.subplots(figsize=(8, 2.5))
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, sr=sr, hop_length=512, x_axis='time',
                                    y_axis='mel', ax=ax, cmap='magma')
    fig.colorbar(img, ax=ax, format='%+2.0f dB', pad=0.02)
    ax.set_xlabel('Temps (s)', color='#aaa', fontsize=9)
    ax.set_ylabel('Frequence (Hz)', color='#aaa', fontsize=9)
    ax.set_facecolor('#1a1a2e')
    ax.tick_params(colors='#888', labelsize=8)
    fig.patch.set_facecolor('#1a1a2e')
    fig.tight_layout()
    return fig_to_base64(fig)


def group_features(features):
    """Regroupe les features par categorie pour l'affichage."""
    groups = {
        "MFCC": {},
        "Spectral": {},
        "Chroma": {},
        "Energy & Rhythm": {}
    }
    for name, val in features.items():
        if name.startswith('mfcc'):
            groups["MFCC"][name] = round(float(val), 4)
        elif name.startswith('spectral') or name.startswith('zero_crossing'):
            groups["Spectral"][name] = round(float(val), 4)
        elif name.startswith('chroma'):
            groups["Chroma"][name] = round(float(val), 4)
        else:
            groups["Energy & Rhythm"][name] = round(float(val), 4)
    return groups


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Aucun fichier envoye"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Aucun fichier selectionne"}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"Format non supporte: {ext}"}), 400

    filepath = UPLOAD_FOLDER / file.filename
    file.save(str(filepath))

    try:
        # === STEP 1: Charger le signal audio ===
        y, sr = librosa.load(str(filepath), sr=Config.SAMPLE_RATE, duration=Config.DURATION)
        duration = len(y) / sr

        audio_info = {
            "duration": round(duration, 2),
            "sample_rate": sr,
            "total_samples": len(y)
        }

        # === STEP 2: Generer les visualisations ===
        waveform_img = generate_waveform_image(y, sr)
        spectrogram_img = generate_spectrogram_image(y, sr)

        # === STEP 3: Extraire les features ===
        features = extractor.extract_all_features(y, sr)

        if features is None:
            return jsonify({"error": "Impossible d'analyser ce fichier audio"}), 400

        grouped_features = group_features(features)
        feature_counts = {group: len(feats) for group, feats in grouped_features.items()}

        # === STEP 4: Normalisation ===
        feature_values = np.array(list(features.values())).reshape(1, -1)

        # Garder quelques exemples avant/apres normalisation
        raw_sample = {k: round(float(v), 2) for k, v in list(features.items())[:6]}
        feature_values_scaled = scaler.transform(feature_values)
        scaled_sample = {}
        for i, key in enumerate(list(features.keys())[:6]):
            scaled_sample[key] = round(float(feature_values_scaled[0][i]), 4)

        # === STEP 5: Prediction SVM ===
        prediction = model.predict(feature_values_scaled)
        genre = label_encoder.inverse_transform(prediction)[0]

        probabilities = model.predict_proba(feature_values_scaled)[0]
        genre_probs = []
        for g, prob in zip(label_encoder.classes_, probabilities):
            genre_probs.append({"genre": g, "probability": round(float(prob) * 100, 1)})
        genre_probs.sort(key=lambda x: x["probability"], reverse=True)

        return jsonify({
            "success": True,
            "filename": file.filename,
            "pipeline": {
                "step1_audio": audio_info,
                "step2_waveform": waveform_img,
                "step2_spectrogram": spectrogram_img,
                "step3_features": grouped_features,
                "step3_counts": feature_counts,
                "step3_total": sum(feature_counts.values()),
                "step4_raw": raw_sample,
                "step4_scaled": scaled_sample,
            },
            "genre": genre,
            "confidence": round(float(max(probabilities)) * 100, 1),
            "probabilities": genre_probs
        })

    finally:
        if filepath.exists():
            filepath.unlink()


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  MUSIC GENRE CLASSIFIER - Web App")
    print("=" * 50)
    print("\n  Ouvrir dans le navigateur: http://localhost:5000")
    print("  Ctrl+C pour arreter le serveur\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
