#!/usr/bin/env python3
"""
Application Web - Classification des Genres Musicaux
=====================================================

Lance un serveur web local pour classifier des fichiers audio
via une interface web avec upload de fichiers.

Usage:
    python app.py

Puis ouvrir http://localhost:5000 dans le navigateur.
"""

import sys
import os
import numpy as np
import joblib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask, render_template, request, jsonify
from src.feature_extraction import FeatureExtractor
from src.config import Config

app = Flask(__name__, template_folder="templates", static_folder="static")

# Dossier temporaire pour les uploads
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

    # Verifier l'extension
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"Format non supporte: {ext}. Utilisez: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    # Sauvegarder le fichier temporairement
    filepath = UPLOAD_FOLDER / file.filename
    file.save(str(filepath))

    try:
        # Extraire les features
        features = extractor.extract_features_from_file(str(filepath))

        if features is None:
            return jsonify({"error": "Impossible d'analyser ce fichier audio"}), 400

        # Predire
        feature_values = np.array(list(features.values())).reshape(1, -1)
        feature_values_scaled = scaler.transform(feature_values)

        prediction = model.predict(feature_values_scaled)
        genre = label_encoder.inverse_transform(prediction)[0]

        probabilities = model.predict_proba(feature_values_scaled)[0]
        genre_probs = []
        for g, prob in zip(label_encoder.classes_, probabilities):
            genre_probs.append({"genre": g, "probability": round(float(prob) * 100, 1)})

        genre_probs.sort(key=lambda x: x["probability"], reverse=True)

        return jsonify({
            "success": True,
            "genre": genre,
            "confidence": round(float(max(probabilities)) * 100, 1),
            "probabilities": genre_probs,
            "filename": file.filename
        })

    finally:
        # Supprimer le fichier temporaire
        if filepath.exists():
            filepath.unlink()


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  MUSIC GENRE CLASSIFIER - Web App")
    print("=" * 50)
    print("\n  Ouvrir dans le navigateur: http://localhost:5000")
    print("  Ctrl+C pour arreter le serveur\n")
    app.run(debug=True, host="0.0.0.0", port=5000)