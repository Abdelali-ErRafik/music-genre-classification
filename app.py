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
import pandas as pd
import joblib
import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask, render_template, request, jsonify
from src.feature_extraction import FeatureExtractor
from src.config import Config

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

app = Flask(__name__, template_folder="templates", static_folder="static")

UPLOAD_FOLDER = Path(__file__).parent / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}
AUDIO_MIME = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
}

# === Charger le modele principal ===
MODEL_PATH = Config.MODELS_DIR / "svm.joblib"
save_dict = joblib.load(MODEL_PATH)

model = save_dict["model"]
scaler = save_dict["scaler"]
label_encoder = save_dict["label_encoder"]
extractor = FeatureExtractor()

print(f"Modele principal charge: {MODEL_PATH}")
print(f"Genres: {list(label_encoder.classes_)}")

# === Entrainer les modeles de comparaison au demarrage ===
comparison_models = {}
features_path = Config.DATA_PROCESSED / Config.FEATURES_FILE

if features_path.exists():
    print("Entrainement des modeles de comparaison...")
    df = pd.read_csv(features_path)
    feature_cols = [c for c in df.columns if c not in ["filename", "genre", "filepath"]]
    X_all = df[feature_cols].values
    y_all = label_encoder.transform(df["genre"].values)
    X_all_scaled = scaler.transform(X_all)

    models_to_train = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(kernel="rbf", C=10, gamma="scale", probability=True),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=20, random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=42
        ),
        "MLP (Neural Net)": MLPClassifier(
            hidden_layer_sizes=(100, 50), max_iter=500, random_state=42
        ),
        "Naive Bayes": GaussianNB(),
    }

    for name, m in models_to_train.items():
        m.fit(X_all_scaled, y_all)
        train_acc = m.score(X_all_scaled, y_all)
        comparison_models[name] = {
            "model": m,
            "train_accuracy": round(train_acc * 100, 1),
        }
        print(f"  {name}: {train_acc:.3f}")

    print(f"{len(comparison_models)} modeles prets pour la comparaison.")
else:
    print("features.csv non trouve, comparaison desactivee.")


# === Fonctions de visualisation ===


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        bbox_inches="tight",
        dpi=100,
        facecolor="#1a1a2e",
        edgecolor="none",
    )
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def generate_waveform_image(y, sr):
    fig, ax = plt.subplots(figsize=(8, 2.5))
    times = np.linspace(0, len(y) / sr, num=len(y))
    ax.plot(times, y, color="#7c4dff", linewidth=0.4, alpha=0.8)
    ax.fill_between(times, y, alpha=0.15, color="#7c4dff")
    ax.set_xlabel("Temps (s)", color="#aaa", fontsize=9)
    ax.set_ylabel("Amplitude", color="#aaa", fontsize=9)
    ax.set_xlim(0, len(y) / sr)
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(colors="#888", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#333")
    fig.patch.set_facecolor("#1a1a2e")
    fig.tight_layout()
    return fig_to_base64(fig)


def generate_spectrogram_image(y, sr):
    fig, ax = plt.subplots(figsize=(8, 2.5))
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512
    )
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(
        S_dB, sr=sr, hop_length=512, x_axis="time", y_axis="mel", ax=ax, cmap="magma"
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB", pad=0.02)
    ax.set_xlabel("Temps (s)", color="#aaa", fontsize=9)
    ax.set_ylabel("Frequence (Hz)", color="#aaa", fontsize=9)
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(colors="#888", labelsize=8)
    fig.patch.set_facecolor("#1a1a2e")
    fig.tight_layout()
    return fig_to_base64(fig)


def generate_mfcc_image(y, sr):
    fig, ax = plt.subplots(figsize=(8, 3))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=2048, hop_length=512)
    img = librosa.display.specshow(
        mfcc, sr=sr, hop_length=512, x_axis="time", ax=ax, cmap="coolwarm"
    )
    fig.colorbar(img, ax=ax, pad=0.02)
    ax.set_ylabel("MFCC Coefficient", color="#aaa", fontsize=9)
    ax.set_xlabel("Temps (s)", color="#aaa", fontsize=9)
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(colors="#888", labelsize=8)
    fig.patch.set_facecolor("#1a1a2e")
    fig.tight_layout()
    return fig_to_base64(fig)


def generate_chroma_image(y, sr):
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=2048, hop_length=512)
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    chroma_mean = np.mean(chroma, axis=1)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10, 3), gridspec_kw={"width_ratios": [3, 2]}
    )

    # Chroma heatmap over time
    img = librosa.display.specshow(
        chroma,
        sr=sr,
        hop_length=512,
        x_axis="time",
        y_axis="chroma",
        ax=ax1,
        cmap="magma",
    )
    ax1.set_ylabel("Note", color="#aaa", fontsize=9)
    ax1.set_xlabel("Temps (s)", color="#aaa", fontsize=9)
    ax1.set_facecolor("#1a1a2e")
    ax1.tick_params(colors="#888", labelsize=8)

    # Bar chart of average chroma
    colors = ["#7c4dff" if v == max(chroma_mean) else "#448aff" for v in chroma_mean]
    ax2.barh(notes, chroma_mean, color=colors, height=0.6)
    ax2.set_xlabel("Energie moyenne", color="#aaa", fontsize=9)
    ax2.set_facecolor("#1a1a2e")
    ax2.tick_params(colors="#888", labelsize=8)
    ax2.invert_yaxis()
    for spine in ax2.spines.values():
        spine.set_color("#333")

    fig.patch.set_facecolor("#1a1a2e")
    fig.tight_layout()
    return fig_to_base64(fig)


def generate_model_comparison_image(model_results):
    names = [r["name"] for r in model_results]
    predictions = [r["genre"] for r in model_results]
    confidences = [r["confidence"] for r in model_results]

    fig, ax = plt.subplots(figsize=(8, 3.5))

    # Color bars by prediction
    genre_colors = {
        "blues": "#2196F3",
        "classical": "#9C27B0",
        "country": "#FF9800",
        "disco": "#E91E63",
        "hiphop": "#00BCD4",
        "jazz": "#FFC107",
        "metal": "#F44336",
        "pop": "#4CAF50",
        "reggae": "#8BC34A",
        "rock": "#FF5722",
    }
    colors = [genre_colors.get(g, "#7c4dff") for g in predictions]

    bars = ax.barh(names, confidences, color=colors, height=0.55, alpha=0.85)

    for bar, pred, conf in zip(bars, predictions, confidences):
        ax.text(
            bar.get_width() + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{pred} ({conf}%)",
            va="center",
            color="#ccc",
            fontsize=8.5,
        )

    ax.set_xlabel("Confiance (%)", color="#aaa", fontsize=9)
    ax.set_xlim(0, 110)
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(colors="#888", labelsize=8.5)
    for spine in ax.spines.values():
        spine.set_color("#333")
    ax.invert_yaxis()
    fig.patch.set_facecolor("#1a1a2e")
    fig.tight_layout()
    return fig_to_base64(fig)


def group_features(features):
    groups = {"MFCC": {}, "Spectral": {}, "Chroma": {}, "Energy & Rhythm": {}}
    for name, val in features.items():
        if name.startswith("mfcc"):
            groups["MFCC"][name] = round(float(val), 4)
        elif name.startswith("spectral") or name.startswith("zero_crossing"):
            groups["Spectral"][name] = round(float(val), 4)
        elif name.startswith("chroma"):
            groups["Chroma"][name] = round(float(val), 4)
        else:
            groups["Energy & Rhythm"][name] = round(float(val), 4)
    return groups


def generate_spectral_centroid_image(y, sr):
    fig, ax = plt.subplots(figsize=(8, 2.5))
    centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=2048, hop_length=512
    )[0]
    frames = range(len(centroid))
    t = librosa.frames_to_time(frames, sr=sr, hop_length=512)
    ax.plot(t, centroid, color="#ff9800", linewidth=1.5, alpha=0.9)
    ax.fill_between(t, centroid, alpha=0.1, color="#ff9800")
    ax.set_xlabel("Temps (s)", color="#aaa", fontsize=9)
    ax.set_ylabel("Frequence (Hz)", color="#aaa", fontsize=9)
    ax.set_xlim(0, t[-1])
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(colors="#888", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#333")
    fig.patch.set_facecolor("#1a1a2e")
    fig.tight_layout()
    return fig_to_base64(fig)


def generate_pca_image(X_pca, genres, new_pca, new_genre):
    fig, ax = plt.subplots(figsize=(8, 6))
    genre_colors = {
        "blues": "#2196F3",
        "classical": "#9C27B0",
        "country": "#FF9800",
        "disco": "#E91E63",
        "hiphop": "#00BCD4",
        "jazz": "#FFC107",
        "metal": "#F44336",
        "pop": "#4CAF50",
        "reggae": "#8BC34A",
        "rock": "#FF5722",
    }
    for g in np.unique(genres):
        mask = genres == g
        color = genre_colors.get(g, "#888")
        ax.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            c=color,
            label=g,
            alpha=0.45,
            s=22,
            edgecolors="none",
        )
    ax.scatter(
        new_pca[0, 0],
        new_pca[0, 1],
        c="#fff",
        marker="*",
        s=350,
        edgecolors="#ffe000",
        linewidths=1.5,
        zorder=10,
        label=f"Votre chanson ({new_genre})",
    )
    ax.legend(
        loc="best",
        fontsize=7,
        facecolor="#1a1a2e",
        edgecolor="#333",
        labelcolor="#ccc",
        ncol=2,
    )
    ax.set_xlabel("Composante Principale 1", color="#aaa", fontsize=9)
    ax.set_ylabel("Composante Principale 2", color="#aaa", fontsize=9)
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(colors="#888", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#333")
    fig.patch.set_facecolor("#1a1a2e")
    fig.tight_layout()
    return fig_to_base64(fig)


def generate_confusion_matrix_image(cm, classes):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(cm, interpolation="nearest", cmap="magma", aspect="auto")
    fig.colorbar(im, ax=ax, pad=0.02, shrink=0.8)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8, color="#ccc")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=8, color="#ccc")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "#aaa",
                fontsize=7,
            )
    ax.set_ylabel("Genre reel", color="#aaa", fontsize=9)
    ax.set_xlabel("Genre predit", color="#aaa", fontsize=9)
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")
    fig.tight_layout()
    return fig_to_base64(fig)


def generate_feature_importance_image(importances, top_n=15):
    names = [x[0] for x in importances[:top_n]][::-1]
    values = [x[1] for x in importances[:top_n]][::-1]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    max_val = max(values)
    colors = ["#7c4dff" if v == max_val else "#448aff" for v in values]
    ax.barh(names, values, color=colors, height=0.6, alpha=0.85)
    ax.set_xlabel("Importance (Random Forest)", color="#aaa", fontsize=9)
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(colors="#888", labelsize=7.5)
    for spine in ax.spines.values():
        spine.set_color("#333")
    fig.patch.set_facecolor("#1a1a2e")
    fig.tight_layout()
    return fig_to_base64(fig)


# === Calculs avances au demarrage ===
pca_2d = None
pca_X = None
pca_genres = None
confusion_img_static = None
importance_img_static = None

if features_path.exists() and comparison_models:
    print("Calcul des visualisations avancees...")

    # PCA 2D
    pca_2d = PCA(n_components=2)
    pca_X = pca_2d.fit_transform(X_all_scaled)
    pca_genres = df["genre"].values

    # Feature importance (Random Forest)
    rf_importances = sorted(
        zip(
            feature_cols,
            comparison_models["Random Forest"]["model"].feature_importances_,
        ),
        key=lambda x: x[1],
        reverse=True,
    )
    importance_img_static = generate_feature_importance_image(rf_importances)

    # Confusion matrix (cross-validation 5-fold)
    print("  Cross-validation pour la matrice de confusion...")
    cv_preds = cross_val_predict(
        SVC(kernel="rbf", C=10, gamma="scale"), X_all_scaled, y_all, cv=5
    )
    conf_matrix = sk_confusion_matrix(y_all, cv_preds)
    confusion_img_static = generate_confusion_matrix_image(
        conf_matrix, label_encoder.classes_
    )

    print("Visualisations avancees pretes.")


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
        # === Audio base64 pour lecture dans le navigateur ===
        with open(str(filepath), "rb") as f:
            audio_bytes = base64.b64encode(f.read()).decode("utf-8")
        audio_mime = AUDIO_MIME.get(ext, "audio/mpeg")

        # === STEP 1: Charger le signal audio ===
        y, sr = librosa.load(
            str(filepath), sr=Config.SAMPLE_RATE, duration=Config.DURATION
        )
        duration = len(y) / sr
        raw_samples = [round(float(s), 6) for s in y[:20]]

        audio_info = {
            "duration": round(duration, 2),
            "sample_rate": sr,
            "total_samples": len(y),
            "raw_samples": raw_samples,
            "min_value": round(float(np.min(y)), 4),
            "max_value": round(float(np.max(y)), 4),
        }

        # === STEP 2: Visualisations ===
        waveform_img = generate_waveform_image(y, sr)
        spectrogram_img = generate_spectrogram_image(y, sr)
        centroid_img = generate_spectral_centroid_image(y, sr)

        # === STEP 3: Features + graphiques MFCC / Chroma ===
        features = extractor.extract_all_features(y, sr)

        if features is None:
            return jsonify({"error": "Impossible d'analyser ce fichier audio"}), 400

        grouped_features = group_features(features)
        feature_counts = {
            group: len(feats) for group, feats in grouped_features.items()
        }

        mfcc_img = generate_mfcc_image(y, sr)
        chroma_img = generate_chroma_image(y, sr)

        # === STEP 4: Normalisation ===
        feature_values = np.array(list(features.values())).reshape(1, -1)
        norm_details = []
        feature_keys = list(features.keys())
        feature_values_scaled = scaler.transform(feature_values)
        for i in range(min(6, len(feature_keys))):
            key = feature_keys[i]
            norm_details.append(
                {
                    "name": key,
                    "raw": round(float(features[key]), 4),
                    "mean": round(float(scaler.mean_[i]), 4),
                    "std": round(float(scaler.scale_[i]), 4),
                    "scaled": round(float(feature_values_scaled[0][i]), 4),
                }
            )

        # === STEP 5: Prediction SVM ===
        prediction = model.predict(feature_values_scaled)
        genre = label_encoder.inverse_transform(prediction)[0]
        probabilities = model.predict_proba(feature_values_scaled)[0]
        genre_probs = []
        for g, prob in zip(label_encoder.classes_, probabilities):
            genre_probs.append({"genre": g, "probability": round(float(prob) * 100, 1)})
        genre_probs.sort(key=lambda x: x["probability"], reverse=True)
        top1 = genre_probs[0]
        top2 = genre_probs[1]

        # PCA plot: project new sample into 2D space
        pca_img = None
        if pca_2d is not None:
            new_pca = pca_2d.transform(feature_values_scaled)
            pca_img = generate_pca_image(pca_X, pca_genres, new_pca, genre)

        # === STEP 6: Comparaison des modeles ===
        model_results = []
        for name, info in comparison_models.items():
            m = info["model"]
            pred = m.predict(feature_values_scaled)
            pred_genre = label_encoder.inverse_transform(pred)[0]

            if hasattr(m, "predict_proba"):
                proba = m.predict_proba(feature_values_scaled)[0]
                conf = round(float(max(proba)) * 100, 1)
            else:
                conf = 0.0

            model_results.append(
                {
                    "name": name,
                    "genre": pred_genre,
                    "confidence": conf,
                    "train_accuracy": info["train_accuracy"],
                }
            )

        model_results.sort(key=lambda x: x["confidence"], reverse=True)

        # Graphique de comparaison
        comparison_img = (
            generate_model_comparison_image(model_results) if model_results else None
        )

        # Consensus
        all_predictions = [r["genre"] for r in model_results]
        from collections import Counter

        vote_counts = Counter(all_predictions)
        consensus_genre = vote_counts.most_common(1)[0][0]
        consensus_count = vote_counts.most_common(1)[0][1]

        return jsonify(
            {
                "success": True,
                "filename": file.filename,
                "audio_base64": audio_bytes,
                "audio_mime": audio_mime,
                "pipeline": {
                    "step1_audio": audio_info,
                    "step2_waveform": waveform_img,
                    "step2_spectrogram": spectrogram_img,
                    "step2_centroid": centroid_img,
                    "step3_features": grouped_features,
                    "step3_counts": feature_counts,
                    "step3_total": sum(feature_counts.values()),
                    "step3_mfcc_img": mfcc_img,
                    "step3_chroma_img": chroma_img,
                    "step3_importance_img": importance_img_static,
                    "step4_norm_details": norm_details,
                    "step5_pca_img": pca_img,
                    "step6_comparison_img": comparison_img,
                    "step6_models": model_results,
                    "step6_consensus": consensus_genre,
                    "step6_consensus_count": consensus_count,
                    "step6_total_models": len(model_results),
                    "step6_confusion_img": confusion_img_static,
                },
                "genre": genre,
                "confidence": round(float(max(probabilities)) * 100, 1),
                "probabilities": genre_probs,
                "top1": top1,
                "top2": top2,
            }
        )

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
