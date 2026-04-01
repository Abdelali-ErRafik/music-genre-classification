import sys
import io
import base64
import numpy as np
import joblib
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask, render_template, request, jsonify
from src.feature_extraction import FeatureExtractor
from src.config import Config

app = Flask(__name__, template_folder="templates")

UPLOAD_FOLDER = Path(__file__).parent / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac"}

# charger le modele SVM sauvegarde
MODEL_PATH = Config.MODELS_DIR / "svm.joblib"
save_dict = joblib.load(MODEL_PATH)
model = save_dict["model"]
scaler = save_dict["scaler"]
label_encoder = save_dict["label_encoder"]
extractor = FeatureExtractor()

print(f"Modele charge: {MODEL_PATH}")
print(f"Genres: {list(label_encoder.classes_)}")


# --- fonctions pour generer les images ---

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def generate_waveform(y, sr):
    # forme d'onde
    fig, ax = plt.subplots(figsize=(10, 3))
    times = np.linspace(0, len(y) / sr, num=len(y))
    ax.plot(times, y, color='steelblue', linewidth=0.5)
    ax.set_xlabel('Temps (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title("Forme d'onde")
    ax.set_xlim(0, len(y) / sr)
    fig.tight_layout()
    return fig_to_base64(fig)


def generate_spectrogram(y, sr):
    # spectrogramme mel
    fig, ax = plt.subplots(figsize=(10, 3))
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax, cmap='magma')
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title('Spectrogramme Mel')
    fig.tight_layout()
    return fig_to_base64(fig)


def generate_mfcc(y, sr):
    # coefficients MFCC
    fig, ax = plt.subplots(figsize=(10, 3))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    img = librosa.display.specshow(mfcc, sr=sr, x_axis='time', ax=ax, cmap='coolwarm')
    fig.colorbar(img, ax=ax)
    ax.set_title('MFCC')
    ax.set_ylabel('Coefficient')
    fig.tight_layout()
    return fig_to_base64(fig)


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
        # charger le signal audio
        y, sr = librosa.load(str(filepath), sr=Config.SAMPLE_RATE, duration=Config.DURATION)

        # generer les visualisations
        waveform_img = generate_waveform(y, sr)
        spectrogram_img = generate_spectrogram(y, sr)
        mfcc_img = generate_mfcc(y, sr)

        # extraire les features
        features = extractor.extract_all_features(y, sr)
        if features is None:
            return jsonify({"error": "Impossible d'analyser ce fichier"}), 400

        # normaliser et predire
        feature_values = np.array(list(features.values())).reshape(1, -1)
        feature_values_scaled = scaler.transform(feature_values)

        prediction = model.predict(feature_values_scaled)
        genre = label_encoder.inverse_transform(prediction)[0]

        # probabilites par genre
        probabilities = model.predict_proba(feature_values_scaled)[0]
        genre_probs = []
        for g, prob in zip(label_encoder.classes_, probabilities):
            genre_probs.append({"genre": g, "probability": round(float(prob) * 100, 1)})
        genre_probs.sort(key=lambda x: x["probability"], reverse=True)

        return jsonify({
            "success": True,
            "filename": file.filename,
            "genre": genre,
            "confidence": round(float(max(probabilities)) * 100, 1),
            "probabilities": genre_probs,
            "duration": round(len(y) / sr, 1),
            "waveform_img": waveform_img,
            "spectrogram_img": spectrogram_img,
            "mfcc_img": mfcc_img,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if filepath.exists():
            filepath.unlink()


if __name__ == "__main__":
    print("\n" + "=" * 45)
    print("  Classification des Genres Musicaux")
    print("=" * 45)
    print("\n  http://localhost:5000")
    print("  Ctrl+C pour arreter\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
