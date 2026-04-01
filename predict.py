import sys
import argparse
import numpy as np
import joblib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.feature_extraction import FeatureExtractor
from src.config import Config


def predict_genre(audio_path, model_path=None):
    audio_path = Path(audio_path)

    # verifier que le fichier existe
    if not audio_path.exists():
        print(f"Erreur: Le fichier '{audio_path}' n'existe pas.")
        sys.exit(1)

    # chemin du modele
    if model_path is None:
        model_path = Config.MODELS_DIR / "svm.joblib"
    else:
        model_path = Path(model_path)

    if not model_path.exists():
        print(f"Erreur: Le modele '{model_path}' n'existe pas.")
        print("Executez d'abord: python main.py")
        sys.exit(1)

    # charger le modele
    print(f"\n{'='*50}")
    print("  PREDICTION DU GENRE MUSICAL")
    print(f"{'='*50}")
    print(f"\nFichier : {audio_path.name}")
    print(f"Modele  : {model_path.name}")

    save_dict = joblib.load(model_path)
    model = save_dict["model"]
    scaler = save_dict["scaler"]
    label_encoder = save_dict["label_encoder"]

    # extraire les features
    print("\nExtraction des caracteristiques audio...")
    extractor = FeatureExtractor()
    features = extractor.extract_features_from_file(str(audio_path))

    if features is None:
        print("Erreur: Impossible d'extraire les caracteristiques.")
        sys.exit(1)

    # preparer et predire
    feature_values = np.array(list(features.values())).reshape(1, -1)
    feature_values_scaled = scaler.transform(feature_values)

    prediction = model.predict(feature_values_scaled)
    genre = label_encoder.inverse_transform(prediction)[0]

    # probabilites
    probabilities = model.predict_proba(feature_values_scaled)[0]
    genre_probs = list(zip(label_encoder.classes_, probabilities))
    genre_probs.sort(key=lambda x: x[1], reverse=True)

    # afficher les resultats
    print(f"\n{'='*50}")
    print(f"  RESULTAT: {genre.upper()}")
    print(f"{'='*50}")
    print(f"\n  Genre predit : {genre}")
    print(f"  Confiance    : {max(probabilities)*100:.1f}%")
    print(f"\n  Probabilites par genre:")
    print(f"  {'-'*40}")

    for g, prob in genre_probs:
        marker = " <--" if g == genre else ""
        print(f"  {g:12s}: {prob*100:5.1f}%{marker}")

    print(f"{'='*50}\n")

    return genre, genre_probs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predire le genre musical d'un fichier audio")
    parser.add_argument("audio", type=str, help="Chemin vers le fichier audio")
    parser.add_argument("--model", type=str, default=None, help="Chemin vers le modele")

    args = parser.parse_args()
    predict_genre(args.audio, args.model)
