#!/usr/bin/env python3
"""
Prédiction du Genre Musical
============================

Ce script permet de classifier un fichier audio en prédisant son genre musical
en utilisant le modèle entraîné (SVM).

Usage:
    python predict.py "chemin/vers/votre/musique.wav"
    python predict.py "chemin/vers/votre/musique.mp3"
    python predict.py "chemin/vers/votre/musique.wav" --model models/svm.joblib
"""

import sys
import argparse
import numpy as np
import joblib
from pathlib import Path

# Ajouter le répertoire src au path
sys.path.insert(0, str(Path(__file__).parent))

from src.feature_extraction import FeatureExtractor
from src.config import Config


def predict_genre(audio_path: str, model_path: str = None):
    """
    Predit le genre musical d'un fichier audio.

    Args:
        audio_path: Chemin vers le fichier audio (.wav, .mp3, etc.)
        model_path: Chemin vers le modele sauvegarde (defaut: models/svm.joblib)
    """
    audio_path = Path(audio_path)

    # Verifier que le fichier existe
    if not audio_path.exists():
        print(f"Erreur: Le fichier '{audio_path}' n'existe pas.")
        sys.exit(1)

    # Chemin du modele
    if model_path is None:
        model_path = Config.MODELS_DIR / "svm.joblib"
    else:
        model_path = Path(model_path)

    if not model_path.exists():
        print(f"Erreur: Le modele '{model_path}' n'existe pas.")
        print("Executez d'abord: python main.py --step train")
        sys.exit(1)

    # 1. Charger le modele
    print(f"\n{'='*55}")
    print("   PREDICTION DU GENRE MUSICAL")
    print(f"{'='*55}")
    print(f"\nFichier : {audio_path.name}")
    print(f"Modele  : {model_path.name}")

    save_dict = joblib.load(model_path)
    model = save_dict['model']
    scaler = save_dict['scaler']
    label_encoder = save_dict['label_encoder']

    # 2. Extraire les features du fichier audio
    print("\nExtraction des caracteristiques audio...")
    extractor = FeatureExtractor()
    features = extractor.extract_features_from_file(str(audio_path))

    if features is None:
        print("Erreur: Impossible d'extraire les caracteristiques audio.")
        print("Verifiez que le fichier est un format audio valide (.wav, .mp3, .ogg, .flac)")
        sys.exit(1)

    # 3. Preparer les features pour la prediction
    feature_values = np.array(list(features.values())).reshape(1, -1)
    feature_values_scaled = scaler.transform(feature_values)

    # 4. Faire la prediction
    prediction = model.predict(feature_values_scaled)
    genre = label_encoder.inverse_transform(prediction)[0]

    # 5. Obtenir les probabilites pour chaque genre
    probabilities = model.predict_proba(feature_values_scaled)[0]
    genre_probs = list(zip(label_encoder.classes_, probabilities))
    genre_probs.sort(key=lambda x: x[1], reverse=True)

    # 6. Afficher les resultats
    print(f"\n{'='*55}")
    print(f"   RESULTAT: {genre.upper()}")
    print(f"{'='*55}")
    print(f"\n   Genre predit : {genre}")
    print(f"   Confiance    : {max(probabilities)*100:.1f}%")
    print(f"\n{'-'*55}")
    print("   Probabilites par genre:")
    print(f"{'-'*55}")

    for g, prob in genre_probs:
        bar_len = int(prob * 30)
        bar = "#" * bar_len + "." * (30 - bar_len)
        marker = " <--" if g == genre else ""
        print(f"   {g:12s} [{bar}] {prob*100:5.1f}%{marker}")

    print(f"{'='*55}\n")

    return genre, genre_probs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predire le genre musical d'un fichier audio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
    python predict.py ma_chanson.wav
    python predict.py "C:/Music/song.mp3"
    python predict.py song.wav --model models/svm.joblib

Formats supportes: .wav, .mp3, .ogg, .flac
        """
    )

    parser.add_argument(
        "audio",
        type=str,
        help="Chemin vers le fichier audio a classifier"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Chemin vers le modele (defaut: models/svm.joblib)"
    )

    args = parser.parse_args()
    predict_genre(args.audio, args.model)