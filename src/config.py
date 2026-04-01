"""
Configuration du Projet
=======================

Ce module contient tous les paramètres de configuration pour le projet
de classification des genres musicaux.
"""

import os
from pathlib import Path


class Config:
    """
    Classe de configuration centralisée pour le projet.

    Attributes:
        PROJECT_ROOT: Chemin racine du projet
        DATA_RAW: Chemin vers les données brutes
        DATA_PROCESSED: Chemin vers les données traitées
        MODELS_DIR: Chemin pour sauvegarder les modèles
        REPORTS_DIR: Chemin pour les rapports
    """

    # ===========================================
    # CHEMINS DU PROJET
    # ===========================================

    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_RAW = PROJECT_ROOT / "data" / "raw"
    DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
    MODELS_DIR = PROJECT_ROOT / "models"
    REPORTS_DIR = PROJECT_ROOT / "reports"

    # ===========================================
    # PARAMÈTRES AUDIO
    # ===========================================

    # Fréquence d'échantillonnage (Hz)
    SAMPLE_RATE = 22050

    # Durée des fichiers audio (secondes)
    DURATION = 30

    # Nombre d'échantillons par fichier
    N_SAMPLES = SAMPLE_RATE * DURATION

    # ===========================================
    # PARAMÈTRES D'EXTRACTION DES FEATURES
    # ===========================================

    # Nombre de coefficients MFCC à extraire
    N_MFCC = 20

    # Taille de la fenêtre FFT
    N_FFT = 2048

    # Pas entre les fenêtres (hop length)
    HOP_LENGTH = 512

    # Nombre de bandes de Mel
    N_MELS = 128

    # Nombre de features chroma
    N_CHROMA = 12

    # ===========================================
    # GENRES MUSICAUX
    # ===========================================

    GENRES = [
        "blues",
        "classical",
        "country",
        "disco",
        "hiphop",
        "jazz",
        "metal",
        "pop",
        "reggae",
        "rock",
    ]

    N_GENRES = len(GENRES)

    # ===========================================
    # PARAMÈTRES D'ENTRAÎNEMENT
    # ===========================================

    # Division des données
    TEST_SIZE = 0.2
    VALIDATION_SIZE = 0.1
    RANDOM_STATE = 42

    # Cross-validation
    N_FOLDS = 5

    # ===========================================
    # PARAMÈTRES DES MODÈLES
    # ===========================================

    # KNN
    KNN_PARAMS = {
        "n_neighbors": [3, 5, 7, 9, 11],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"],
    }

    # SVM
    SVM_PARAMS = {
        "C": [0.1, 1, 10, 100],
        "kernel": ["rbf", "linear", "poly"],
        "gamma": ["scale", "auto"],
    }

    # Random Forest
    RF_PARAMS = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, 30, None],
        "min_samples_split": [2, 5, 10],
    }

    # Gradient Boosting
    GB_PARAMS = {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7],
    }

    # MLP (Neural Network)
    MLP_PARAMS = {
        "hidden_layer_sizes": [(100,), (100, 50), (100, 100)],
        "activation": ["relu", "tanh"],
        "alpha": [0.0001, 0.001, 0.01],
    }

    # ===========================================
    # FICHIERS DE SORTIE
    # ===========================================

    # Nom du fichier CSV des features
    FEATURES_FILE = "features.csv"

    # Nom du fichier des métriques
    METRICS_FILE = "model_metrics.csv"

    # ===========================================
    # MÉTHODES UTILITAIRES
    # ===========================================

    @classmethod
    def create_directories(cls):
        """Crée tous les répertoires nécessaires s'ils n'existent pas."""
        directories = [
            cls.DATA_RAW,
            cls.DATA_PROCESSED,
            cls.MODELS_DIR,
            cls.REPORTS_DIR,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

        print("✅ Tous les répertoires ont été créés.")

    @classmethod
    def get_genre_folders(cls):
        """Retourne la liste des dossiers de genres."""
        return [cls.DATA_RAW / genre for genre in cls.GENRES]

    @classmethod
    def print_config(cls):
        """Affiche la configuration actuelle."""
        print("=" * 50)
        print("CONFIGURATION DU PROJET")
        print("=" * 50)
        print(f"📁 Racine du projet : {cls.PROJECT_ROOT}")
        print(f"📁 Données brutes   : {cls.DATA_RAW}")
        print(f"📁 Données traitées : {cls.DATA_PROCESSED}")
        print(f"📁 Modèles          : {cls.MODELS_DIR}")
        print(f"📁 Rapports         : {cls.REPORTS_DIR}")
        print("-" * 50)
        print(f"🎵 Fréquence d'échantillonnage : {cls.SAMPLE_RATE} Hz")
        print(f"🎵 Durée des fichiers          : {cls.DURATION} s")
        print(f"🎵 Nombre de MFCC              : {cls.N_MFCC}")
        print(f"🎵 Nombre de genres            : {cls.N_GENRES}")
        print("-" * 50)
        print(f"🤖 Test size        : {cls.TEST_SIZE}")
        print(f"🤖 Random state     : {cls.RANDOM_STATE}")
        print(f"🤖 K-Folds          : {cls.N_FOLDS}")
        print("=" * 50)


# Test de la configuration
if __name__ == "__main__":
    Config.print_config()
    Config.create_directories()
