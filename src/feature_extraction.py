"""
Module d'Extraction des Caractéristiques Audio
===============================================

Ce module contient toutes les fonctions pour extraire les caractéristiques
acoustiques des fichiers audio (MFCC, spectral features, tempo, etc.)
"""

import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
import warnings

from .config import Config

warnings.filterwarnings("ignore")


class FeatureExtractor:
    """
    Classe pour extraire les caractéristiques audio.

    Cette classe fournit des méthodes pour extraire:
    - MFCC (Mel-Frequency Cepstral Coefficients)
    - Caractéristiques spectrales (centroid, bandwidth, rolloff, contrast)
    - Zero Crossing Rate
    - Tempo et Beat
    - Chroma Features
    - RMS Energy

    Attributes:
        sample_rate: Fréquence d'échantillonnage
        n_mfcc: Nombre de coefficients MFCC
        n_fft: Taille de la fenêtre FFT
        hop_length: Pas entre les fenêtres
    """

    def __init__(self):
        """Initialise le FeatureExtractor avec les paramètres de Config."""
        self.sample_rate = Config.SAMPLE_RATE
        self.n_mfcc = Config.N_MFCC
        self.n_fft = Config.N_FFT
        self.hop_length = Config.HOP_LENGTH
        self.n_mels = Config.N_MELS
        self.n_chroma = Config.N_CHROMA

    def extract_mfcc(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Extrait les coefficients MFCC.

        Les MFCC représentent le spectre de puissance à court terme
        sur l'échelle de Mel, très utilisés pour la reconnaissance audio.

        Args:
            y: Signal audio
            sr: Fréquence d'échantillonnage

        Returns:
            Dictionnaire avec mean et std pour chaque coefficient MFCC
        """
        mfcc = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=self.n_mfcc, n_fft=self.n_fft, hop_length=self.hop_length
        )

        features = {}
        for i in range(self.n_mfcc):
            features[f"mfcc_{i+1}_mean"] = np.mean(mfcc[i])
            features[f"mfcc_{i+1}_std"] = np.std(mfcc[i])

        return features

    def extract_spectral_centroid(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Extrait le centroïde spectral.

        Le centroïde spectral indique le "centre de gravité" du spectre
        et est corrélé avec la "brillance" du son.

        Args:
            y: Signal audio
            sr: Fréquence d'échantillonnage

        Returns:
            Dictionnaire avec mean et std du centroïde spectral
        """
        centroid = librosa.feature.spectral_centroid(
            y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
        )[0]

        return {
            "spectral_centroid_mean": np.mean(centroid),
            "spectral_centroid_std": np.std(centroid),
        }

    def extract_spectral_bandwidth(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Extrait la largeur de bande spectrale.

        Mesure la largeur du spectre autour du centroïde.

        Args:
            y: Signal audio
            sr: Fréquence d'échantillonnage

        Returns:
            Dictionnaire avec mean et std de la largeur de bande
        """
        bandwidth = librosa.feature.spectral_bandwidth(
            y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
        )[0]

        return {
            "spectral_bandwidth_mean": np.mean(bandwidth),
            "spectral_bandwidth_std": np.std(bandwidth),
        }

    def extract_spectral_rolloff(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Extrait le rolloff spectral.

        Fréquence en dessous de laquelle se trouve 85% de l'énergie du spectre.
        Utile pour distinguer les sons harmoniques des sons percussifs.

        Args:
            y: Signal audio
            sr: Fréquence d'échantillonnage

        Returns:
            Dictionnaire avec mean et std du rolloff
        """
        rolloff = librosa.feature.spectral_rolloff(
            y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length, roll_percent=0.85
        )[0]

        return {
            "spectral_rolloff_mean": np.mean(rolloff),
            "spectral_rolloff_std": np.std(rolloff),
        }

    def extract_spectral_contrast(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Extrait le contraste spectral.

        Mesure la différence entre les pics et les vallées du spectre
        dans différentes bandes de fréquence.

        Args:
            y: Signal audio
            sr: Fréquence d'échantillonnage

        Returns:
            Dictionnaire avec mean et std pour chaque bande
        """
        contrast = librosa.feature.spectral_contrast(
            y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
        )

        features = {}
        for i in range(contrast.shape[0]):
            features[f"spectral_contrast_{i+1}_mean"] = np.mean(contrast[i])
            features[f"spectral_contrast_{i+1}_std"] = np.std(contrast[i])

        return features

    def extract_zero_crossing_rate(self, y: np.ndarray) -> Dict[str, float]:
        """
        Extrait le taux de passage par zéro.

        Nombre de fois que le signal change de signe.
        Utile pour distinguer la parole de la musique.

        Args:
            y: Signal audio

        Returns:
            Dictionnaire avec mean et std du ZCR
        """
        zcr = librosa.feature.zero_crossing_rate(
            y, frame_length=self.n_fft, hop_length=self.hop_length
        )[0]

        return {
            "zero_crossing_rate_mean": np.mean(zcr),
            "zero_crossing_rate_std": np.std(zcr),
        }

    def extract_rms_energy(self, y: np.ndarray) -> Dict[str, float]:
        """
        Extrait l'énergie RMS (Root Mean Square).

        Mesure de l'amplitude/volume du signal.

        Args:
            y: Signal audio

        Returns:
            Dictionnaire avec mean et std de l'énergie RMS
        """
        rms = librosa.feature.rms(
            y=y, frame_length=self.n_fft, hop_length=self.hop_length
        )[0]

        return {"rms_mean": np.mean(rms), "rms_std": np.std(rms)}

    def extract_tempo(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Extrait le tempo (BPM).

        Args:
            y: Signal audio
            sr: Fréquence d'échantillonnage

        Returns:
            Dictionnaire avec le tempo estimé
        """
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        # Gérer le cas où tempo est un array
        if isinstance(tempo, np.ndarray):
            tempo = float(tempo[0]) if len(tempo) > 0 else 0.0

        return {"tempo": float(tempo)}

    def extract_chroma_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Extrait les caractéristiques chroma.

        Représente la distribution de l'énergie parmi les 12 classes
        de hauteur (do, do#, ré, etc.). Utile pour l'analyse harmonique.

        Args:
            y: Signal audio
            sr: Fréquence d'échantillonnage

        Returns:
            Dictionnaire avec mean et std pour chaque note
        """
        chroma = librosa.feature.chroma_stft(
            y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
        )

        features = {}
        chroma_labels = [
            "C",
            "C#",
            "D",
            "D#",
            "E",
            "F",
            "F#",
            "G",
            "G#",
            "A",
            "A#",
            "B",
        ]

        for i, label in enumerate(chroma_labels):
            features[f"chroma_{label}_mean"] = np.mean(chroma[i])
            features[f"chroma_{label}_std"] = np.std(chroma[i])

        return features

    def extract_mel_spectrogram_stats(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Extrait les statistiques du mel-spectrogramme.

        Args:
            y: Signal audio
            sr: Fréquence d'échantillonnage

        Returns:
            Dictionnaire avec les statistiques globales
        """
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels
        )

        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        return {
            "mel_spec_mean": np.mean(mel_spec_db),
            "mel_spec_std": np.std(mel_spec_db),
            "mel_spec_max": np.max(mel_spec_db),
            "mel_spec_min": np.min(mel_spec_db),
        }

    def extract_all_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Extrait toutes les caractéristiques audio.

        Args:
            y: Signal audio
            sr: Fréquence d'échantillonnage

        Returns:
            Dictionnaire avec toutes les caractéristiques
        """
        features = {}

        # MFCC
        features.update(self.extract_mfcc(y, sr))

        # Caractéristiques spectrales
        features.update(self.extract_spectral_centroid(y, sr))
        features.update(self.extract_spectral_bandwidth(y, sr))
        features.update(self.extract_spectral_rolloff(y, sr))
        features.update(self.extract_spectral_contrast(y, sr))

        # Zero Crossing Rate
        features.update(self.extract_zero_crossing_rate(y))

        # RMS Energy
        features.update(self.extract_rms_energy(y))

        # Tempo
        features.update(self.extract_tempo(y, sr))

        # Chroma
        features.update(self.extract_chroma_features(y, sr))

        # Mel Spectrogram stats
        features.update(self.extract_mel_spectrogram_stats(y, sr))

        return features

    def extract_features_from_file(self, filepath: str) -> Optional[Dict[str, float]]:
        """
        Extrait les caractéristiques d'un fichier audio.

        Args:
            filepath: Chemin vers le fichier audio

        Returns:
            Dictionnaire avec toutes les caractéristiques, ou None si erreur
        """
        try:
            y, sr = librosa.load(
                filepath, sr=self.sample_rate, duration=Config.DURATION
            )
            return self.extract_all_features(y, sr)
        except Exception as e:
            print(f"❌ Erreur pour {filepath}: {e}")
            return None

    def extract_features_from_dataset(
        self,
        df: pd.DataFrame,
        save_path: Optional[Path] = None,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """
        Extrait les caractéristiques de tous les fichiers du dataset.

        Args:
            df: DataFrame avec les informations des fichiers
            save_path: Chemin pour sauvegarder le CSV (optionnel)
            show_progress: Afficher une barre de progression

        Returns:
            DataFrame avec toutes les caractéristiques
        """
        all_features = []
        iterator = (
            tqdm(df.iterrows(), total=len(df), desc="Extraction des features")
            if show_progress
            else df.iterrows()
        )

        for idx, row in iterator:
            features = self.extract_features_from_file(row["filepath"])

            if features is not None:
                features["filename"] = row["filename"]
                features["genre"] = row["genre"]
                all_features.append(features)

        # Créer le DataFrame
        features_df = pd.DataFrame(all_features)

        # Réorganiser les colonnes (filename et genre en premier)
        cols = ["filename", "genre"] + [
            c for c in features_df.columns if c not in ["filename", "genre"]
        ]
        features_df = features_df[cols]

        print(f"✅ {len(features_df)} fichiers traités avec succès.")
        print(
            f"📊 Nombre de caractéristiques extraites: {len(features_df.columns) - 2}"
        )

        # Sauvegarder si chemin spécifié
        if save_path:
            features_df.to_csv(save_path, index=False)
            print(f"💾 Features sauvegardées dans: {save_path}")

        return features_df

    def get_feature_names(self) -> List[str]:
        """
        Retourne la liste des noms de toutes les caractéristiques.

        Returns:
            Liste des noms de features
        """
        # Créer un signal de test pour obtenir les noms
        y_test = np.random.randn(self.sample_rate)  # 1 seconde de bruit
        features = self.extract_all_features(y_test, self.sample_rate)
        return list(features.keys())

    def print_feature_summary(self):
        """Affiche un résumé des caractéristiques extraites."""
        feature_names = self.get_feature_names()

        print("\n" + "=" * 50)
        print("CARACTÉRISTIQUES AUDIO EXTRAITES")
        print("=" * 50)

        categories = {
            "MFCC": [f for f in feature_names if f.startswith("mfcc")],
            "Spectral": [f for f in feature_names if f.startswith("spectral")],
            "Chroma": [f for f in feature_names if f.startswith("chroma")],
            "Autres": [
                f
                for f in feature_names
                if not any(f.startswith(p) for p in ["mfcc", "spectral", "chroma"])
            ],
        }

        for category, features in categories.items():
            print(f"\n📊 {category}: {len(features)} features")
            if len(features) <= 10:
                for f in features:
                    print(f"   - {f}")
            else:
                for f in features[:5]:
                    print(f"   - {f}")
                print(f"   ... et {len(features) - 5} autres")

        print(f"\n📈 TOTAL: {len(feature_names)} caractéristiques")
        print("=" * 50)


# Test du module
if __name__ == "__main__":
    extractor = FeatureExtractor()
    extractor.print_feature_summary()
