import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from tqdm import tqdm
import warnings

from .config import Config

warnings.filterwarnings("ignore")


class FeatureExtractor:

    def __init__(self):
        self.sample_rate = Config.SAMPLE_RATE
        self.n_mfcc = Config.N_MFCC
        self.n_fft = Config.N_FFT
        self.hop_length = Config.HOP_LENGTH
        self.n_mels = Config.N_MELS
        self.n_chroma = Config.N_CHROMA

    def extract_all_features(self, y, sr):
        # on extrait toutes les features d'un signal audio
        features = {}

        # MFCC (13 coefficients)
        mfcc = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=self.n_mfcc, n_fft=self.n_fft, hop_length=self.hop_length
        )
        for i in range(self.n_mfcc):
            features[f"mfcc_{i+1}_mean"] = np.mean(mfcc[i])
            features[f"mfcc_{i+1}_std"] = np.std(mfcc[i])

        # centroide spectral
        centroid = librosa.feature.spectral_centroid(
            y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
        )[0]
        features["spectral_centroid_mean"] = np.mean(centroid)
        features["spectral_centroid_std"] = np.std(centroid)

        # largeur de bande spectrale
        bandwidth = librosa.feature.spectral_bandwidth(
            y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
        )[0]
        features["spectral_bandwidth_mean"] = np.mean(bandwidth)
        features["spectral_bandwidth_std"] = np.std(bandwidth)

        # rolloff spectral
        rolloff = librosa.feature.spectral_rolloff(
            y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
        )[0]
        features["spectral_rolloff_mean"] = np.mean(rolloff)
        features["spectral_rolloff_std"] = np.std(rolloff)

        # contraste spectral
        contrast = librosa.feature.spectral_contrast(
            y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
        )
        for i in range(contrast.shape[0]):
            features[f"spectral_contrast_{i+1}_mean"] = np.mean(contrast[i])
            features[f"spectral_contrast_{i+1}_std"] = np.std(contrast[i])

        # zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(
            y, frame_length=self.n_fft, hop_length=self.hop_length
        )[0]
        features["zero_crossing_rate_mean"] = np.mean(zcr)
        features["zero_crossing_rate_std"] = np.std(zcr)

        # energie RMS
        rms = librosa.feature.rms(
            y=y, frame_length=self.n_fft, hop_length=self.hop_length
        )[0]
        features["rms_mean"] = np.mean(rms)
        features["rms_std"] = np.std(rms)

        # tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        if isinstance(tempo, np.ndarray):
            tempo = float(tempo[0]) if len(tempo) > 0 else 0.0
        features["tempo"] = float(tempo)

        # chroma features (12 notes)
        chroma = librosa.feature.chroma_stft(
            y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
        )
        notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        for i, note in enumerate(notes):
            features[f"chroma_{note}_mean"] = np.mean(chroma[i])
            features[f"chroma_{note}_std"] = np.std(chroma[i])

        return features

    def extract_features_from_file(self, filepath):
        # extraire les features d'un fichier audio
        try:
            y, sr = librosa.load(
                filepath, sr=self.sample_rate, duration=Config.DURATION
            )
            return self.extract_all_features(y, sr)
        except Exception as e:
            print(f"Erreur pour {filepath}: {e}")
            return None

    def extract_features_from_dataset(self, df, save_path=None, show_progress=True):
        # extraire les features de tous les fichiers du dataset
        all_features = []

        if show_progress:
            iterator = tqdm(
                df.iterrows(), total=len(df), desc="Extraction des features"
            )
        else:
            iterator = df.iterrows()

        for idx, row in iterator:
            features = self.extract_features_from_file(row["filepath"])
            if features is not None:
                features["filename"] = row["filename"]
                features["genre"] = row["genre"]
                all_features.append(features)

        # creer le dataframe
        features_df = pd.DataFrame(all_features)

        # mettre filename et genre en premier
        cols = ["filename", "genre"] + [
            c for c in features_df.columns if c not in ["filename", "genre"]
        ]
        features_df = features_df[cols]

        print(f"{len(features_df)} fichiers traites.")
        print(f"Nombre de features: {len(features_df.columns) - 2}")

        # sauvegarder si besoin
        if save_path:
            features_df.to_csv(save_path, index=False)
            print(f"Features sauvegardees dans: {save_path}")

        return features_df


if __name__ == "__main__":
    extractor = FeatureExtractor()
    print("Module d'extraction charge.")
