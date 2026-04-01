import os
import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from tqdm import tqdm

from .config import Config


class DataLoader:

    def __init__(self, data_path=None):
        self.data_path = data_path or Config.DATA_RAW
        self.sample_rate = Config.SAMPLE_RATE
        self.duration = Config.DURATION
        self.genres = Config.GENRES

    def scan_dataset(self):
        # on parcourt les dossiers pour trouver les fichiers audio
        files_info = []

        for genre in self.genres:
            genre_path = self.data_path / genre

            if not genre_path.exists():
                print(f"Dossier non trouve: {genre_path}")
                continue

            for file in genre_path.iterdir():
                if file.suffix.lower() in [".wav", ".au", ".mp3"]:
                    files_info.append(
                        {
                            "filepath": str(file),
                            "filename": file.name,
                            "genre": genre,
                            "extension": file.suffix,
                        }
                    )

        df = pd.DataFrame(files_info)
        print(f"{len(df)} fichiers audio trouves.")
        return df

    def load_audio(self, filepath, sr=None, duration=None):
        # charger un fichier audio avec librosa
        sr = sr or self.sample_rate
        duration = duration or self.duration

        try:
            y, sr = librosa.load(filepath, sr=sr, duration=duration)
            return y, sr
        except Exception as e:
            print(f"Erreur chargement {filepath}: {e}")
            return None, sr

    def get_genre_distribution(self, df):
        return df["genre"].value_counts()

    def print_dataset_summary(self, df):
        # afficher un resume du dataset
        print("\n" + "=" * 40)
        print("RESUME DU DATASET")
        print("=" * 40)
        print(f"Nombre total de fichiers: {len(df)}")
        print(f"Nombre de genres: {df['genre'].nunique()}")
        print("\nDistribution par genre:")
        for genre, count in self.get_genre_distribution(df).items():
            print(f"  {genre:12s}: {count}")
        print("=" * 40)


if __name__ == "__main__":
    loader = DataLoader()
    df = loader.scan_dataset()

    if len(df) > 0:
        loader.print_dataset_summary(df)
