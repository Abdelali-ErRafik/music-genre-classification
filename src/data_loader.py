"""
Module de Chargement des Données
================================

Ce module gère le chargement des fichiers audio et la préparation
des données pour l'extraction des caractéristiques.
"""

import os
import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Optional, Dict

from .config import Config


class DataLoader:
    """
    Classe pour charger et gérer les données audio.
    
    Cette classe fournit des méthodes pour:
    - Scanner les fichiers audio dans le dataset
    - Charger les fichiers audio avec librosa
    - Vérifier l'intégrité des données
    - Préparer les données pour l'entraînement
    
    Attributes:
        data_path: Chemin vers les données brutes
        sample_rate: Fréquence d'échantillonnage
        duration: Durée des fichiers audio
    """
    
    def __init__(self, data_path: Optional[Path] = None):
        """
        Initialise le DataLoader.
        
        Args:
            data_path: Chemin vers les données. Si None, utilise Config.DATA_RAW
        """
        self.data_path = data_path or Config.DATA_RAW
        self.sample_rate = Config.SAMPLE_RATE
        self.duration = Config.DURATION
        self.genres = Config.GENRES
        
    def scan_dataset(self) -> pd.DataFrame:
        """
        Scanne le dataset et retourne un DataFrame avec les informations des fichiers.
        
        Returns:
            DataFrame avec colonnes: filepath, filename, genre, extension
        """
        files_info = []
        
        for genre in self.genres:
            genre_path = self.data_path / genre
            
            if not genre_path.exists():
                print(f"⚠️  Dossier non trouvé: {genre_path}")
                continue
                
            for file in genre_path.iterdir():
                if file.suffix.lower() in ['.wav', '.au', '.mp3']:
                    files_info.append({
                        'filepath': str(file),
                        'filename': file.name,
                        'genre': genre,
                        'extension': file.suffix
                    })
        
        df = pd.DataFrame(files_info)
        print(f"✅ {len(df)} fichiers audio trouvés.")
        
        return df
    
    def load_audio(self, filepath: str, 
                   sr: Optional[int] = None,
                   duration: Optional[float] = None) -> Tuple[np.ndarray, int]:
        """
        Charge un fichier audio.
        
        Args:
            filepath: Chemin vers le fichier audio
            sr: Fréquence d'échantillonnage (optionnel)
            duration: Durée à charger en secondes (optionnel)
            
        Returns:
            Tuple (signal audio, fréquence d'échantillonnage)
        """
        sr = sr or self.sample_rate
        duration = duration or self.duration
        
        try:
            # Charger l'audio avec librosa
            y, sr = librosa.load(filepath, sr=sr, duration=duration)
            return y, sr
        except Exception as e:
            print(f"❌ Erreur lors du chargement de {filepath}: {e}")
            return None, sr
    
    def load_all_audio(self, df: pd.DataFrame, 
                       show_progress: bool = True) -> List[Dict]:
        """
        Charge tous les fichiers audio du dataset.
        
        Args:
            df: DataFrame avec les informations des fichiers
            show_progress: Afficher une barre de progression
            
        Returns:
            Liste de dictionnaires avec les données audio
        """
        audio_data = []
        iterator = tqdm(df.iterrows(), total=len(df)) if show_progress else df.iterrows()
        
        for idx, row in iterator:
            y, sr = self.load_audio(row['filepath'])
            
            if y is not None:
                audio_data.append({
                    'filepath': row['filepath'],
                    'filename': row['filename'],
                    'genre': row['genre'],
                    'signal': y,
                    'sample_rate': sr,
                    'duration': len(y) / sr
                })
        
        print(f"✅ {len(audio_data)} fichiers audio chargés avec succès.")
        return audio_data
    
    def check_dataset_integrity(self, df: pd.DataFrame) -> Dict:
        """
        Vérifie l'intégrité du dataset.
        
        Args:
            df: DataFrame avec les informations des fichiers
            
        Returns:
            Dictionnaire avec les statistiques du dataset
        """
        stats = {
            'total_files': len(df),
            'genres_count': df['genre'].value_counts().to_dict(),
            'extensions': df['extension'].value_counts().to_dict(),
            'corrupted_files': [],
            'is_balanced': False
        }
        
        # Vérifier si les classes sont équilibrées
        genre_counts = df['genre'].value_counts()
        stats['is_balanced'] = genre_counts.std() < 5
        
        # Vérifier les fichiers corrompus
        print("🔍 Vérification de l'intégrité des fichiers...")
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            try:
                y, sr = librosa.load(row['filepath'], duration=1)
                if y is None or len(y) == 0:
                    stats['corrupted_files'].append(row['filepath'])
            except Exception as e:
                stats['corrupted_files'].append(row['filepath'])
        
        print(f"✅ Vérification terminée.")
        print(f"   - Fichiers corrompus: {len(stats['corrupted_files'])}")
        
        return stats
    
    def get_genre_distribution(self, df: pd.DataFrame) -> pd.Series:
        """
        Retourne la distribution des genres.
        
        Args:
            df: DataFrame avec les informations des fichiers
            
        Returns:
            Series avec le nombre de fichiers par genre
        """
        return df['genre'].value_counts()
    
    def split_dataset(self, df: pd.DataFrame, 
                      test_size: float = 0.2,
                      stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Divise le dataset en ensembles d'entraînement et de test.
        
        Args:
            df: DataFrame complet
            test_size: Proportion pour le test
            stratify: Stratifier par genre
            
        Returns:
            Tuple (train_df, test_df)
        """
        from sklearn.model_selection import train_test_split
        
        stratify_col = df['genre'] if stratify else None
        
        train_df, test_df = train_test_split(
            df, 
            test_size=test_size,
            stratify=stratify_col,
            random_state=Config.RANDOM_STATE
        )
        
        print(f"✅ Dataset divisé:")
        print(f"   - Entraînement: {len(train_df)} fichiers")
        print(f"   - Test: {len(test_df)} fichiers")
        
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
    
    def print_dataset_summary(self, df: pd.DataFrame):
        """
        Affiche un résumé du dataset.
        
        Args:
            df: DataFrame avec les informations des fichiers
        """
        print("\n" + "=" * 50)
        print("RÉSUMÉ DU DATASET")
        print("=" * 50)
        print(f"📁 Nombre total de fichiers: {len(df)}")
        print(f"🎵 Nombre de genres: {df['genre'].nunique()}")
        print("\n📊 Distribution par genre:")
        print("-" * 30)
        for genre, count in self.get_genre_distribution(df).items():
            bar = "█" * (count // 10)
            print(f"  {genre:12s}: {count:4d} {bar}")
        print("=" * 50)


# Test du module
if __name__ == "__main__":
    loader = DataLoader()
    
    # Scanner le dataset
    df = loader.scan_dataset()
    
    if len(df) > 0:
        # Afficher le résumé
        loader.print_dataset_summary(df)
        
        # Tester le chargement d'un fichier
        if len(df) > 0:
            y, sr = loader.load_audio(df.iloc[0]['filepath'])
            if y is not None:
                print(f"\n✅ Test de chargement réussi:")
                print(f"   - Durée: {len(y)/sr:.2f} secondes")
                print(f"   - Échantillons: {len(y)}")
