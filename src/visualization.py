"""
Module de Visualisation
=======================

Ce module contient toutes les fonctions pour visualiser les données audio,
les caractéristiques extraites et les résultats des modèles.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from .config import Config

# Configuration du style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class Visualizer:
    """
    Classe pour créer des visualisations audio et des résultats.
    
    Cette classe fournit des méthodes pour:
    - Visualiser les formes d'onde
    - Afficher les spectrogrammes
    - Créer des graphiques de distribution des features
    - Visualiser les résultats des modèles
    
    Attributes:
        figsize: Taille par défaut des figures
        dpi: Résolution des figures
        save_path: Chemin pour sauvegarder les figures
    """
    
    def __init__(self, save_path: Optional[Path] = None):
        """
        Initialise le Visualizer.
        
        Args:
            save_path: Chemin pour sauvegarder les figures (optionnel)
        """
        self.save_path = save_path or Config.REPORTS_DIR
        self.figsize = (12, 6)
        self.dpi = 100
        self.sample_rate = Config.SAMPLE_RATE
        
    def plot_waveform(self, y: np.ndarray, sr: int, 
                      title: str = "Forme d'onde",
                      save_name: Optional[str] = None) -> plt.Figure:
        """
        Affiche la forme d'onde d'un signal audio.
        
        Args:
            y: Signal audio
            sr: Fréquence d'échantillonnage
            title: Titre du graphique
            save_name: Nom du fichier pour sauvegarder
            
        Returns:
            Figure matplotlib
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        librosa.display.waveshow(y, sr=sr, ax=ax, color='steelblue')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Temps (s)")
        ax.set_ylabel("Amplitude")
        
        plt.tight_layout()
        
        if save_name:
            fig.savefig(self.save_path / save_name, dpi=self.dpi, bbox_inches='tight')
            
        return fig
    
    def plot_spectrogram(self, y: np.ndarray, sr: int,
                         title: str = "Spectrogramme",
                         save_name: Optional[str] = None) -> plt.Figure:
        """
        Affiche le spectrogramme d'un signal audio.
        
        Args:
            y: Signal audio
            sr: Fréquence d'échantillonnage
            title: Titre du graphique
            save_name: Nom du fichier pour sauvegarder
            
        Returns:
            Figure matplotlib
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=ax)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        
        plt.tight_layout()
        
        if save_name:
            fig.savefig(self.save_path / save_name, dpi=self.dpi, bbox_inches='tight')
            
        return fig
    
    def plot_mel_spectrogram(self, y: np.ndarray, sr: int,
                             title: str = "Mel-Spectrogramme",
                             save_name: Optional[str] = None) -> plt.Figure:
        """
        Affiche le mel-spectrogramme d'un signal audio.
        
        Args:
            y: Signal audio
            sr: Fréquence d'échantillonnage
            title: Titre du graphique
            save_name: Nom du fichier pour sauvegarder
            
        Returns:
            Figure matplotlib
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=Config.N_MELS)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        img = librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', 
                                        y_axis='mel', ax=ax)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        
        plt.tight_layout()
        
        if save_name:
            fig.savefig(self.save_path / save_name, dpi=self.dpi, bbox_inches='tight')
            
        return fig
    
    def plot_mfcc(self, y: np.ndarray, sr: int,
                  title: str = "MFCC",
                  save_name: Optional[str] = None) -> plt.Figure:
        """
        Affiche les coefficients MFCC.
        
        Args:
            y: Signal audio
            sr: Fréquence d'échantillonnage
            title: Titre du graphique
            save_name: Nom du fichier pour sauvegarder
            
        Returns:
            Figure matplotlib
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=Config.N_MFCC)
        img = librosa.display.specshow(mfcc, sr=sr, x_axis='time', ax=ax)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel("Coefficient MFCC")
        fig.colorbar(img, ax=ax)
        
        plt.tight_layout()
        
        if save_name:
            fig.savefig(self.save_path / save_name, dpi=self.dpi, bbox_inches='tight')
            
        return fig
    
    def plot_chroma(self, y: np.ndarray, sr: int,
                    title: str = "Chromagramme",
                    save_name: Optional[str] = None) -> plt.Figure:
        """
        Affiche le chromagramme.
        
        Args:
            y: Signal audio
            sr: Fréquence d'échantillonnage
            title: Titre du graphique
            save_name: Nom du fichier pour sauvegarder
            
        Returns:
            Figure matplotlib
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        img = librosa.display.specshow(chroma, sr=sr, x_axis='time', 
                                        y_axis='chroma', ax=ax)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        fig.colorbar(img, ax=ax)
        
        plt.tight_layout()
        
        if save_name:
            fig.savefig(self.save_path / save_name, dpi=self.dpi, bbox_inches='tight')
            
        return fig
    
    def plot_audio_analysis(self, y: np.ndarray, sr: int,
                            title: str = "Analyse Audio Complète",
                            save_name: Optional[str] = None) -> plt.Figure:
        """
        Affiche une analyse complète du signal audio (4 graphiques).
        
        Args:
            y: Signal audio
            sr: Fréquence d'échantillonnage
            title: Titre principal
            save_name: Nom du fichier pour sauvegarder
            
        Returns:
            Figure matplotlib
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 1. Forme d'onde
        librosa.display.waveshow(y, sr=sr, ax=axes[0, 0], color='steelblue')
        axes[0, 0].set_title("Forme d'onde")
        axes[0, 0].set_xlabel("Temps (s)")
        axes[0, 0].set_ylabel("Amplitude")
        
        # 2. Spectrogramme
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        img1 = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', 
                                         ax=axes[0, 1])
        axes[0, 1].set_title("Spectrogramme")
        fig.colorbar(img1, ax=axes[0, 1], format="%+2.0f dB")
        
        # 3. Mel-Spectrogramme
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=Config.N_MELS)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        img2 = librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', 
                                         y_axis='mel', ax=axes[1, 0])
        axes[1, 0].set_title("Mel-Spectrogramme")
        fig.colorbar(img2, ax=axes[1, 0], format="%+2.0f dB")
        
        # 4. MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=Config.N_MFCC)
        img3 = librosa.display.specshow(mfcc, sr=sr, x_axis='time', ax=axes[1, 1])
        axes[1, 1].set_title("MFCC")
        axes[1, 1].set_ylabel("Coefficient MFCC")
        fig.colorbar(img3, ax=axes[1, 1])
        
        plt.tight_layout()
        
        if save_name:
            fig.savefig(self.save_path / save_name, dpi=self.dpi, bbox_inches='tight')
            
        return fig
    
    def plot_genre_distribution(self, df: pd.DataFrame,
                                save_name: Optional[str] = None) -> plt.Figure:
        """
        Affiche la distribution des genres.
        
        Args:
            df: DataFrame avec la colonne 'genre'
            save_name: Nom du fichier pour sauvegarder
            
        Returns:
            Figure matplotlib
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        genre_counts = df['genre'].value_counts()
        colors = sns.color_palette("husl", len(genre_counts))
        
        bars = ax.bar(genre_counts.index, genre_counts.values, color=colors)
        
        ax.set_title("Distribution des Genres Musicaux", fontsize=14, fontweight='bold')
        ax.set_xlabel("Genre")
        ax.set_ylabel("Nombre de fichiers")
        
        # Ajouter les valeurs sur les barres
        for bar, count in zip(bars, genre_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    str(count), ha='center', va='bottom', fontsize=10)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_name:
            fig.savefig(self.save_path / save_name, dpi=self.dpi, bbox_inches='tight')
            
        return fig
    
    def plot_feature_distribution(self, df: pd.DataFrame, 
                                  feature: str,
                                  save_name: Optional[str] = None) -> plt.Figure:
        """
        Affiche la distribution d'une feature par genre.
        
        Args:
            df: DataFrame avec les features
            feature: Nom de la feature à visualiser
            save_name: Nom du fichier pour sauvegarder
            
        Returns:
            Figure matplotlib
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        sns.boxplot(data=df, x='genre', y=feature, ax=ax, palette="husl")
        
        ax.set_title(f"Distribution de '{feature}' par Genre", 
                     fontsize=14, fontweight='bold')
        ax.set_xlabel("Genre")
        ax.set_ylabel(feature)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_name:
            fig.savefig(self.save_path / save_name, dpi=self.dpi, bbox_inches='tight')
            
        return fig
    
    def plot_feature_boxplots(self, df: pd.DataFrame, 
                              features: List[str],
                              save_name: Optional[str] = None) -> plt.Figure:
        """
        Affiche les boxplots de plusieurs features par genre.
        
        Args:
            df: DataFrame avec les features
            features: Liste des features à visualiser
            save_name: Nom du fichier pour sauvegarder
            
        Returns:
            Figure matplotlib
        """
        n_features = len(features)
        n_cols = 2
        n_rows = (n_features + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4*n_rows))
        axes = axes.flatten()
        
        for idx, feature in enumerate(features):
            sns.boxplot(data=df, x='genre', y=feature, ax=axes[idx], palette="husl")
            axes[idx].set_title(f"{feature}", fontsize=12)
            axes[idx].set_xlabel("")
            axes[idx].tick_params(axis='x', rotation=45)
        
        # Masquer les axes vides
        for idx in range(n_features, len(axes)):
            axes[idx].set_visible(False)
        
        fig.suptitle("Distribution des Caractéristiques par Genre", 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_name:
            fig.savefig(self.save_path / save_name, dpi=self.dpi, bbox_inches='tight')
            
        return fig
    
    def plot_correlation_matrix(self, df: pd.DataFrame,
                                features: Optional[List[str]] = None,
                                save_name: Optional[str] = None) -> plt.Figure:
        """
        Affiche la matrice de corrélation des features.
        
        Args:
            df: DataFrame avec les features
            features: Liste des features (optionnel, sinon toutes les numériques)
            save_name: Nom du fichier pour sauvegarder
            
        Returns:
            Figure matplotlib
        """
        if features is None:
            # Exclure les colonnes non numériques
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            features = [c for c in numeric_cols if c not in ['filename']]
        
        # Limiter à 30 features pour la lisibilité
        if len(features) > 30:
            features = features[:30]
        
        corr_matrix = df[features].corr()
        
        fig, ax = plt.subplots(figsize=(14, 12))
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='RdBu_r',
                    center=0, square=True, ax=ax, cbar_kws={"shrink": 0.8})
        
        ax.set_title("Matrice de Corrélation des Caractéristiques", 
                     fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_name:
            fig.savefig(self.save_path / save_name, dpi=self.dpi, bbox_inches='tight')
            
        return fig
    
    def plot_pca_2d(self, df: pd.DataFrame,
                    save_name: Optional[str] = None) -> plt.Figure:
        """
        Affiche la projection PCA en 2D des genres.
        
        Args:
            df: DataFrame avec les features et la colonne 'genre'
            save_name: Nom du fichier pour sauvegarder
            
        Returns:
            Figure matplotlib
        """
        # Préparer les données
        feature_cols = [c for c in df.columns 
                        if c not in ['filename', 'genre', 'filepath']]
        X = df[feature_cols].values
        y = df['genre'].values
        
        # Standardiser
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Couleurs par genre
        genres = np.unique(y)
        colors = sns.color_palette("husl", len(genres))
        
        for genre, color in zip(genres, colors):
            mask = y == genre
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       c=[color], label=genre, alpha=0.7, s=50)
        
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        ax.set_title("Projection PCA des Genres Musicaux", fontsize=14, fontweight='bold')
        ax.legend(title="Genre", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if save_name:
            fig.savefig(self.save_path / save_name, dpi=self.dpi, bbox_inches='tight')
            
        return fig
    
    def plot_tsne_2d(self, df: pd.DataFrame,
                     perplexity: int = 30,
                     save_name: Optional[str] = None) -> plt.Figure:
        """
        Affiche la projection t-SNE en 2D des genres.
        
        Args:
            df: DataFrame avec les features et la colonne 'genre'
            perplexity: Paramètre de perplexité pour t-SNE
            save_name: Nom du fichier pour sauvegarder
            
        Returns:
            Figure matplotlib
        """
        # Préparer les données
        feature_cols = [c for c in df.columns 
                        if c not in ['filename', 'genre', 'filepath']]
        X = df[feature_cols].values
        y = df['genre'].values
        
        # Standardiser
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # t-SNE
        print("⏳ Calcul t-SNE en cours (peut prendre quelques minutes)...")
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=Config.RANDOM_STATE)
        X_tsne = tsne.fit_transform(X_scaled)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Couleurs par genre
        genres = np.unique(y)
        colors = sns.color_palette("husl", len(genres))
        
        for genre, color in zip(genres, colors):
            mask = y == genre
            ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                       c=[color], label=genre, alpha=0.7, s=50)
        
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_title("Projection t-SNE des Genres Musicaux", fontsize=14, fontweight='bold')
        ax.legend(title="Genre", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if save_name:
            fig.savefig(self.save_path / save_name, dpi=self.dpi, bbox_inches='tight')
            
        return fig
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                              labels: List[str],
                              title: str = "Matrice de Confusion",
                              save_name: Optional[str] = None) -> plt.Figure:
        """
        Affiche la matrice de confusion.
        
        Args:
            y_true: Labels réels
            y_pred: Labels prédits
            labels: Liste des noms de classes
            title: Titre du graphique
            save_name: Nom du fichier pour sauvegarder
            
        Returns:
            Figure matplotlib
        """
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels, ax=ax)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Prédit")
        ax.set_ylabel("Réel")
        
        plt.tight_layout()
        
        if save_name:
            fig.savefig(self.save_path / save_name, dpi=self.dpi, bbox_inches='tight')
            
        return fig
    
    def plot_model_comparison(self, results: Dict[str, Dict[str, float]],
                              metric: str = 'accuracy',
                              save_name: Optional[str] = None) -> plt.Figure:
        """
        Compare les performances de plusieurs modèles.
        
        Args:
            results: Dictionnaire {nom_modèle: {métrique: valeur}}
            metric: Métrique à comparer
            save_name: Nom du fichier pour sauvegarder
            
        Returns:
            Figure matplotlib
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models = list(results.keys())
        scores = [results[m][metric] for m in models]
        colors = sns.color_palette("husl", len(models))
        
        bars = ax.barh(models, scores, color=colors)
        
        ax.set_xlabel(metric.capitalize())
        ax.set_title(f"Comparaison des Modèles ({metric})", 
                     fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1)
        
        # Ajouter les valeurs sur les barres
        for bar, score in zip(bars, scores):
            ax.text(score + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{score:.3f}', va='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_name:
            fig.savefig(self.save_path / save_name, dpi=self.dpi, bbox_inches='tight')
            
        return fig


# Test du module
if __name__ == "__main__":
    visualizer = Visualizer()
    print("✅ Module de visualisation chargé avec succès.")
