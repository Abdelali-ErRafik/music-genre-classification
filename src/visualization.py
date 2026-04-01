import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .config import Config

# style des graphiques
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


class Visualizer:

    def __init__(self, save_path=None):
        self.save_path = save_path or Config.REPORTS_DIR
        self.figsize = (12, 6)
        self.dpi = 100
        self.sample_rate = Config.SAMPLE_RATE

    def plot_waveform(self, y, sr, title="Forme d'onde", save_name=None):
        # afficher la forme d'onde
        fig, ax = plt.subplots(figsize=self.figsize)
        librosa.display.waveshow(y, sr=sr, ax=ax, color="steelblue")
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Temps (s)")
        ax.set_ylabel("Amplitude")
        plt.tight_layout()
        if save_name:
            fig.savefig(self.save_path / save_name, dpi=self.dpi, bbox_inches="tight")
        return fig

    def plot_spectrogram(self, y, sr, title="Spectrogramme", save_name=None):
        # afficher le spectrogramme
        fig, ax = plt.subplots(figsize=self.figsize)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        img = librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="hz", ax=ax)
        ax.set_title(title, fontsize=14)
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        plt.tight_layout()
        if save_name:
            fig.savefig(self.save_path / save_name, dpi=self.dpi, bbox_inches="tight")
        return fig

    def plot_mel_spectrogram(self, y, sr, title="Mel-Spectrogramme", save_name=None):
        # mel spectrogramme
        fig, ax = plt.subplots(figsize=self.figsize)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=Config.N_MELS)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        img = librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel", ax=ax)
        ax.set_title(title, fontsize=14)
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        plt.tight_layout()
        if save_name:
            fig.savefig(self.save_path / save_name, dpi=self.dpi, bbox_inches="tight")
        return fig

    def plot_mfcc(self, y, sr, title="MFCC", save_name=None):
        # afficher les coefficients MFCC
        fig, ax = plt.subplots(figsize=self.figsize)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=Config.N_MFCC)
        img = librosa.display.specshow(mfcc, sr=sr, x_axis="time", ax=ax)
        ax.set_title(title, fontsize=14)
        ax.set_ylabel("Coefficient MFCC")
        fig.colorbar(img, ax=ax)
        plt.tight_layout()
        if save_name:
            fig.savefig(self.save_path / save_name, dpi=self.dpi, bbox_inches="tight")
        return fig

    def plot_chroma(self, y, sr, title="Chromagramme", save_name=None):
        # chromagramme
        fig, ax = plt.subplots(figsize=self.figsize)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        img = librosa.display.specshow(chroma, sr=sr, x_axis="time", y_axis="chroma", ax=ax)
        ax.set_title(title, fontsize=14)
        fig.colorbar(img, ax=ax)
        plt.tight_layout()
        if save_name:
            fig.savefig(self.save_path / save_name, dpi=self.dpi, bbox_inches="tight")
        return fig

    def plot_genre_distribution(self, df, save_name=None):
        # distribution des genres
        fig, ax = plt.subplots(figsize=(10, 6))
        counts = df["genre"].value_counts()
        colors = sns.color_palette("husl", len(counts))
        bars = ax.bar(counts.index, counts.values, color=colors)
        ax.set_title("Distribution des Genres Musicaux", fontsize=14)
        ax.set_xlabel("Genre")
        ax.set_ylabel("Nombre de fichiers")
        for bar, count in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    str(count), ha="center", fontsize=10)
        plt.xticks(rotation=45)
        plt.tight_layout()
        if save_name:
            fig.savefig(self.save_path / save_name, dpi=self.dpi, bbox_inches="tight")
        return fig

    def plot_feature_distribution(self, df, feature, save_name=None):
        # boxplot d'une feature par genre
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=df, x="genre", y=feature, ax=ax, palette="husl")
        ax.set_title(f"Distribution de '{feature}' par Genre", fontsize=14)
        ax.set_xlabel("Genre")
        ax.set_ylabel(feature)
        plt.xticks(rotation=45)
        plt.tight_layout()
        if save_name:
            fig.savefig(self.save_path / save_name, dpi=self.dpi, bbox_inches="tight")
        return fig

    def plot_correlation_matrix(self, df, features=None, save_name=None):
        # matrice de correlation
        if features is None:
            num_cols = df.select_dtypes(include=[np.number]).columns
            features = [c for c in num_cols if c not in ["filename"]]

        # limiter pour la lisibilite
        if len(features) > 30:
            features = features[:30]

        corr = df[features].corr()
        fig, ax = plt.subplots(figsize=(14, 12))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=False, cmap="RdBu_r",
                    center=0, square=True, ax=ax)
        ax.set_title("Matrice de Correlation", fontsize=14)
        plt.tight_layout()
        if save_name:
            fig.savefig(self.save_path / save_name, dpi=self.dpi, bbox_inches="tight")
        return fig

    def plot_pca_2d(self, df, save_name=None):
        # projection PCA en 2D
        feat_cols = [c for c in df.columns if c not in ["filename", "genre", "filepath"]]
        X = df[feat_cols].values
        y = df["genre"].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        fig, ax = plt.subplots(figsize=(12, 8))
        genres = np.unique(y)
        colors = sns.color_palette("husl", len(genres))

        for genre, color in zip(genres, colors):
            mask = y == genre
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=[color],
                      label=genre, alpha=0.7, s=50)

        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        ax.set_title("Projection PCA des Genres Musicaux", fontsize=14)
        ax.legend(title="Genre", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        if save_name:
            fig.savefig(self.save_path / save_name, dpi=self.dpi, bbox_inches="tight")
        return fig


if __name__ == "__main__":
    visualizer = Visualizer()
    print("Module de visualisation charge.")
