"""
Module d'Évaluation des Modèles
===============================

Ce module contient toutes les fonctions pour évaluer les performances
des modèles de classification des genres musicaux.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)

from .config import Config


class Evaluator:
    """
    Classe pour évaluer les performances des modèles.
    
    Cette classe fournit des méthodes pour:
    - Calculer les métriques de classification
    - Générer des rapports de classification
    - Visualiser les performances
    - Analyser les erreurs
    
    Attributes:
        genres: Liste des genres musicaux
        save_path: Chemin pour sauvegarder les résultats
    """
    
    def __init__(self, genres: Optional[List[str]] = None,
                 save_path: Optional[Path] = None):
        """
        Initialise l'Evaluator.
        
        Args:
            genres: Liste des genres (utilise Config.GENRES si None)
            save_path: Chemin pour sauvegarder les résultats
        """
        self.genres = genres or Config.GENRES
        self.save_path = save_path or Config.REPORTS_DIR
        
    def calculate_metrics(self, y_true: np.ndarray, 
                          y_pred: np.ndarray,
                          average: str = 'weighted') -> Dict[str, float]:
        """
        Calcule toutes les métriques de classification.
        
        Args:
            y_true: Labels réels
            y_pred: Labels prédits
            average: Type de moyenne ('micro', 'macro', 'weighted')
            
        Returns:
            Dictionnaire avec toutes les métriques
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
            'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0)
        }
        
        return metrics
    
    def calculate_per_class_metrics(self, y_true: np.ndarray,
                                     y_pred: np.ndarray) -> pd.DataFrame:
        """
        Calcule les métriques pour chaque classe.
        
        Args:
            y_true: Labels réels
            y_pred: Labels prédits
            
        Returns:
            DataFrame avec les métriques par classe
        """
        report = classification_report(
            y_true, y_pred,
            target_names=self.genres,
            output_dict=True,
            zero_division=0
        )
        
        # Convertir en DataFrame
        df = pd.DataFrame(report).transpose()
        
        # Garder seulement les classes (pas les moyennes)
        class_df = df.iloc[:len(self.genres)]
        
        return class_df
    
    def get_confusion_matrix(self, y_true: np.ndarray,
                             y_pred: np.ndarray,
                             normalize: Optional[str] = None) -> np.ndarray:
        """
        Calcule la matrice de confusion.
        
        Args:
            y_true: Labels réels
            y_pred: Labels prédits
            normalize: 'true', 'pred', 'all' ou None
            
        Returns:
            Matrice de confusion
        """
        return confusion_matrix(y_true, y_pred, normalize=normalize)
    
    def plot_confusion_matrix(self, y_true: np.ndarray,
                              y_pred: np.ndarray,
                              normalize: bool = False,
                              title: str = "Matrice de Confusion",
                              save_name: Optional[str] = None) -> plt.Figure:
        """
        Affiche la matrice de confusion.
        
        Args:
            y_true: Labels réels
            y_pred: Labels prédits
            normalize: Normaliser la matrice
            title: Titre du graphique
            save_name: Nom du fichier pour sauvegarder
            
        Returns:
            Figure matplotlib
        """
        norm = 'true' if normalize else None
        cm = self.get_confusion_matrix(y_true, y_pred, normalize=norm)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        fmt = '.2f' if normalize else 'd'
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=self.genres, yticklabels=self.genres, ax=ax)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Prédit", fontsize=12)
        ax.set_ylabel("Réel", fontsize=12)
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_name:
            fig.savefig(self.save_path / save_name, dpi=100, bbox_inches='tight')
            
        return fig
    
    def plot_classification_report(self, y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   title: str = "Rapport de Classification",
                                   save_name: Optional[str] = None) -> plt.Figure:
        """
        Visualise le rapport de classification.
        
        Args:
            y_true: Labels réels
            y_pred: Labels prédits
            title: Titre du graphique
            save_name: Nom du fichier pour sauvegarder
            
        Returns:
            Figure matplotlib
        """
        class_metrics = self.calculate_per_class_metrics(y_true, y_pred)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        metrics = ['precision', 'recall', 'f1-score']
        colors = ['steelblue', 'coral', 'seagreen']
        
        for ax, metric, color in zip(axes, metrics, colors):
            values = class_metrics[metric].values
            
            bars = ax.barh(self.genres, values, color=color)
            ax.set_xlabel(metric.capitalize())
            ax.set_xlim(0, 1)
            ax.set_title(metric.capitalize())
            
            # Ajouter les valeurs sur les barres
            for bar, val in zip(bars, values):
                ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                        f'{val:.2f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_name:
            fig.savefig(self.save_path / save_name, dpi=100, bbox_inches='tight')
            
        return fig
    
    def plot_roc_curves(self, y_true: np.ndarray,
                        y_proba: np.ndarray,
                        title: str = "Courbes ROC",
                        save_name: Optional[str] = None) -> plt.Figure:
        """
        Affiche les courbes ROC pour chaque classe.
        
        Args:
            y_true: Labels réels (encodés)
            y_proba: Probabilités prédites (shape: n_samples x n_classes)
            title: Titre du graphique
            save_name: Nom du fichier pour sauvegarder
            
        Returns:
            Figure matplotlib
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        n_classes = len(self.genres)
        colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
        
        for i, (genre, color) in enumerate(zip(self.genres, colors)):
            # Créer des labels binaires pour cette classe
            y_true_binary = (y_true == i).astype(int)
            y_score = y_proba[:, i]
            
            fpr, tpr, _ = roc_curve(y_true_binary, y_score)
            auc = roc_auc_score(y_true_binary, y_score)
            
            ax.plot(fpr, tpr, color=color, lw=2,
                    label=f'{genre} (AUC = {auc:.2f})')
        
        # Ligne diagonale (classifieur aléatoire)
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Aléatoire')
        
        ax.set_xlabel('Taux de Faux Positifs')
        ax.set_ylabel('Taux de Vrais Positifs')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            fig.savefig(self.save_path / save_name, dpi=100, bbox_inches='tight')
            
        return fig
    
    def analyze_errors(self, y_true: np.ndarray,
                       y_pred: np.ndarray,
                       filenames: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Analyse les erreurs de classification.
        
        Args:
            y_true: Labels réels
            y_pred: Labels prédits
            filenames: Noms des fichiers (optionnel)
            
        Returns:
            DataFrame avec les erreurs
        """
        # Trouver les indices des erreurs
        error_idx = np.where(y_true != y_pred)[0]
        
        errors = []
        for idx in error_idx:
            error_info = {
                'index': idx,
                'true_label': self.genres[y_true[idx]] if isinstance(y_true[idx], (int, np.integer)) else y_true[idx],
                'predicted_label': self.genres[y_pred[idx]] if isinstance(y_pred[idx], (int, np.integer)) else y_pred[idx]
            }
            if filenames is not None:
                error_info['filename'] = filenames[idx]
            errors.append(error_info)
        
        errors_df = pd.DataFrame(errors)
        
        return errors_df
    
    def get_most_confused_pairs(self, y_true: np.ndarray,
                                y_pred: np.ndarray,
                                top_n: int = 5) -> List[Tuple[str, str, int]]:
        """
        Identifie les paires de genres les plus souvent confondues.
        
        Args:
            y_true: Labels réels
            y_pred: Labels prédits
            top_n: Nombre de paires à retourner
            
        Returns:
            Liste de tuples (genre_réel, genre_prédit, count)
        """
        cm = self.get_confusion_matrix(y_true, y_pred)
        
        # Ignorer la diagonale (prédictions correctes)
        np.fill_diagonal(cm, 0)
        
        # Trouver les paires les plus confondues
        confused_pairs = []
        for i in range(len(self.genres)):
            for j in range(len(self.genres)):
                if cm[i, j] > 0:
                    confused_pairs.append((self.genres[i], self.genres[j], cm[i, j]))
        
        # Trier par nombre d'erreurs
        confused_pairs.sort(key=lambda x: x[2], reverse=True)
        
        return confused_pairs[:top_n]
    
    def evaluate_model(self, model_name: str,
                       y_true: np.ndarray,
                       y_pred: np.ndarray,
                       y_proba: Optional[np.ndarray] = None) -> Dict:
        """
        Évaluation complète d'un modèle.
        
        Args:
            model_name: Nom du modèle
            y_true: Labels réels
            y_pred: Labels prédits
            y_proba: Probabilités prédites (optionnel)
            
        Returns:
            Dictionnaire avec tous les résultats
        """
        results = {
            'model_name': model_name,
            'metrics': self.calculate_metrics(y_true, y_pred),
            'per_class_metrics': self.calculate_per_class_metrics(y_true, y_pred),
            'confusion_matrix': self.get_confusion_matrix(y_true, y_pred),
            'most_confused': self.get_most_confused_pairs(y_true, y_pred)
        }
        
        return results
    
    def compare_models(self, results_dict: Dict[str, Dict]) -> pd.DataFrame:
        """
        Compare les résultats de plusieurs modèles.
        
        Args:
            results_dict: Dictionnaire {nom_modèle: résultats}
            
        Returns:
            DataFrame de comparaison
        """
        comparison = []
        
        for model_name, results in results_dict.items():
            metrics = results['metrics']
            comparison.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score']
            })
        
        df = pd.DataFrame(comparison)
        df = df.sort_values('Accuracy', ascending=False)
        
        return df
    
    def plot_model_comparison(self, results_dict: Dict[str, Dict],
                              save_name: Optional[str] = None) -> plt.Figure:
        """
        Visualise la comparaison des modèles.
        
        Args:
            results_dict: Dictionnaire {nom_modèle: résultats}
            save_name: Nom du fichier pour sauvegarder
            
        Returns:
            Figure matplotlib
        """
        comparison_df = self.compare_models(results_dict)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(comparison_df))
        width = 0.2
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        colors = ['steelblue', 'coral', 'seagreen', 'purple']
        
        for i, (metric, color) in enumerate(zip(metrics, colors)):
            offset = (i - 1.5) * width
            bars = ax.bar(x + offset, comparison_df[metric], width, 
                          label=metric, color=color, alpha=0.8)
        
        ax.set_xlabel('Modèle')
        ax.set_ylabel('Score')
        ax.set_title('Comparaison des Modèles', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            fig.savefig(self.save_path / save_name, dpi=100, bbox_inches='tight')
            
        return fig
    
    def generate_report(self, model_name: str,
                        y_true: np.ndarray,
                        y_pred: np.ndarray) -> str:
        """
        Génère un rapport textuel complet.
        
        Args:
            model_name: Nom du modèle
            y_true: Labels réels
            y_pred: Labels prédits
            
        Returns:
            Rapport formaté en texte
        """
        metrics = self.calculate_metrics(y_true, y_pred)
        confused = self.get_most_confused_pairs(y_true, y_pred)
        
        report = f"""
{'='*60}
RAPPORT D'ÉVALUATION - {model_name.upper()}
{'='*60}

📊 MÉTRIQUES GLOBALES
{'-'*30}
  • Accuracy  : {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)
  • Precision : {metrics['precision']:.4f}
  • Recall    : {metrics['recall']:.4f}
  • F1-Score  : {metrics['f1_score']:.4f}

📋 RAPPORT PAR CLASSE
{'-'*30}
{classification_report(y_true, y_pred, target_names=self.genres, zero_division=0)}

⚠️  GENRES LES PLUS CONFONDUS
{'-'*30}
"""
        for real, pred, count in confused[:5]:
            report += f"  • {real} → {pred}: {count} erreurs\n"
        
        report += f"\n{'='*60}\n"
        
        return report
    
    def print_evaluation(self, model_name: str,
                         y_true: np.ndarray,
                         y_pred: np.ndarray):
        """
        Affiche l'évaluation complète d'un modèle.
        
        Args:
            model_name: Nom du modèle
            y_true: Labels réels
            y_pred: Labels prédits
        """
        print(self.generate_report(model_name, y_true, y_pred))


# Test du module
if __name__ == "__main__":
    evaluator = Evaluator()
    print("✅ Module d'évaluation chargé avec succès.")
