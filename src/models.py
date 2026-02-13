"""
Module des Modèles de Machine Learning
======================================

Ce module contient les définitions et l'entraînement des modèles
de classification pour la reconnaissance des genres musicaux.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import warnings

# Scikit-learn
from sklearn.model_selection import (
    train_test_split, 
    cross_val_score, 
    GridSearchCV,
    StratifiedKFold
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    AdaBoostClassifier,
    VotingClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

from .config import Config

warnings.filterwarnings('ignore')


class ModelTrainer:
    """
    Classe pour entraîner et gérer les modèles de classification.
    
    Cette classe fournit des méthodes pour:
    - Préparer les données pour l'entraînement
    - Entraîner différents modèles de ML
    - Optimiser les hyperparamètres
    - Sauvegarder et charger les modèles
    
    Attributes:
        models: Dictionnaire des modèles disponibles
        scaler: StandardScaler pour normaliser les features
        label_encoder: LabelEncoder pour encoder les genres
    """
    
    def __init__(self):
        """Initialise le ModelTrainer avec les modèles par défaut."""
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.trained_models = {}
        self.best_model = None
        self.best_model_name = None
        
        # Définir les modèles disponibles
        self.models = self._get_default_models()
        
    def _get_default_models(self) -> Dict[str, Any]:
        """
        Retourne les modèles par défaut.
        
        Returns:
            Dictionnaire {nom: modèle}
        """
        return {
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'SVM': SVC(kernel='rbf', C=10, gamma='scale', probability=True),
            'Random Forest': RandomForestClassifier(
                n_estimators=200, 
                max_depth=20,
                random_state=Config.RANDOM_STATE
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=Config.RANDOM_STATE
            ),
            'MLP': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                max_iter=500,
                random_state=Config.RANDOM_STATE
            ),
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=Config.RANDOM_STATE
            ),
            'Decision Tree': DecisionTreeClassifier(
                max_depth=20,
                random_state=Config.RANDOM_STATE
            ),
            'Naive Bayes': GaussianNB(),
            'AdaBoost': AdaBoostClassifier(
                n_estimators=100,
                random_state=Config.RANDOM_STATE
            )
        }
    
    def prepare_data(self, df: pd.DataFrame,
                     test_size: float = 0.2,
                     val_size: float = 0.1) -> Tuple:
        """
        Prépare les données pour l'entraînement.
        
        Args:
            df: DataFrame avec les features et la colonne 'genre'
            test_size: Proportion pour le test
            val_size: Proportion pour la validation
            
        Returns:
            Tuple (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Séparer features et labels
        feature_cols = [c for c in df.columns 
                        if c not in ['filename', 'genre', 'filepath']]
        
        X = df[feature_cols].values
        y = df['genre'].values
        
        # Encoder les labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Premier split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y_encoded, 
            test_size=test_size,
            stratify=y_encoded,
            random_state=Config.RANDOM_STATE
        )
        
        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio,
            stratify=y_temp,
            random_state=Config.RANDOM_STATE
        )
        
        # Normaliser les features
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)
        
        print(f"✅ Données préparées:")
        print(f"   - Entraînement : {len(X_train)} échantillons")
        print(f"   - Validation   : {len(X_val)} échantillons")
        print(f"   - Test         : {len(X_test)} échantillons")
        print(f"   - Features     : {X_train.shape[1]}")
        print(f"   - Classes      : {len(self.label_encoder.classes_)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_model(self, model_name: str, 
                    X_train: np.ndarray, 
                    y_train: np.ndarray,
                    X_val: Optional[np.ndarray] = None,
                    y_val: Optional[np.ndarray] = None) -> Dict:
        """
        Entraîne un modèle spécifique.
        
        Args:
            model_name: Nom du modèle
            X_train: Features d'entraînement
            y_train: Labels d'entraînement
            X_val: Features de validation (optionnel)
            y_val: Labels de validation (optionnel)
            
        Returns:
            Dictionnaire avec les résultats
        """
        if model_name not in self.models:
            raise ValueError(f"Modèle inconnu: {model_name}")
        
        print(f"🔄 Entraînement de {model_name}...")
        
        model = self.models[model_name]
        model.fit(X_train, y_train)
        
        # Calculer les scores
        train_score = model.score(X_train, y_train)
        val_score = model.score(X_val, y_val) if X_val is not None else None
        
        # Sauvegarder le modèle entraîné
        self.trained_models[model_name] = model
        
        results = {
            'model_name': model_name,
            'train_accuracy': train_score,
            'val_accuracy': val_score
        }
        
        print(f"   ✅ Train Accuracy: {train_score:.4f}")
        if val_score:
            print(f"   ✅ Val Accuracy  : {val_score:.4f}")
        
        return results
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray) -> pd.DataFrame:
        """
        Entraîne tous les modèles disponibles.
        
        Args:
            X_train: Features d'entraînement
            y_train: Labels d'entraînement
            X_val: Features de validation
            y_val: Labels de validation
            
        Returns:
            DataFrame avec les résultats de tous les modèles
        """
        print("\n" + "=" * 50)
        print("ENTRAÎNEMENT DE TOUS LES MODÈLES")
        print("=" * 50 + "\n")
        
        all_results = []
        
        for model_name in self.models.keys():
            try:
                results = self.train_model(model_name, X_train, y_train, X_val, y_val)
                all_results.append(results)
            except Exception as e:
                print(f"   ❌ Erreur pour {model_name}: {e}")
        
        # Créer le DataFrame des résultats
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values('val_accuracy', ascending=False)
        
        # Identifier le meilleur modèle
        self.best_model_name = results_df.iloc[0]['model_name']
        self.best_model = self.trained_models[self.best_model_name]
        
        print("\n" + "=" * 50)
        print(f"🏆 Meilleur modèle: {self.best_model_name}")
        print(f"   Accuracy: {results_df.iloc[0]['val_accuracy']:.4f}")
        print("=" * 50)
        
        return results_df
    
    def cross_validate(self, model_name: str, 
                       X: np.ndarray, 
                       y: np.ndarray,
                       cv: int = 5) -> Dict:
        """
        Effectue une validation croisée pour un modèle.
        
        Args:
            model_name: Nom du modèle
            X: Features
            y: Labels
            cv: Nombre de folds
            
        Returns:
            Dictionnaire avec les scores de cross-validation
        """
        if model_name not in self.models:
            raise ValueError(f"Modèle inconnu: {model_name}")
        
        print(f"🔄 Cross-validation pour {model_name} ({cv} folds)...")
        
        model = self.models[model_name]
        
        # Stratified K-Fold
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=Config.RANDOM_STATE)
        
        scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
        
        results = {
            'model_name': model_name,
            'cv_scores': scores,
            'cv_mean': np.mean(scores),
            'cv_std': np.std(scores)
        }
        
        print(f"   ✅ CV Accuracy: {results['cv_mean']:.4f} (+/- {results['cv_std']:.4f})")
        
        return results
    
    def cross_validate_all(self, X: np.ndarray, y: np.ndarray,
                           cv: int = 5) -> pd.DataFrame:
        """
        Effectue une validation croisée pour tous les modèles.
        
        Args:
            X: Features (normalisées)
            y: Labels
            cv: Nombre de folds
            
        Returns:
            DataFrame avec les résultats de cross-validation
        """
        print("\n" + "=" * 50)
        print(f"CROSS-VALIDATION ({cv} FOLDS)")
        print("=" * 50 + "\n")
        
        all_results = []
        
        for model_name in tqdm(self.models.keys(), desc="Cross-validation"):
            try:
                results = self.cross_validate(model_name, X, y, cv)
                all_results.append({
                    'model_name': results['model_name'],
                    'cv_mean': results['cv_mean'],
                    'cv_std': results['cv_std']
                })
            except Exception as e:
                print(f"   ❌ Erreur pour {model_name}: {e}")
        
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values('cv_mean', ascending=False)
        
        return results_df
    
    def hyperparameter_tuning(self, model_name: str,
                              X_train: np.ndarray,
                              y_train: np.ndarray,
                              param_grid: Optional[Dict] = None,
                              cv: int = 3) -> Dict:
        """
        Optimise les hyperparamètres d'un modèle.
        
        Args:
            model_name: Nom du modèle
            X_train: Features d'entraînement
            y_train: Labels d'entraînement
            param_grid: Grille de paramètres (utilise les défauts si None)
            cv: Nombre de folds
            
        Returns:
            Dictionnaire avec les meilleurs paramètres
        """
        # Grilles de paramètres par défaut
        default_grids = {
            'KNN': Config.KNN_PARAMS,
            'SVM': Config.SVM_PARAMS,
            'Random Forest': Config.RF_PARAMS,
            'Gradient Boosting': Config.GB_PARAMS,
            'MLP': Config.MLP_PARAMS
        }
        
        if param_grid is None:
            if model_name not in default_grids:
                print(f"⚠️  Pas de grille par défaut pour {model_name}")
                return {}
            param_grid = default_grids[model_name]
        
        print(f"🔄 Optimisation des hyperparamètres pour {model_name}...")
        
        model = self.models[model_name]
        
        grid_search = GridSearchCV(
            model, 
            param_grid, 
            cv=cv, 
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        results = {
            'model_name': model_name,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_estimator': grid_search.best_estimator_
        }
        
        print(f"   ✅ Meilleurs paramètres: {results['best_params']}")
        print(f"   ✅ Meilleur score: {results['best_score']:.4f}")
        
        # Mettre à jour le modèle avec les meilleurs paramètres
        self.models[model_name] = results['best_estimator']
        
        return results
    
    def create_ensemble(self, model_names: List[str],
                        voting: str = 'soft') -> VotingClassifier:
        """
        Crée un ensemble de modèles.
        
        Args:
            model_names: Liste des noms de modèles à combiner
            voting: Type de vote ('hard' ou 'soft')
            
        Returns:
            VotingClassifier
        """
        estimators = []
        for name in model_names:
            if name in self.trained_models:
                estimators.append((name, self.trained_models[name]))
            else:
                print(f"⚠️  {name} non entraîné, ignoré")
        
        if len(estimators) < 2:
            raise ValueError("Au moins 2 modèles entraînés sont nécessaires")
        
        ensemble = VotingClassifier(estimators=estimators, voting=voting)
        
        print(f"✅ Ensemble créé avec {len(estimators)} modèles ({voting} voting)")
        
        return ensemble
    
    def predict(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """
        Fait des prédictions avec un modèle entraîné.
        
        Args:
            model_name: Nom du modèle
            X: Features (doivent être normalisées)
            
        Returns:
            Array des prédictions
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Modèle {model_name} non entraîné")
        
        return self.trained_models[model_name].predict(X)
    
    def predict_proba(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """
        Retourne les probabilités de prédiction.
        
        Args:
            model_name: Nom du modèle
            X: Features (doivent être normalisées)
            
        Returns:
            Array des probabilités par classe
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Modèle {model_name} non entraîné")
        
        model = self.trained_models[model_name]
        
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)
        else:
            raise ValueError(f"{model_name} ne supporte pas predict_proba")
    
    def decode_predictions(self, y_pred: np.ndarray) -> np.ndarray:
        """
        Décode les prédictions numériques en noms de genres.
        
        Args:
            y_pred: Array des prédictions encodées
            
        Returns:
            Array des noms de genres
        """
        return self.label_encoder.inverse_transform(y_pred)
    
    def save_model(self, model_name: str, 
                   filepath: Optional[Path] = None) -> str:
        """
        Sauvegarde un modèle entraîné.
        
        Args:
            model_name: Nom du modèle
            filepath: Chemin de sauvegarde (optionnel)
            
        Returns:
            Chemin du fichier sauvegardé
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Modèle {model_name} non entraîné")
        
        if filepath is None:
            filepath = Config.MODELS_DIR / f"{model_name.lower().replace(' ', '_')}.joblib"
        
        # Sauvegarder le modèle, le scaler et le label encoder
        save_dict = {
            'model': self.trained_models[model_name],
            'scaler': self.scaler,
            'label_encoder': self.label_encoder
        }
        
        joblib.dump(save_dict, filepath)
        print(f"💾 Modèle sauvegardé: {filepath}")
        
        return str(filepath)
    
    def load_model(self, filepath: Path) -> str:
        """
        Charge un modèle sauvegardé.
        
        Args:
            filepath: Chemin du fichier
            
        Returns:
            Nom du modèle chargé
        """
        save_dict = joblib.load(filepath)
        
        model_name = filepath.stem.replace('_', ' ').title()
        self.trained_models[model_name] = save_dict['model']
        self.scaler = save_dict['scaler']
        self.label_encoder = save_dict['label_encoder']
        
        print(f"📂 Modèle chargé: {model_name}")
        
        return model_name
    
    def get_feature_importance(self, model_name: str) -> Optional[pd.DataFrame]:
        """
        Retourne l'importance des features pour les modèles qui le supportent.
        
        Args:
            model_name: Nom du modèle
            
        Returns:
            DataFrame avec l'importance des features, ou None
        """
        if model_name not in self.trained_models:
            return None
        
        model = self.trained_models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            return pd.DataFrame({
                'importance': importance
            }).sort_values('importance', ascending=False)
        
        return None
    
    def print_models_summary(self):
        """Affiche un résumé des modèles disponibles et entraînés."""
        print("\n" + "=" * 50)
        print("RÉSUMÉ DES MODÈLES")
        print("=" * 50)
        
        print("\n📋 Modèles disponibles:")
        for name in self.models.keys():
            status = "✅ Entraîné" if name in self.trained_models else "⬜ Non entraîné"
            print(f"   - {name}: {status}")
        
        if self.best_model_name:
            print(f"\n🏆 Meilleur modèle: {self.best_model_name}")
        
        print("=" * 50)


# Test du module
if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.print_models_summary()
