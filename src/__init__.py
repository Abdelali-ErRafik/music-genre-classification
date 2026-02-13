"""
Classification des Genres Musicaux
===================================

Ce package contient tous les modules nécessaires pour la classification
automatique des genres musicaux à partir des caractéristiques audio.

Modules:
    - config: Configuration et paramètres du projet
    - data_loader: Chargement et gestion des données audio
    - feature_extraction: Extraction des caractéristiques audio
    - visualization: Fonctions de visualisation
    - models: Définition et entraînement des modèles
    - evaluation: Métriques et évaluation des performances
    - utils: Fonctions utilitaires
"""

from .config import Config
from .data_loader import DataLoader
from .feature_extraction import FeatureExtractor
from .visualization import Visualizer
from .models import ModelTrainer
from .evaluation import Evaluator
from .utils import setup_logging, timer

__version__ = "1.0.0"
__author__ = "Votre Équipe"

__all__ = [
    "Config",
    "DataLoader", 
    "FeatureExtractor",
    "Visualizer",
    "ModelTrainer",
    "Evaluator",
    "setup_logging",
    "timer"
]
