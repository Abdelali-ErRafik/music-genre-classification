"""
Module Utilitaire
=================

Ce module contient des fonctions utilitaires générales
pour le projet de classification des genres musicaux.
"""

import os
import sys
import time
import logging
import functools
from pathlib import Path
from datetime import datetime
from typing import Callable, Any, Optional
import numpy as np
import pandas as pd

from .config import Config


def setup_logging(
    log_file: Optional[str] = None, level: int = logging.INFO
) -> logging.Logger:
    """
    Configure le système de logging.

    Args:
        log_file: Chemin du fichier de log (optionnel)
        level: Niveau de logging

    Returns:
        Logger configuré
    """
    logger = logging.getLogger("music_genre_classifier")
    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def timer(func: Callable) -> Callable:
    """
    Décorateur pour mesurer le temps d'exécution d'une fonction.

    Args:
        func: Fonction à décorer

    Returns:
        Fonction décorée
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        elapsed = end_time - start_time
        print(f"⏱️  {func.__name__} exécuté en {elapsed:.2f} secondes")

        return result

    return wrapper


def ensure_directory(path: Path) -> Path:
    """
    Crée un répertoire s'il n'existe pas.

    Args:
        path: Chemin du répertoire

    Returns:
        Path du répertoire
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_timestamp() -> str:
    """
    Retourne un timestamp formaté.

    Returns:
        Timestamp au format YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def format_duration(seconds: float) -> str:
    """
    Formate une durée en secondes en format lisible.

    Args:
        seconds: Durée en secondes

    Returns:
        Durée formatée (ex: "2h 30m 15s")
    """
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def print_header(title: str, char: str = "=", width: int = 60):
    """
    Affiche un titre formaté.

    Args:
        title: Titre à afficher
        char: Caractère de décoration
        width: Largeur totale
    """
    print("\n" + char * width)
    print(title.center(width))
    print(char * width + "\n")


def print_section(title: str, char: str = "-", width: int = 40):
    """
    Affiche un titre de section.

    Args:
        title: Titre de la section
        char: Caractère de décoration
        width: Largeur totale
    """
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def check_dependencies():
    """
    Vérifie que toutes les dépendances sont installées.

    Returns:
        Dictionnaire avec le statut de chaque dépendance
    """
    dependencies = {
        "numpy": "numpy",
        "pandas": "pandas",
        "librosa": "librosa",
        "sklearn": "scikit-learn",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "tqdm": "tqdm",
        "joblib": "joblib",
    }

    status = {}

    print("\n🔍 Vérification des dépendances:")
    print("-" * 40)

    for module, package in dependencies.items():
        try:
            __import__(module)
            status[package] = True
            print(f"   ✅ {package}")
        except ImportError:
            status[package] = False
            print(f"   ❌ {package} (manquant)")

    print("-" * 40)

    return status


def set_random_seed(seed: int = 42):
    """
    Fixe la graine aléatoire pour la reproductibilité.

    Args:
        seed: Valeur de la graine
    """
    import random

    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass

    print(f"🎲 Graine aléatoire fixée à {seed}")


class Timer:
    """
    Context manager pour mesurer le temps d'exécution.

    Usage:
        with Timer("Description"):
            # code à mesurer
    """

    def __init__(self, description: str = "Opération"):
        """
        Initialise le timer.

        Args:
            description: Description de l'opération
        """
        self.description = description
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        """Démarre le timer."""
        self.start_time = time.time()
        print(f"⏳ {self.description}...")
        return self

    def __exit__(self, *args):
        """Arrête le timer et affiche la durée."""
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        print(f"✅ {self.description} terminé en {format_duration(elapsed)}")

    @property
    def elapsed(self) -> float:
        """Retourne le temps écoulé en secondes."""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time


# Test du module
if __name__ == "__main__":
    print_header("TEST DES UTILITAIRES")
    check_dependencies()
    print("\n✅ Module utilitaire chargé avec succès.")
