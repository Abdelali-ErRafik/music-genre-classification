import time
import numpy as np


def print_header(title, char="=", width=60):
    # afficher un titre
    print("\n" + char * width)
    print(title.center(width))
    print(char * width + "\n")


def print_section(title, char="-", width=40):
    # afficher un sous-titre
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def check_dependencies():
    # verifier que les librairies sont installees
    deps = {
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
    print("\nVerification des dependances:")
    print("-" * 40)

    for module, package in deps.items():
        try:
            __import__(module)
            status[package] = True
            print(f"  [OK] {package}")
        except ImportError:
            status[package] = False
            print(f"  [X]  {package} (manquant)")

    print("-" * 40)
    return status


def set_random_seed(seed=42):
    # fixer la graine pour la reproductibilite
    import random

    random.seed(seed)
    np.random.seed(seed)
    print(f"Graine aleatoire fixee a {seed}")


class Timer:
    # mesurer le temps d'execution

    def __init__(self, description="Operation"):
        self.description = description
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.time()
        print(f"{self.description}...")
        return self

    def __exit__(self, *args):
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        if elapsed > 60:
            mins = int(elapsed // 60)
            secs = int(elapsed % 60)
            print(f"{self.description} termine en {mins}m {secs}s")
        else:
            print(f"{self.description} termine en {elapsed:.1f}s")


if __name__ == "__main__":
    print_header("TEST DES UTILITAIRES")
    check_dependencies()
