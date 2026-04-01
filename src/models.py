import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from .config import Config

warnings.filterwarnings("ignore")


class ModelTrainer:

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.trained_models = {}
        self.best_model = None
        self.best_model_name = None
        # on definit nos 3 modeles
        self.models = self._get_default_models()

    def _get_default_models(self):
        return {
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "SVM": SVC(kernel="rbf", C=10, gamma="scale", probability=True),
            "Random Forest": RandomForestClassifier(
                n_estimators=200, max_depth=20, random_state=Config.RANDOM_STATE
            ),
        }

    def prepare_data(self, df, test_size=0.2, val_size=0.1):
        # on separe features et labels
        feat_cols = [c for c in df.columns if c not in ["filename", "genre", "filepath"]]
        X = df[feat_cols].values
        y = df["genre"].values

        # encoder les labels
        y_enc = self.label_encoder.fit_transform(y)

        # split train+val / test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y_enc, test_size=test_size,
            stratify=y_enc, random_state=Config.RANDOM_STATE,
        )

        # split train / val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio,
            stratify=y_temp, random_state=Config.RANDOM_STATE,
        )

        # normaliser les features
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)

        print(f"Donnees preparees:")
        print(f"  Train : {len(X_train)} echantillons")
        print(f"  Val   : {len(X_val)} echantillons")
        print(f"  Test  : {len(X_test)} echantillons")
        print(f"  Features : {X_train.shape[1]}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_model(self, name, X_train, y_train, X_val=None, y_val=None):
        # entrainer un modele
        if name not in self.models:
            raise ValueError(f"Modele inconnu: {name}")

        print(f"Entrainement de {name}...")
        model = self.models[name]
        model.fit(X_train, y_train)

        train_acc = model.score(X_train, y_train)
        val_acc = model.score(X_val, y_val) if X_val is not None else None

        self.trained_models[name] = model

        result = {
            "model_name": name,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
        }

        print(f"  Train Accuracy: {train_acc:.4f}")
        if val_acc:
            print(f"  Val Accuracy  : {val_acc:.4f}")

        return result

    def train_all_models(self, X_train, y_train, X_val, y_val):
        # entrainer tous les modeles
        print("\n" + "=" * 40)
        print("ENTRAINEMENT DES MODELES")
        print("=" * 40 + "\n")

        results = []
        for name in self.models:
            try:
                res = self.train_model(name, X_train, y_train, X_val, y_val)
                results.append(res)
            except Exception as e:
                print(f"Erreur pour {name}: {e}")

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values("val_accuracy", ascending=False)

        # garder le meilleur modele
        self.best_model_name = results_df.iloc[0]["model_name"]
        self.best_model = self.trained_models[self.best_model_name]

        print(f"\nMeilleur modele: {self.best_model_name}")
        print(f"Accuracy: {results_df.iloc[0]['val_accuracy']:.4f}")

        return results_df

    def predict(self, name, X):
        if name not in self.trained_models:
            raise ValueError(f"Modele {name} non entraine")
        return self.trained_models[name].predict(X)

    def save_model(self, name, filepath=None):
        # sauvegarder le modele avec le scaler et le label encoder
        if name not in self.trained_models:
            raise ValueError(f"Modele {name} non entraine")

        if filepath is None:
            filepath = Config.MODELS_DIR / f"{name.lower().replace(' ', '_')}.joblib"

        save_dict = {
            "model": self.trained_models[name],
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,
        }
        joblib.dump(save_dict, filepath)
        print(f"Modele sauvegarde: {filepath}")
        return str(filepath)

    def load_model(self, filepath):
        # charger un modele sauvegarde
        save_dict = joblib.load(filepath)
        name = filepath.stem.replace("_", " ").title()
        self.trained_models[name] = save_dict["model"]
        self.scaler = save_dict["scaler"]
        self.label_encoder = save_dict["label_encoder"]
        print(f"Modele charge: {name}")
        return name


if __name__ == "__main__":
    trainer = ModelTrainer()
    print("Modeles disponibles:", list(trainer.models.keys()))
