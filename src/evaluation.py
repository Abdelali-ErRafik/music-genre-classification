import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from .config import Config


class Evaluator:

    def __init__(self, genres=None, save_path=None):
        self.genres = genres or Config.GENRES
        self.save_path = save_path or Config.REPORTS_DIR

    def calculate_metrics(self, y_true, y_pred, average="weighted"):
        # calculer les metriques de classification
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
            "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, average=average, zero_division=0),
        }
        return metrics

    def get_confusion_matrix(self, y_true, y_pred, normalize=None):
        return confusion_matrix(y_true, y_pred, normalize=normalize)

    def plot_confusion_matrix(self, y_true, y_pred, normalize=False,
                              title="Matrice de Confusion", save_name=None):
        # afficher la matrice de confusion
        norm = "true" if normalize else None
        cm = self.get_confusion_matrix(y_true, y_pred, normalize=norm)

        fig, ax = plt.subplots(figsize=(12, 10))
        fmt = ".2f" if normalize else "d"
        sns.heatmap(
            cm, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=self.genres, yticklabels=self.genres, ax=ax,
        )
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Predit")
        ax.set_ylabel("Reel")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        if save_name:
            fig.savefig(self.save_path / save_name, dpi=100, bbox_inches="tight")
        return fig

    def plot_classification_report(self, y_true, y_pred,
                                    title="Rapport de Classification", save_name=None):
        # visualiser precision, recall, f1 par genre
        report = classification_report(
            y_true, y_pred, target_names=self.genres,
            output_dict=True, zero_division=0
        )
        class_metrics = pd.DataFrame(report).transpose().iloc[:len(self.genres)]

        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        fig.suptitle(title, fontsize=14)

        for ax, metric, color in zip(axes, ["precision", "recall", "f1-score"],
                                      ["steelblue", "coral", "seagreen"]):
            vals = class_metrics[metric].values
            ax.barh(self.genres, vals, color=color)
            ax.set_xlim(0, 1)
            ax.set_title(metric.capitalize())
            for i, v in enumerate(vals):
                ax.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=9)

        plt.tight_layout()
        if save_name:
            fig.savefig(self.save_path / save_name, dpi=100, bbox_inches="tight")
        return fig

    def compare_models(self, results_dict):
        # comparer les resultats de plusieurs modeles
        comparison = []
        for name, res in results_dict.items():
            m = res["metrics"]
            comparison.append({
                "Model": name,
                "Accuracy": m["accuracy"],
                "Precision": m["precision"],
                "Recall": m["recall"],
                "F1-Score": m["f1_score"],
            })
        df = pd.DataFrame(comparison)
        df = df.sort_values("Accuracy", ascending=False)
        return df

    def plot_model_comparison(self, results_dict, save_name=None):
        # graphique de comparaison des modeles
        comp_df = self.compare_models(results_dict)

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(comp_df))
        width = 0.2

        for i, (metric, color) in enumerate(zip(
            ["Accuracy", "Precision", "Recall", "F1-Score"],
            ["steelblue", "coral", "seagreen", "purple"]
        )):
            offset = (i - 1.5) * width
            ax.bar(x + offset, comp_df[metric], width, label=metric, color=color, alpha=0.8)

        ax.set_xlabel("Modele")
        ax.set_ylabel("Score")
        ax.set_title("Comparaison des Modeles", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(comp_df["Model"], rotation=45, ha="right")
        ax.legend()
        ax.set_ylim(0, 1)
        plt.tight_layout()

        if save_name:
            fig.savefig(self.save_path / save_name, dpi=100, bbox_inches="tight")
        return fig

    def print_evaluation(self, model_name, y_true, y_pred):
        # afficher les resultats d'evaluation
        metrics = self.calculate_metrics(y_true, y_pred)
        print(f"\n{'='*50}")
        print(f"EVALUATION - {model_name}")
        print(f"{'='*50}")
        print(f"Accuracy  : {metrics['accuracy']:.4f}")
        print(f"Precision : {metrics['precision']:.4f}")
        print(f"Recall    : {metrics['recall']:.4f}")
        print(f"F1-Score  : {metrics['f1_score']:.4f}")
        print(f"\n{classification_report(y_true, y_pred, target_names=self.genres, zero_division=0)}")


if __name__ == "__main__":
    evaluator = Evaluator()
    print("Module d'evaluation charge.")
