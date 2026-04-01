import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.data_loader import DataLoader
from src.feature_extraction import FeatureExtractor
from src.visualization import Visualizer
from src.models import ModelTrainer
from src.evaluation import Evaluator
from src.utils import print_header, print_section, Timer, check_dependencies, set_random_seed


def step_extract_features():
    # etape 1 : extraction des features audio
    print_header("ETAPE 1: EXTRACTION DES CARACTERISTIQUES")

    loader = DataLoader()
    df = loader.scan_dataset()

    if len(df) == 0:
        print("Aucun fichier audio trouve!")
        print(f"Placez les fichiers dans: {Config.DATA_RAW}")
        return None

    loader.print_dataset_summary(df)

    # extraire les features
    print_section("Extraction des features")
    extractor = FeatureExtractor()
    features_path = Config.DATA_PROCESSED / Config.FEATURES_FILE

    with Timer("Extraction des caracteristiques"):
        features_df = extractor.extract_features_from_dataset(df, save_path=features_path)

    return features_df


def step_visualize(features_df):
    # etape 2 : visualisations
    print_header("ETAPE 2: VISUALISATION DES DONNEES")

    vis = Visualizer()

    # distribution des genres
    vis.plot_genre_distribution(features_df, save_name="genre_distribution.png")

    # quelques features importantes
    for feature in ["spectral_centroid_mean", "tempo", "zero_crossing_rate_mean", "mfcc_1_mean"]:
        if feature in features_df.columns:
            vis.plot_feature_distribution(features_df, feature, save_name=f"dist_{feature}.png")

    # matrice de correlation
    vis.plot_correlation_matrix(features_df, save_name="correlation_matrix.png")

    # PCA
    vis.plot_pca_2d(features_df, save_name="pca_projection.png")

    print("Visualisations sauvegardees dans:", Config.REPORTS_DIR)


def step_train_models(features_df):
    # etape 3 : entrainement des modeles
    print_header("ETAPE 3: ENTRAINEMENT DES MODELES")

    trainer = ModelTrainer()

    # preparer les donnees
    print_section("Preparation des donnees")
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(features_df)

    # entrainer tous les modeles
    print_section("Entrainement")
    with Timer("Entrainement de tous les modeles"):
        results_df = trainer.train_all_models(X_train, y_train, X_val, y_val)

    print("\nResultats:")
    print(results_df.to_string(index=False))

    # sauvegarder le meilleur modele
    print_section("Sauvegarde du modele")
    trainer.save_model(trainer.best_model_name)

    return trainer, X_test, y_test


def step_evaluate(trainer, X_test, y_test):
    # etape 4 : evaluation
    print_header("ETAPE 4: EVALUATION DES MODELES")

    evaluator = Evaluator()
    best_name = trainer.best_model_name

    # predictions sur le test set
    print_section(f"Evaluation de {best_name}")
    y_pred = trainer.predict(best_name, X_test)

    # afficher les metriques
    evaluator.print_evaluation(best_name, y_test, y_pred)

    # matrice de confusion
    evaluator.plot_confusion_matrix(
        y_test, y_pred, normalize=True,
        title=f"Matrice de Confusion - {best_name}",
        save_name="confusion_matrix.png",
    )

    # rapport de classification
    evaluator.plot_classification_report(
        y_test, y_pred,
        title=f"Rapport - {best_name}",
        save_name="classification_report.png",
    )

    # comparer tous les modeles
    print_section("Comparaison des modeles")
    all_results = {}
    for name in trainer.trained_models:
        y_pred_m = trainer.predict(name, X_test)
        metrics = evaluator.calculate_metrics(y_test, y_pred_m)
        all_results[name] = {"metrics": metrics}

    comp_df = evaluator.compare_models(all_results)
    print("\nComparaison finale:")
    print(comp_df.to_string(index=False))

    evaluator.plot_model_comparison(all_results, save_name="model_comparison.png")

    print(f"\nResultats sauvegardes dans: {Config.REPORTS_DIR}")


def main():
    print_header("CLASSIFICATION DES GENRES MUSICAUX", "=", 60)

    # verifier les dependances
    deps = check_dependencies()
    missing = [p for p, ok in deps.items() if not ok]
    if missing:
        print(f"Packages manquants: {missing}")
        print("Installez avec: pip install -r requirements.txt")
        return

    # creer les repertoires
    Config.create_directories()

    # fixer la graine aleatoire
    set_random_seed(42)

    # etape 1 : extraction
    features_df = step_extract_features()
    if features_df is None:
        return

    # etape 2 : visualisation
    step_visualize(features_df)

    # etape 3 : entrainement
    trainer, X_test, y_test = step_train_models(features_df)

    # etape 4 : evaluation
    step_evaluate(trainer, X_test, y_test)

    print_header("PIPELINE TERMINE", "=", 60)


if __name__ == "__main__":
    main()
