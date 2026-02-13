#!/usr/bin/env python3
"""
Classification des Genres Musicaux - Script Principal
=====================================================

Ce script exécute le pipeline complet de classification des genres musicaux:
1. Chargement des données
2. Extraction des caractéristiques audio
3. Entraînement des modèles
4. Évaluation et visualisation des résultats

Usage:
    python main.py                    # Pipeline complet
    python main.py --step extract     # Extraction des features uniquement
    python main.py --step train       # Entraînement uniquement
    python main.py --step evaluate    # Évaluation uniquement
    python main.py --help             # Aide

Auteur: Votre Équipe
Date: 2026
"""

import argparse
import sys
from pathlib import Path

# Ajouter le répertoire src au path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.data_loader import DataLoader
from src.feature_extraction import FeatureExtractor
from src.visualization import Visualizer
from src.models import ModelTrainer
from src.evaluation import Evaluator
from src.utils import print_header, print_section, Timer, check_dependencies, set_random_seed


def step_extract_features():
    """
    Étape 1: Extraction des caractéristiques audio.
    """
    print_header("ÉTAPE 1: EXTRACTION DES CARACTÉRISTIQUES")
    
    # Charger les données
    print_section("Chargement des données")
    loader = DataLoader()
    df = loader.scan_dataset()
    
    if len(df) == 0:
        print("❌ Aucun fichier audio trouvé!")
        print(f"   Veuillez placer les fichiers dans: {Config.DATA_RAW}")
        print("   Structure attendue: data/raw/genre/fichier.wav")
        return None
    
    loader.print_dataset_summary(df)
    
    # Extraire les features
    print_section("Extraction des features")
    extractor = FeatureExtractor()
    extractor.print_feature_summary()
    
    features_path = Config.DATA_PROCESSED / Config.FEATURES_FILE
    
    with Timer("Extraction des caractéristiques"):
        features_df = extractor.extract_features_from_dataset(
            df, 
            save_path=features_path
        )
    
    return features_df


def step_visualize(features_df):
    """
    Étape 2: Visualisation des données.
    """
    print_header("ÉTAPE 2: VISUALISATION DES DONNÉES")
    
    visualizer = Visualizer()
    
    # Distribution des genres
    print_section("Distribution des genres")
    visualizer.plot_genre_distribution(features_df, save_name="genre_distribution.png")
    
    # Visualisation de quelques features importantes
    print_section("Distribution des caractéristiques")
    important_features = [
        'spectral_centroid_mean',
        'tempo',
        'zero_crossing_rate_mean',
        'mfcc_1_mean'
    ]
    
    for feature in important_features:
        if feature in features_df.columns:
            visualizer.plot_feature_distribution(
                features_df, 
                feature,
                save_name=f"dist_{feature}.png"
            )
    
    # Matrice de corrélation
    print_section("Matrice de corrélation")
    visualizer.plot_correlation_matrix(features_df, save_name="correlation_matrix.png")
    
    # PCA
    print_section("Projection PCA")
    visualizer.plot_pca_2d(features_df, save_name="pca_projection.png")
    
    print("✅ Visualisations sauvegardées dans:", Config.REPORTS_DIR)


def step_train_models(features_df):
    """
    Étape 3: Entraînement des modèles.
    """
    print_header("ÉTAPE 3: ENTRAÎNEMENT DES MODÈLES")
    
    trainer = ModelTrainer()
    
    # Préparer les données
    print_section("Préparation des données")
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(features_df)
    
    # Entraîner tous les modèles
    print_section("Entraînement")
    with Timer("Entraînement de tous les modèles"):
        results_df = trainer.train_all_models(X_train, y_train, X_val, y_val)
    
    print("\n📊 Résultats de l'entraînement:")
    print(results_df.to_string(index=False))
    
    # Sauvegarder le meilleur modèle
    print_section("Sauvegarde du modèle")
    trainer.save_model(trainer.best_model_name)
    
    return trainer, X_test, y_test


def step_evaluate(trainer, X_test, y_test):
    """
    Étape 4: Évaluation des modèles.
    """
    print_header("ÉTAPE 4: ÉVALUATION DES MODÈLES")
    
    evaluator = Evaluator()
    visualizer = Visualizer()
    
    # Évaluer le meilleur modèle
    best_model_name = trainer.best_model_name
    
    print_section(f"Évaluation de {best_model_name}")
    y_pred = trainer.predict(best_model_name, X_test)
    
    # Afficher le rapport
    evaluator.print_evaluation(best_model_name, y_test, y_pred)
    
    # Visualisations
    print_section("Visualisation des résultats")
    
    # Matrice de confusion
    evaluator.plot_confusion_matrix(
        y_test, y_pred, 
        normalize=True,
        title=f"Matrice de Confusion - {best_model_name}",
        save_name="confusion_matrix.png"
    )
    
    # Rapport de classification
    evaluator.plot_classification_report(
        y_test, y_pred,
        title=f"Rapport de Classification - {best_model_name}",
        save_name="classification_report.png"
    )
    
    # Comparer tous les modèles
    print_section("Comparaison des modèles")
    
    all_results = {}
    for model_name in trainer.trained_models.keys():
        y_pred_model = trainer.predict(model_name, X_test)
        metrics = evaluator.calculate_metrics(y_test, y_pred_model)
        all_results[model_name] = {'metrics': metrics}
    
    comparison_df = evaluator.compare_models(all_results)
    print("\n📊 Comparaison finale des modèles:")
    print(comparison_df.to_string(index=False))
    
    # Graphique de comparaison
    evaluator.plot_model_comparison(all_results, save_name="model_comparison.png")
    
    print("\n✅ Évaluation terminée!")
    print(f"   Résultats sauvegardés dans: {Config.REPORTS_DIR}")


def main():
    """
    Fonction principale - Pipeline complet.
    """
    # Parser les arguments
    parser = argparse.ArgumentParser(
        description="Classification des Genres Musicaux",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
    python main.py                    # Pipeline complet
    python main.py --step extract     # Extraction des features uniquement
    python main.py --step train       # Entraînement uniquement
    python main.py --step evaluate    # Évaluation uniquement
        """
    )
    
    parser.add_argument(
        '--step', 
        type=str, 
        choices=['extract', 'train', 'evaluate', 'all'],
        default='all',
        help="Étape à exécuter (défaut: all)"
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="Graine aléatoire (défaut: 42)"
    )
    
    args = parser.parse_args()
    
    # Afficher le header
    print_header("🎵 CLASSIFICATION DES GENRES MUSICAUX 🎵", "=", 60)
    
    # Vérifier les dépendances
    print_section("Vérification des dépendances")
    deps = check_dependencies()
    
    missing = [pkg for pkg, installed in deps.items() if not installed]
    if missing:
        print(f"\n⚠️  Packages manquants: {missing}")
        print("   Installez-les avec: pip install -r requirements.txt")
        return
    
    # Créer les répertoires
    Config.create_directories()
    
    # Fixer la graine aléatoire
    set_random_seed(args.seed)
    
    # Exécuter les étapes
    if args.step in ['extract', 'all']:
        features_df = step_extract_features()
        if features_df is None:
            return
    else:
        # Charger les features existantes
        features_path = Config.DATA_PROCESSED / Config.FEATURES_FILE
        if features_path.exists():
            import pandas as pd
            features_df = pd.read_csv(features_path)
            print(f"✅ Features chargées depuis: {features_path}")
        else:
            print(f"❌ Fichier features non trouvé: {features_path}")
            print("   Exécutez d'abord: python main.py --step extract")
            return
    
    if args.step in ['train', 'all']:
        if args.step == 'all':
            step_visualize(features_df)
        
        trainer, X_test, y_test = step_train_models(features_df)
    
    if args.step in ['evaluate', 'all']:
        if args.step == 'evaluate':
            # Charger le modèle
            trainer = ModelTrainer()
            X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(features_df)
            
            # Chercher les modèles sauvegardés
            model_files = list(Config.MODELS_DIR.glob("*.joblib"))
            if model_files:
                for model_file in model_files:
                    trainer.load_model(model_file)
                trainer.best_model_name = list(trainer.trained_models.keys())[0]
            else:
                print("❌ Aucun modèle sauvegardé trouvé.")
                print("   Exécutez d'abord: python main.py --step train")
                return
        
        step_evaluate(trainer, X_test, y_test)
    
    # Résumé final
    print_header("✅ PIPELINE TERMINÉ AVEC SUCCÈS ✅", "=", 60)
    print(f"""
📁 Fichiers générés:
    - Features: {Config.DATA_PROCESSED / Config.FEATURES_FILE}
    - Modèles : {Config.MODELS_DIR}
    - Figures : {Config.REPORTS_DIR}

🚀 Prochaines étapes:
    1. Examiner les visualisations dans {Config.REPORTS_DIR}
    2. Analyser les erreurs de classification
    3. Optimiser les hyperparamètres si nécessaire
    4. Préparer le rapport et la présentation
    """)


if __name__ == "__main__":
    main()
