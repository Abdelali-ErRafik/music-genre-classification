# 🎵 Classification des Genres Musicaux

## Description du Projet

Ce projet vise à construire un système capable de classifier automatiquement des pistes audio dans différents genres musicaux (pop, jazz, classique, rock, hip-hop, etc.) en analysant leurs caractéristiques acoustiques.

**Module :** Python pour l'Analyse de Données  
**Date de soutenance :** Semaine du 23 février 2026

---

## 📁 Structure du Projet

```
music-genre-classification/
│
├── data/
│   ├── raw/                    # Fichiers audio bruts (GTZAN)
│   └── processed/              # Features extraites (CSV)
│
├── notebooks/
│   ├── 01_exploration.ipynb    # Analyse exploratoire
│   ├── 02_feature_extraction.ipynb
│   ├── 03_modeling.ipynb       # Entraînement des modèles
│   └── 04_evaluation.ipynb     # Évaluation finale
│
├── src/
│   ├── __init__.py
│   ├── config.py               # Configuration du projet
│   ├── data_loader.py          # Chargement des données
│   ├── feature_extraction.py   # Extraction des caractéristiques
│   ├── visualization.py        # Fonctions de visualisation
│   ├── models.py               # Définition des modèles
│   ├── evaluation.py           # Métriques d'évaluation
│   └── utils.py                # Fonctions utilitaires
│
├── models/                     # Modèles entraînés sauvegardés
├── reports/                    # Rapport et présentation
├── tests/                      # Tests unitaires
│
├── main.py                     # Script principal
├── requirements.txt            # Dépendances
└── README.md
```

---

## 🚀 Installation

### 1. Cloner le projet
```bash
git clone <url-du-repo>
cd music-genre-classification
```

### 2. Créer un environnement virtuel
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 4. Télécharger le dataset GTZAN
Télécharger depuis : https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification

Extraire les fichiers dans `data/raw/`

---

## 📊 Dataset : GTZAN

- **1 000 fichiers audio** (30 secondes chacun)
- **10 genres :** Blues, Classical, Country, Disco, Hip-hop, Jazz, Metal, Pop, Reggae, Rock
- **Format :** WAV, 22 050 Hz, mono

---

## 🔧 Utilisation

### Exécuter le pipeline complet
```bash
python main.py
```

### Exécuter étape par étape
```bash
python main.py --step extract    # Extraction des features
python main.py --step train      # Entraînement
python main.py --step evaluate   # Évaluation
```

---

## 📈 Caractéristiques Audio Extraites

| Caractéristique | Description |
|-----------------|-------------|
| MFCC (1-20) | Coefficients cepstraux sur l'échelle de Mel |
| Spectral Centroid | Centre de gravité du spectre |
| Spectral Bandwidth | Largeur du spectre |
| Spectral Rolloff | Fréquence de coupure à 85% de l'énergie |
| Zero Crossing Rate | Taux de passages par zéro |
| Tempo | Battements par minute (BPM) |
| RMS Energy | Énergie moyenne du signal |
| Chroma Features | Distribution des 12 notes musicales |

---

## 🤖 Modèles Implémentés

- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Random Forest
- Gradient Boosting
- Réseau de Neurones (MLP)
- CNN sur Mel-Spectrogrammes (optionnel)

---

## 📝 Auteurs

- [Votre nom]
- [Noms des membres du groupe]

---

## 📚 Références

- GTZAN Dataset: http://marsyas.info/downloads/datasets.html
- Librosa Documentation: https://librosa.org/doc/
- Scikit-learn: https://scikit-learn.org/
