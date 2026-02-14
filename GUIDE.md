# Music Genre Classification - Guide Complet (A to Z)

## Table des Matières

1. [Vue d'ensemble du projet](#1-vue-densemble-du-projet)
2. [Architecture du projet](#2-architecture-du-projet)
3. [Le Dataset (GTZAN)](#3-le-dataset-gtzan)
4. [Extraction des caractéristiques audio](#4-extraction-des-caractéristiques-audio)
5. [Prétraitement des données](#5-prétraitement-des-données)
6. [Entraînement des modèles](#6-entraînement-des-modèles)
7. [Évaluation et métriques](#7-évaluation-et-métriques)
8. [Prédiction sur un nouveau fichier](#8-prédiction-sur-un-nouveau-fichier)
9. [Application Web (Frontend)](#9-application-web-frontend)
10. [Comment tout exécuter](#10-comment-tout-exécuter)

---

## 1. Vue d'ensemble du projet

Ce projet utilise le **Machine Learning** pour classifier automatiquement la musique en **10 genres** :

| Genre | Description |
|-------|-------------|
| Blues | Musique afro-américaine, guitare expressive |
| Classical | Musique orchestrale, instruments acoustiques |
| Country | Musique rurale américaine, guitare et voix |
| Disco | Musique de danse des années 70-80 |
| Hip-Hop | Rap, beats, rythmes urbains |
| Jazz | Improvisation, swing, instruments cuivre |
| Metal | Guitares distordues, batterie rapide |
| Pop | Musique populaire, mélodies accrocheuses |
| Reggae | Rythmes jamaïcains, offbeat |
| Rock | Guitares électriques, batterie, énergie |

**Le principe** : on ne donne pas le fichier audio brut au modèle. On extrait d'abord des **caractéristiques numériques** (features) du son, puis le modèle apprend à associer ces chiffres à un genre.

```
Fichier Audio (.wav)
        |
        v
Extraction de 102 features numériques
        |
        v
Modèle de Machine Learning (SVM)
        |
        v
Genre prédit : "rock", "jazz", etc.
```

---

## 2. Architecture du projet

```
music-genre-classification/
│
├── data/
│   ├── raw/                    # 1000 fichiers audio (100 par genre)
│   │   ├── blues/
│   │   ├── classical/
│   │   ├── country/
│   │   ├── disco/
│   │   ├── hiphop/
│   │   ├── jazz/
│   │   ├── metal/
│   │   ├── pop/
│   │   ├── reggae/
│   │   └── rock/
│   └── processed/
│       └── features.csv        # Les 102 features extraites pour les 1000 fichiers
│
├── src/                        # Code source (modules Python)
│   ├── config.py               # Configuration centralisée
│   ├── data_loader.py          # Chargement des fichiers audio
│   ├── feature_extraction.py   # Extraction des caractéristiques
│   ├── models.py               # Définition et entraînement des modèles
│   ├── evaluation.py           # Évaluation des performances
│   ├── visualization.py        # Graphiques et visualisations
│   └── utils.py                # Fonctions utilitaires
│
├── models/
│   └── svm.joblib              # Modèle SVM entraîné et sauvegardé
│
├── reports/                    # Graphiques générés (confusion matrix, etc.)
├── templates/
│   └── index.html              # Interface web (frontend)
│
├── main.py                     # Pipeline complet (extract → train → evaluate)
├── predict.py                  # Prédiction en ligne de commande
├── app.py                      # Application web Flask
└── requirements.txt            # Dépendances Python
```

### Rôle de chaque module

| Module | Rôle |
|--------|------|
| `config.py` | Tous les paramètres du projet (sample rate, durée, chemins, hyperparamètres) |
| `data_loader.py` | Scanne le dossier `data/raw/`, charge les fichiers audio avec librosa |
| `feature_extraction.py` | Extrait les 102 caractéristiques de chaque fichier audio |
| `models.py` | Définit 9 modèles ML, entraîne, sauvegarde/charge avec joblib |
| `evaluation.py` | Calcule accuracy, precision, recall, F1, matrice de confusion |
| `visualization.py` | Génère les graphiques (spectrogrammes, PCA, comparaisons) |
| `utils.py` | Timer, logging, vérification des dépendances |

---

## 3. Le Dataset (GTZAN)

Le projet utilise le **GTZAN Genre Collection**, un dataset de référence en MIR (Music Information Retrieval).

- **1000 fichiers audio** au total
- **100 fichiers par genre** (10 genres)
- **Format** : WAV, mono, 22050 Hz
- **Durée** : 30 secondes chacun

### Chargement des données (`data_loader.py`)

La classe `DataLoader` scanne le dossier `data/raw/` et crée un DataFrame :

```python
loader = DataLoader()
df = loader.scan_dataset()
# df contient: filepath, filename, genre, extension
# Exemple: data/raw/rock/rock.00042.wav | rock.00042.wav | rock | .wav
```

---

## 4. Extraction des caractéristiques audio

C'est l'étape la plus importante. On transforme un signal audio en **102 nombres** qui décrivent le son.

### Pourquoi extraire des features ?

Un fichier audio de 30 secondes à 22050 Hz = **661 500 échantillons**. C'est beaucoup trop pour un modèle ML classique. On résume donc le son en 102 chiffres significatifs.

### Les 102 caractéristiques extraites

La classe `FeatureExtractor` extrait les features suivantes :

#### a) MFCC - Mel-Frequency Cepstral Coefficients (40 features)

Les MFCC sont les features les plus importantes en audio ML. Ils représentent la **forme spectrale** du son, similaire à comment l'oreille humaine perçoit les fréquences.

- 20 coefficients MFCC × 2 (moyenne + écart-type) = **40 features**
- `mfcc_1_mean`, `mfcc_1_std`, ..., `mfcc_20_mean`, `mfcc_20_std`

```python
mfcc = librosa.feature.mfcc(y=signal, sr=22050, n_mfcc=20)
```

#### b) Caractéristiques spectrales (22 features)

| Feature | Nombre | Ce qu'elle mesure |
|---------|--------|-------------------|
| Spectral Centroid | 2 | "Centre de gravité" du spectre → brillance du son |
| Spectral Bandwidth | 2 | Largeur du spectre → richesse harmonique |
| Spectral Rolloff | 2 | Fréquence sous laquelle se trouve 85% de l'énergie |
| Spectral Contrast | 14 | Différence pics/vallées dans 7 bandes de fréquence |
| Zero Crossing Rate | 2 | Combien de fois le signal passe par zéro → bruit vs harmonie |

#### c) Énergie RMS (2 features)

Mesure le **volume** moyen du signal.

- `rms_mean`, `rms_std`

#### d) Tempo (1 feature)

Le **BPM** (battements par minute) estimé.

- `tempo` → ex: 120.0 BPM

#### e) Chroma Features (24 features)

Distribution de l'énergie parmi les **12 notes de musique** (Do, Do#, Ré, ..., Si).

- 12 notes × 2 (moyenne + écart-type) = **24 features**
- `chroma_C_mean`, `chroma_C_std`, ..., `chroma_B_mean`, `chroma_B_std`

#### f) Mel-Spectrogram Stats (4 features)

Statistiques globales du mel-spectrogramme.

- `mel_spec_mean`, `mel_spec_std`, `mel_spec_max`, `mel_spec_min`

#### g) Résumé

```
MFCC               : 40 features
Spectral Centroid   :  2 features
Spectral Bandwidth  :  2 features
Spectral Rolloff    :  2 features
Spectral Contrast   : 14 features
Zero Crossing Rate  :  2 features
RMS Energy          :  2 features
Tempo               :  1 feature
Chroma              : 24 features
Mel Spectrogram     :  4 features
─────────────────────────────────
TOTAL               : 93 features
```

> Note : le nombre exact peut varier légèrement selon la version de librosa (spectral contrast produit 7 ou 8 bandes).

### Résultat : features.csv

Après extraction, on obtient un fichier CSV :

```
filename          | genre | mfcc_1_mean | mfcc_1_std | ... | mel_spec_min
blues.00000.wav   | blues | -113.5      | 71.2       | ... | -80.0
rock.00042.wav    | rock  | -180.2      | 65.8       | ... | -75.3
...               | ...   | ...         | ...        | ... | ...
```

**1000 lignes × 102+ colonnes**

---

## 5. Prétraitement des données

Avant l'entraînement, les données passent par 3 étapes (`models.py` → `prepare_data`) :

### a) Encodage des labels

Les genres texte sont convertis en nombres :

```
blues=0, classical=1, country=2, disco=3, hiphop=4,
jazz=5, metal=6, pop=7, reggae=8, rock=9
```

Fait avec `LabelEncoder` de scikit-learn.

### b) Division des données

```
1000 fichiers
    ├── 70% → Entraînement (700 fichiers) → Le modèle apprend
    ├── 10% → Validation    (100 fichiers) → Choix du meilleur modèle
    └── 20% → Test          (200 fichiers) → Évaluation finale
```

La division est **stratifiée** : chaque split garde la même proportion de genres.

### c) Normalisation (StandardScaler)

Les features ont des échelles très différentes (tempo ~120, MFCC ~-200). On les normalise :

```
X_normalized = (X - moyenne) / écart_type
```

Après normalisation, chaque feature a une moyenne de 0 et un écart-type de 1. C'est essentiel pour le SVM et le KNN.

---

## 6. Entraînement des modèles

Le projet entraîne **9 modèles** différents et garde le meilleur :

| Modèle | Type | Comment il fonctionne |
|--------|------|-----------------------|
| **SVM** | Support Vector Machine | Trouve l'hyperplan optimal séparant les genres dans un espace à haute dimension |
| **KNN** | K-Nearest Neighbors | Classe un son selon les K sons les plus similaires |
| **Random Forest** | Ensemble d'arbres | Vote majoritaire de 200 arbres de décision |
| **Gradient Boosting** | Ensemble séquentiel | Arbres qui corrigent les erreurs des précédents |
| **MLP** | Réseau de neurones | 2 couches cachées (100 et 50 neurones) |
| **Logistic Regression** | Linéaire | Régression logistique multi-classes |
| **Decision Tree** | Arbre simple | Un seul arbre de décision |
| **Naive Bayes** | Probabiliste | Suppose l'indépendance des features |
| **AdaBoost** | Ensemble adaptatif | Pondère les exemples difficiles |

### Processus d'entraînement

```python
trainer = ModelTrainer()

# 1. Préparer les données
X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(features_df)

# 2. Entraîner les 9 modèles
results = trainer.train_all_models(X_train, y_train, X_val, y_val)

# 3. Le meilleur modèle est sélectionné automatiquement (meilleure accuracy sur validation)
print(trainer.best_model_name)  # → "SVM"

# 4. Sauvegarder le meilleur modèle
trainer.save_model(trainer.best_model_name)
# → Sauvegarde : models/svm.joblib (contient le modèle + scaler + label_encoder)
```

### Pourquoi le SVM gagne souvent ?

Le SVM avec noyau RBF (`kernel='rbf'`) est très efficace quand :
- Les données sont normalisées
- Le nombre de features (~100) est modéré
- Les classes ne sont pas parfaitement linéairement séparables

---

## 7. Évaluation et métriques

### Métriques utilisées

| Métrique | Signification |
|----------|---------------|
| **Accuracy** | % de prédictions correctes sur toutes les classes |
| **Precision** | Parmi les prédictions "rock", combien sont vraiment rock ? |
| **Recall** | Parmi tous les vrais rock, combien ont été trouvés ? |
| **F1-Score** | Moyenne harmonique de precision et recall |

### Matrice de confusion

Tableau montrant pour chaque genre réel, quel genre a été prédit :

```
                Prédit →
Réel ↓     blues  classical  rock  ...
blues        18      0        2
classical     0     20        0
rock          1      0       17
```

- **Diagonale** = prédictions correctes
- **Hors diagonale** = erreurs (confusions entre genres)

### Paires souvent confondues

Certains genres se ressemblent acoustiquement :
- Rock ↔ Metal (guitares électriques)
- Rock ↔ Country (structures similaires)
- Disco ↔ Pop (rythmes dansants)

---

## 8. Prédiction sur un nouveau fichier

### En ligne de commande (`predict.py`)

```bash
python predict.py "chemin/vers/ma_chanson.mp3"
```

### Ce qui se passe en interne

```
1. Charger le modèle sauvegardé (models/svm.joblib)
   → Contient : modèle SVM + StandardScaler + LabelEncoder

2. Charger le fichier audio avec librosa
   → Signal numérique (array de 661 500 valeurs)

3. Extraire les 102 features
   → Un vecteur de 102 nombres

4. Normaliser avec le même scaler que l'entraînement
   → Même échelle que les données d'entraînement

5. Prédire avec le SVM
   → Genre prédit + probabilités pour chaque genre

6. Afficher le résultat
   → "ROCK - Confiance: 72.3%"
```

---

## 9. Application Web (Frontend)

### Architecture

```
┌─────────────────────────────────┐
│         Navigateur Web          │
│   (http://localhost:5000)       │
│                                 │
│   ┌─────────────────────────┐   │
│   │   Interface HTML/CSS    │   │
│   │   - Drag & Drop zone    │   │
│   │   - Bouton "Classifier" │   │
│   │   - Barres de résultat  │   │
│   └────────────┬────────────┘   │
└────────────────┼────────────────┘
                 │ POST /predict
                 │ (fichier audio)
                 ▼
┌─────────────────────────────────┐
│       Serveur Flask (app.py)    │
│                                 │
│   1. Recevoir le fichier        │
│   2. Extraire 102 features      │
│   3. Normaliser (scaler)        │
│   4. Prédire (SVM)              │
│   5. Retourner JSON             │
└─────────────────────────────────┘
                 │
                 ▼
         Réponse JSON :
         {
           "genre": "rock",
           "confidence": 72.3,
           "probabilities": [
             {"genre": "rock", "probability": 72.3},
             {"genre": "metal", "probability": 12.1},
             ...
           ]
         }
```

### Fichiers concernés

**`app.py`** - Le serveur Flask :
- Route `/` → Sert la page HTML
- Route `/predict` (POST) → Reçoit le fichier, fait la prédiction, retourne du JSON
- Le modèle est chargé une seule fois au démarrage du serveur

**`templates/index.html`** - L'interface web :
- Zone de drag & drop pour uploader un fichier
- JavaScript `fetch()` envoie le fichier au serveur
- Affiche le résultat avec des barres de probabilité colorées

### Comment ça marche étape par étape

1. L'utilisateur ouvre `http://localhost:5000`
2. Flask sert le fichier `templates/index.html`
3. L'utilisateur glisse un fichier audio dans la zone
4. Clic sur "Classifier ce morceau"
5. JavaScript envoie le fichier via `POST /predict` (FormData)
6. Flask reçoit le fichier, le sauvegarde temporairement
7. `FeatureExtractor` extrait les 102 features avec librosa
8. Le `StandardScaler` normalise les features
9. Le modèle SVM prédit le genre + probabilités
10. Flask retourne un JSON avec le résultat
11. JavaScript affiche le genre et les barres de probabilité
12. Le fichier temporaire est supprimé

---

## 10. Comment tout exécuter

### Prérequis

```bash
# Installer les dépendances
pip install -r requirements.txt
```

### Option A : Pipeline complet (entraînement)

```bash
# Exécuter tout : extraction → visualisation → entraînement → évaluation
python main.py

# Ou étape par étape :
python main.py --step extract    # Extraire les features
python main.py --step train      # Entraîner les modèles
python main.py --step evaluate   # Évaluer les performances
```

### Option B : Prédire un fichier (ligne de commande)

```bash
python predict.py "chemin/vers/chanson.mp3"
```

### Option C : Application web

```bash
python app.py
# Ouvrir http://localhost:5000 dans le navigateur
```

---

## Résumé du flux complet

```
    DONNÉES                    ENTRAÎNEMENT                 UTILISATION
    ────────                   ────────────                 ───────────

  1000 fichiers WAV            features.csv
  (data/raw/)                  (1000 × 102)
       │                            │
       ▼                            ▼
  FeatureExtractor ──────►  ModelTrainer.prepare_data()
  (librosa)                         │
                                    ├── Encode labels (0-9)
                                    ├── Split 70/10/20
                                    └── Normalize (scaler)
                                         │
                                         ▼
                              Train 9 modèles
                                         │
                                         ▼
                              Best model = SVM
                                         │
                                         ▼
                              Save → models/svm.joblib
                                    (model + scaler + encoder)
                                         │
                         ┌───────────────┼───────────────┐
                         ▼               ▼               ▼
                    predict.py       app.py         notebooks/
                    (terminal)       (web)          (exploration)
                         │               │
                         ▼               ▼
                   "Genre: rock"    Interface web
                                   avec barres
                                   de probabilité
```
