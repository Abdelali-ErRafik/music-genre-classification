# Classification des Genres Musicaux

Projet de classification automatique de genres musicaux a partir de fichiers audio.

## Dataset

On utilise le dataset **GTZAN** :
- 1000 fichiers audio (.wav)
- 10 genres : blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock
- 100 fichiers par genre, 30 secondes chacun

## Comment ca marche

1. On charge les fichiers audio avec `librosa`
2. On extrait des features : MFCC, spectral centroid, zero crossing rate, tempo, chroma, etc.
3. On normalise les features avec `StandardScaler`
4. On entraine 5 modeles de classification :
   - KNN (K plus proches voisins)
   - SVM (Support Vector Machine)
   - Random Forest
   - Gradient Boosting
   - MLP (Reseau de neurones)
5. On evalue et compare les resultats

## Structure du projet

```
data/raw/              -> fichiers audio (.wav) par genre
data/processed/        -> features extraites (features.csv)
models/                -> modele sauvegarde (svm.joblib)
notebooks/             -> notebook de presentation
src/                   -> code source (config, extraction, modeles, evaluation)
templates/             -> page web
app.py                 -> application web Flask
main.py                -> pipeline complet
predict.py             -> prediction en ligne de commande
```

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

```bash
# pipeline complet (extraction + entrainement + evaluation)
python main.py

# predire le genre d'un fichier audio
python predict.py "data/raw/jazz/jazz.00000.wav"

# lancer l'application web
python app.py
# puis ouvrir http://localhost:5000
```

## Technologies

- Python 3
- librosa (traitement audio)
- scikit-learn (machine learning)
- matplotlib / seaborn (visualisation)
- Flask (application web)
