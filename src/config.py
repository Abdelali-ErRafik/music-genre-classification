from pathlib import Path


class Config:
    # chemins du projet
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_RAW = PROJECT_ROOT / "data" / "raw"
    DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
    MODELS_DIR = PROJECT_ROOT / "models"
    REPORTS_DIR = PROJECT_ROOT / "reports"

    # parametres audio
    SAMPLE_RATE = 22050
    DURATION = 30
    N_SAMPLES = SAMPLE_RATE * DURATION

    # parametres pour l'extraction des features
    N_MFCC = 20
    N_FFT = 2048
    HOP_LENGTH = 512
    N_MELS = 128
    N_CHROMA = 12

    # les 10 genres du dataset GTZAN
    GENRES = [
        "blues",
        "classical",
        "country",
        "disco",
        "hiphop",
        "jazz",
        "metal",
        "pop",
        "reggae",
        "rock",
    ]
    N_GENRES = len(GENRES)

    # parametres train/test
    TEST_SIZE = 0.2
    VALIDATION_SIZE = 0.1
    RANDOM_STATE = 42
    N_FOLDS = 5

    # fichiers de sortie
    FEATURES_FILE = "features.csv"
    METRICS_FILE = "model_metrics.csv"

    @classmethod
    def create_directories(cls):
        # on cree les dossiers necessaires
        for d in [cls.DATA_RAW, cls.DATA_PROCESSED, cls.MODELS_DIR, cls.REPORTS_DIR]:
            d.mkdir(parents=True, exist_ok=True)
        print("Repertoires crees.")


if __name__ == "__main__":
    Config.create_directories()
