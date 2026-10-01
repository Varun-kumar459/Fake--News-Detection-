"""Central configuration: paths and model settings."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

# Option A: real dataset (Kaggle "Fake and Real News Dataset" - ISOT)
FAKE_CSV = DATA_DIR / "Fake.csv"
TRUE_CSV = DATA_DIR / "True.csv"

# Option B: single CSV with columns: text,label  (label: 0 = Real, 1 = Fake)
SAMPLE_CSV = DATA_DIR / "sample_news.csv"

MODEL_PATH = MODEL_DIR / "fake_news_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

LABELS = {0: "Real", 1: "Fake"}
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_FEATURES = 20000
