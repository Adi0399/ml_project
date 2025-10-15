from pathlib import Path
import os

# Data/model directories are configurable via env vars.
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))

# Random seed for reproducibility
SEED = int(os.getenv("SEED", "42"))
