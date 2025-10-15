from pathlib import Path
from typing import Optional, List
import pandas as pd
from .config import DATA_DIR
from .wesad_parser import parse_subject_pkl

def load_from_folder(folder: Optional[Path] = None) -> pd.DataFrame:
    """
    Aggregate all subjects under DATA_DIR by reading S*/S*.pkl,
    building windowed features + 3-class labels.
    """
    base = Path(folder) if folder else Path(DATA_DIR)
    if not base.exists():
        raise FileNotFoundError(f"DATA_DIR not found: {base.resolve()}")

    pkl_paths: List[Path] = []
    for p in base.rglob("S*/S*.pkl"):
        pkl_paths.append(p)
    if not pkl_paths:
        raise FileNotFoundError(f"No subject pickles found under {base.resolve()}")

    dfs = []
    for p in sorted(pkl_paths):
        try:
            df = parse_subject_pkl(p)
            dfs.append(df)
        except Exception as e:
            print(f"[warn] failed parsing {p}: {e}")

    if not dfs:
        raise RuntimeError("No subject could be parsed.")
    return pd.concat(dfs, ignore_index=True, sort=False)
