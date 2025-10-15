from pathlib import Path
import json
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from .config import MODELS_DIR, SEED
from .data import load_from_folder
from .features import make_features

def train_eval_save(model_out: Path | None = None, target_col: str = "label_id") -> dict:
    df = load_from_folder()
    X, y = make_features(df, target_col=target_col)

    # stratify for 3-class
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        random_state=SEED
    )
    model.fit(Xtr, ytr)
    preds = model.predict(Xte)
    score = accuracy_score(yte, preds)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out = model_out or (MODELS_DIR / "best_model.joblib")
    joblib.dump(model, out)

    metrics = {"metric": "accuracy", "value": float(score)}
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics

if __name__ == "__main__":
    print(train_eval_save())
