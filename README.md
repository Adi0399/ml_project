# ml_project

Reproducible ML scaffold to migrate a Jupyter notebook into a package + CI.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -e .
# Put a tiny sample in data/raw (e.g., CSVs) or set DATA_DIR to your dataset folder
python scripts/train.py
```

- Place your original notebook in `notebooks/`.
- Keep notebook and Python synced with **jupytext** (optional).
- Save trained models in `models/` (tracked with Git LFS).
