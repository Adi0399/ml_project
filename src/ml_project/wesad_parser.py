from pathlib import Path
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd
import pickle

MAP_TO_3 = {1: "NEUTRAL", 2: "STRESS", 3: "NEUTRAL", 4: "MEDITATION"}
CLASS_ID = {"NEUTRAL": 0, "STRESS": 1, "MEDITATION": 2}

def _safe_read_pkl(p: Path):
    with open(p, "rb") as f:
        return pickle.load(f, encoding="latin1")

def _pick(d: dict, keys: List[str]):
    """Return first non-None value without triggering array truth checks."""
    for k in keys:
        if k in d:
            v = d[k]
            if v is not None:
                return v
    return None

def _resample_to_4hz(series: np.ndarray, src_hz: float) -> np.ndarray:
    if series is None:
        return np.array([])
    series = np.asarray(series).squeeze()
    src_len = series.shape[0]
    if src_len == 0:
        return series
    if src_hz == 4.0:
        return series
    dst_len = int(round(src_len * (4.0 / float(src_hz))))
    x = np.linspace(0, 1, src_len)
    xi = np.linspace(0, 1, max(dst_len, 1))
    return np.interp(xi, x, series)

def _window(a: np.ndarray, win: int, step: int) -> List[np.ndarray]:
    out = []
    i = 0
    n = len(a)
    while i + win <= n:
        out.append(a[i:i+win])
        i += step
    return out

def _feat_1d(x: np.ndarray) -> Dict[str, float]:
    if x.size == 0:
        return {"mean": 0, "std": 0, "min": 0, "max": 0, "iqr": 0, "slope": 0}
    q75, q25 = np.percentile(x, [75, 25])
    iqr = float(q75 - q25)
    t = np.arange(len(x))
    try:
        slope = float(np.polyfit(t, x, 1)[0])
    except Exception:
        slope = 0.0
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "iqr": iqr,
        "slope": slope,
    }

def parse_subject_pkl(pkl_path: Path, win_sec: int = 60, step_sec: int = 30) -> pd.DataFrame:
    obj = _safe_read_pkl(pkl_path)

    # labels can live under varying keys
    label = None
    for k in ["label", "y", "labels", "y_wrist", "label_wrist"]:
        if k in obj:
            label = obj[k]
            break

    signals = obj.get("signal", {})
    wrist = signals.get("wrist", {})

    # pick channels safely (no boolean 'or' on arrays)
    eda = _pick(wrist, ["EDA","eda","Eda"])
    temp = _pick(wrist, ["TEMP","temp","Temp"])
    bvp = _pick(wrist, ["BVP","bvp","Bvp"])
    acc = _pick(wrist, ["ACC","acc","Acc"])

    # sampling rates (defaults typical for E4)
    fs = obj.get("fs", {})
    eda_hz = float(fs.get("EDA", 4.0))
    temp_hz = float(fs.get("TEMP", 4.0))
    bvp_hz = float(fs.get("BVP", 64.0))
    acc_hz = float(fs.get("ACC", 32.0))

    # acc magnitude
    acc_mag = np.array([])
    if acc is not None:
        acc = np.asarray(acc)
        if acc.ndim == 2 and acc.shape[1] >= 3:
            acc_mag = np.linalg.norm(acc[:, :3], axis=1)
        else:
            acc_mag = np.asarray(acc).squeeze()

    # resample to 4 Hz
    eda4  = _resample_to_4hz(eda,  eda_hz)
    temp4 = _resample_to_4hz(temp, temp_hz)
    bvp4  = _resample_to_4hz(bvp,  bvp_hz)
    acc4  = _resample_to_4hz(acc_mag, acc_hz)

    L = max([len(x) for x in (eda4,temp4,bvp4,acc4) if x is not None] + [0])

    def pad(a):
        a = np.asarray(a) if a is not None else np.array([])
        if a.size == 0:
            return np.zeros(L)
        if a.size < L:
            z = np.zeros(L)
            z[:a.size] = a
            return z
        return a[:L]

    eda4, temp4, bvp4, acc4 = pad(eda4), pad(temp4), pad(bvp4), pad(acc4)

    # labels align
    if label is None:
        lab = np.zeros(L, dtype=int)
    else:
        lab = np.asarray(label).squeeze()
        if lab.size != L and lab.size > 0:
            idx = np.linspace(0, lab.size - 1, L).round().astype(int)
            lab = lab[idx]
        elif lab.size == 0:
            lab = np.zeros(L, dtype=int)

    # window at 4 Hz
    win = win_sec * 4
    step = step_sec * 4
    eda_ws  = _window(eda4,  win, step)
    temp_ws = _window(temp4, win, step)
    bvp_ws  = _window(bvp4,  win, step)
    acc_ws  = _window(acc4,  win, step)
    lab_ws  = _window(lab,   win, step)
    n = min(len(eda_ws), len(temp_ws), len(bvp_ws), len(acc_ws), len(lab_ws))

    rows = []
    subj = pkl_path.stem.split("_")[0]  # e.g., S5
    for i in range(n):
        feats = {}
        for pref, arr in (("eda", eda_ws[i]), ("temp", temp_ws[i]), ("bvp", bvp_ws[i]), ("acc", acc_ws[i])):
            f = _feat_1d(arr)
            for k, v in f.items():
                feats[f"{pref}_{k}"] = v

        lw = lab_ws[i].astype(int)
        if lw.size == 0:
            lbl_name = "NEUTRAL"
        else:
            vals, counts = np.unique(lw, return_counts=True)
            if vals.size:
                nz = np.where(vals != 0)[0]
                j = (nz[np.argmax(counts[nz])] if nz.size else np.argmax(counts))
                raw = int(vals[j])
            else:
                raw = 1
            lbl_name = MAP_TO_3.get(raw, "NEUTRAL")

        feats["label_name"] = lbl_name
        feats["label_id"] = CLASS_ID[lbl_name]
        feats["subject"] = subj
        rows.append(feats)

    return pd.DataFrame(rows)
