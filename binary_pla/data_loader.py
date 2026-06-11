"""
Data loading for the binary per-device PLA pipeline.

One binary dataset per authorized device per trial:
  positive (1) = target device, negative (0) = other authorized devices.
  Rogue devices are held out entirely.

repr_type "fv"  → 505-dim feature vectors (fv_windowed.h5 / fv_original.h5)
repr_type "dgt" → 150×150 DGT matrices   (dgt_windowed.h5 / dgt_original.h5)
"""

import os
import sys
import functools
import numpy as np
import h5py
import torch
from torch.utils.data import TensorDataset, DataLoader
import warnings

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from binary_pla.config import (
    FV_ORIGINAL_PATH, FV_WINDOWED_PATH,
    DGT_ORIGINAL_PATH, DGT_WINDOWED_PATH,
    N_WIN, TRIALS,
)

# device name → 0-indexed label used in HDF5 files
def device_label(name: str) -> int:
    """'device3' → 2,  'device12' → 11"""
    return int(name.replace("device", "")) - 1


@functools.lru_cache(maxsize=4)
def _load_cache(repr_type: str, use_windowed: bool):
    if repr_type == "fv":
        path = FV_WINDOWED_PATH if use_windowed else FV_ORIGINAL_PATH
    elif repr_type == "dgt":
        path = DGT_WINDOWED_PATH if use_windowed else DGT_ORIGINAL_PATH
    else:
        raise ValueError(f"Unknown repr_type '{repr_type}'. Choose 'fv' or 'dgt'.")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cache not found: {path}\n"
            "Run the preprocessing pipeline to generate it first."
        )

    with h5py.File(path, "r") as f:
        X = f["data"][:].astype(np.float32)
        y = f["labels"][:].astype(np.int64)
        # Read windows_per_transient attribute if stored by the preprocessor.
        # Falls back to _N_WINDOWS so FV caches (which don't store it) still work.
        n_win = int(f["data"].attrs.get("windows_per_transient", N_WIN))
    return X, y, n_win


_VALID_COMBOS = {
    "fv":  {"mlp_fv"},
    "dgt": {"gru_dgt", "bigru_dgt"},
}

def _validate_model_name(model_name: str, repr_type: str = None):
    if model_name not in ("mlp_fv", "gru_dgt", "bigru_dgt"):
        raise ValueError(f"Unknown model_name '{model_name}'. Use 'mlp_fv', 'gru_dgt', or 'bigru_dgt'.")
    if repr_type is not None and model_name not in _VALID_COMBOS.get(repr_type, set()):
        raise ValueError(
            f"Incompatible combination: repr_type='{repr_type}' with model_name='{model_name}'. "
            f"Valid pairs: fv→mlp_fv, dgt→gru_dgt/bigru_dgt."
        )


def _make_loader(X, y, batch_size, shuffle):
    ds = TensorDataset(
        torch.from_numpy(np.ascontiguousarray(X)),
        torch.from_numpy(y.astype(np.int64)),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _transient_groups(y, authorized_labels, n_win):
    """
    Assign each sample a unique transient group ID.

    The encoding i*10_000 + k//n_win is identical for both baseline
    (n_win=1) and windowed (n_win=10) caches, so group IDs from a baseline
    split can be used directly to filter the windowed cache.
    """
    groups = np.zeros(len(y), dtype=np.int64)
    for i, lbl in enumerate(authorized_labels):
        pos = np.where(y == lbl)[0]
        groups[pos] = i * 10_000 + np.arange(len(pos)) // n_win
    return groups


def _split_per_device(y_auth_b, groups_b, authorized_labels, val_frac, test_frac, seed):
    """
    Split per device (not pooled) to guarantee ≥1 test and ≥1 val transient
    per device. Pooled splits can leave devices with few transients (e.g. 6)
    with zero positive test samples, producing NaN auth_tvr.
    """
    rng = np.random.default_rng(seed)
    tr_idx, va_idx, te_idx = [], [], []

    for lbl in authorized_labels:
        dev_pos       = np.where(y_auth_b == lbl)[0]
        unique_groups = np.unique(groups_b[dev_pos])
        n             = len(unique_groups)
        shuffled      = rng.permutation(unique_groups)

        # Desired counts — guarantee at least 1 transient per split.
        n_test  = max(1, round(n * test_frac))
        n_val   = max(1, round((n - n_test) * val_frac / (1.0 - test_frac)))
        n_train = n - n_test - n_val

        if n_train < 1:
            # Fewer than 3 transients: keep test=1, val=1, train=0 for this device.
            n_train = 0
            warnings.warn(
                f"Device label {lbl} has only {n} transients — "
                f"cannot guarantee train≥1 "
                f"(test={n_test}, val={n_val}, train=0)."
            )

        te_set = set(shuffled[:n_test])
        va_set = set(shuffled[n_test:n_test + n_val])

        for pos in dev_pos:
            g = groups_b[pos]
            if g in te_set:
                te_idx.append(pos)
            elif g in va_set:
                va_idx.append(pos)
            else:
                tr_idx.append(pos)

    return (
        np.array(tr_idx, dtype=np.int64),
        np.array(va_idx, dtype=np.int64),
        np.array(te_idx, dtype=np.int64),
    )


def load_trial_data(
    trial_name:   str,
    repr_type:    str   = "fv",
    model_name:   str   = "mlp_fv",
    val_frac:     float = 0.15,
    test_frac:    float = 0.15,
    batch_size:   int   = 64,
    seed:         int   = 42,
    extra_arrays: bool  = False,
):
    """
    Three-way transient-aware split (70/15/15).

    Train uses windowed data; val and test use one fingerprint per transient
    (baseline) so TVR estimates reflect genuine per-transient performance.
    Z-score is fitted on windowed train only. The split is done per device
    to guarantee ≥1 test and ≥1 val transient for each authorized device.

    Set extra_arrays=True to also return raw normalised numpy arrays
    (needed by notebooks doing ROC or PCA analysis).
    """
    if trial_name not in TRIALS:
        raise ValueError(f"Unknown trial '{trial_name}'. Choose from {list(TRIALS)}.")
    _validate_model_name(model_name, repr_type)

    authorized_names  = TRIALS[trial_name]["authorized"]
    rogue_names       = TRIALS[trial_name]["rogue"]
    authorized_labels = [device_label(n) for n in authorized_names]
    rogue_labels      = [device_label(n) for n in rogue_names]

    # Load both caches — baseline for splitting/val/test, windowed for training.
    X_base, y_base, _     = _load_cache(repr_type, use_windowed=False)
    X_win,  y_win,  n_win = _load_cache(repr_type, use_windowed=True)

    # Separate authorized and rogue subsets from the baseline cache.
    auth_mask_b  = np.isin(y_base, authorized_labels)
    rogue_mask_b = np.isin(y_base, rogue_labels)
    X_auth_b, y_auth_b   = X_base[auth_mask_b],  y_base[auth_mask_b]
    X_rogue_b, y_rogue_b = X_base[rogue_mask_b], y_base[rogue_mask_b]

    # Separate authorized subset from the windowed cache.
    auth_mask_w = np.isin(y_win, authorized_labels)
    X_auth_w, y_auth_w = X_win[auth_mask_w], y_win[auth_mask_w]

    # Build transient group IDs using the same encoding for both caches.
    # baseline: n_per_transient=1  → each sample is its own group
    # windowed: n_per_transient=n_win → every n_win consecutive samples share a group
    groups_b = _transient_groups(y_auth_b, authorized_labels, 1)
    groups_w = _transient_groups(y_auth_w, authorized_labels, n_win)

    # Split per device — guarantees ≥1 test and ≥1 val transient per device.
    tr_b_idx, va_b_idx, te_b_idx = _split_per_device(
        y_auth_b, groups_b, authorized_labels, val_frac, test_frac, seed
    )

    X_va, y_va = X_auth_b[va_b_idx], y_auth_b[va_b_idx]
    X_te, y_te = X_auth_b[te_b_idx], y_auth_b[te_b_idx]

    # Transient group IDs assigned to train — used to filter the windowed cache.
    train_group_ids = set(groups_b[tr_b_idx])

    # Select only windowed samples whose transient belongs to the train split.
    train_win_mask = np.isin(groups_w, list(train_group_ids))
    X_tr_w, y_tr_w = X_auth_w[train_win_mask], y_auth_w[train_win_mask]

    # Fit z-score on windowed training data; apply to all partitions.
    flat_tr = X_tr_w.reshape(len(X_tr_w), -1)
    mu  = flat_tr.mean(0)
    sig = flat_tr.std(0) + 1e-8

    def _norm(X):
        f = X.reshape(len(X), -1)
        return ((f - mu) / sig).astype(np.float32).reshape(X.shape)

    X_tr_w   = _norm(X_tr_w)
    X_va     = _norm(X_va)
    X_te     = _norm(X_te)
    X_rogue_b = _norm(X_rogue_b)

    # Build per-device binary loaders.
    per_device = {}
    for dev in authorized_names:
        lbl    = device_label(dev)
        y_tr_b = (y_tr_w == lbl).astype(np.int64)
        y_va_b = (y_va   == lbl).astype(np.int64)
        y_te_b = (y_te   == lbl).astype(np.int64)

        d = {
            "train_loader": _make_loader(X_tr_w, y_tr_b, batch_size, True),
            "val_loader"  : _make_loader(X_va,   y_va_b, batch_size, False),
            "test_loader" : _make_loader(X_te,   y_te_b, batch_size, False),
            "n_train_pos" : int(y_tr_b.sum()),
            "n_val_pos"   : int(y_va_b.sum()),
            "n_test_pos"  : int(y_te_b.sum()),
        }
        if extra_arrays:
            d["X_tr_pos"] = X_tr_w[y_tr_w == lbl]
            d["X_te_pos"] = X_te[y_te == lbl]

        per_device[dev] = d

    rogue_loader = _make_loader(X_rogue_b, y_rogue_b, batch_size, False)

    n_train_tr = len(np.unique(groups_w[train_win_mask]))
    print(
        f"[load_trial_data] {trial_name} | repr={repr_type} | seed={seed}\n"
        f"  train: {len(X_tr_w)} windowed samples from {n_train_tr} transients\n"
        f"  val:   {len(X_va)} baseline samples  |  "
        f"test: {len(X_te)} baseline samples  |  "
        f"rogue: {len(X_rogue_b)} baseline samples"
    )

    result = {
        "trial_name"  : trial_name,
        "authorized"  : authorized_names,
        "rogue"       : rogue_names,
        "use_windowed": True,
        "per_device"  : per_device,
        "rogue_loader": rogue_loader,
    }
    if extra_arrays:
        result["X_rogue"] = X_rogue_b
        result["y_rogue"] = y_rogue_b
    return result

