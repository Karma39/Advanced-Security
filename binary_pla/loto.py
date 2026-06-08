"""
Leave-One-Transient-Out (LOTO) cross-validation for the binary PLA pipeline.

For each fold k, transient k is held out as the test sample and transient
(k+1)%N as the val sample. The optimal decision threshold is selected on
pooled val scores, then applied blind to test scores.

Rogues are split by device (not transient) so val_rogue and test_rogue come
from entirely different physical hardware. Folds run in parallel via joblib.
"""

import os
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import roc_curve

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from binary_pla.config import LR, WEIGHT_DECAY, DROPOUT, BATCH_SIZE, N_JOBS


def _split_rogues(y_rogue_b, rogue_labels, seed):
    """Split rogue devices into val_rogue and test_rogue halves. Returns boolean masks."""
    rng = np.random.default_rng(seed)
    labels = np.array(rogue_labels)
    shuffled = rng.permutation(labels)
    n_val = len(shuffled) // 2
    val_lbl  = set(shuffled[:n_val].tolist())
    test_lbl = set(shuffled[n_val:].tolist())
    val_mask  = np.isin(y_rogue_b, list(val_lbl))
    test_mask = np.isin(y_rogue_b, list(test_lbl))
    return val_mask, test_mask


def _prepare_fold(
    X_auth_b, y_auth_b,
    X_auth_w, y_auth_w,
    X_val_rogue_b,  y_val_rogue_b,
    X_test_rogue_b, y_test_rogue_b,
    authorized_labels,
    target_label,
    fold_idx,
    n_win,
    model_name,
):
    """Slice and normalise data for one LOTO fold. Returns serialisable numpy arrays."""
    from binary_pla.data_loader import _validate_model_name
    _validate_model_name(model_name)

    target_b_idx = np.where(y_auth_b == target_label)[0]
    n_transients  = len(target_b_idx)

    te_pos = target_b_idx[fold_idx]
    va_pos = target_b_idx[(fold_idx + 1) % n_transients]

    holdout      = {fold_idx, (fold_idx + 1) % n_transients}
    train_tr_idx = [i for i in range(n_transients) if i not in holdout]

    target_w_mask   = y_auth_w == target_label
    target_w_global = np.where(target_w_mask)[0]
    assert len(target_w_global) % n_win == 0, (
        f"Device {target_label}: {len(target_w_global)} windowed samples is not "
        f"divisible by n_win={n_win}. Some transients contributed fewer windows "
        f"than expected — the transient-to-window mapping is broken."
    )
    target_w_tr     = np.isin(np.arange(len(target_w_global)) // n_win, train_tr_idx)
    train_w_idx     = np.concatenate([
        target_w_global[target_w_tr],
        np.where(~target_w_mask)[0],
    ])
    X_tr_raw = X_auth_w[train_w_idx]
    y_tr_raw = y_auth_w[train_w_idx]

    other_val_idx = [
        np.where(y_auth_b == lbl)[0][0]
        for lbl in authorized_labels
        if lbl != target_label and (y_auth_b == lbl).any()
    ]
    val_b_idx = np.array([va_pos] + other_val_idx)
    X_va_raw  = X_auth_b[val_b_idx]
    y_va_raw  = y_auth_b[val_b_idx]

    X_te_raw = X_auth_b[te_pos : te_pos + 1]
    y_te_raw = y_auth_b[te_pos : te_pos + 1]

    flat_tr = X_tr_raw.reshape(len(X_tr_raw), -1)
    mu  = flat_tr.mean(0)
    sig = flat_tr.std(0) + 1e-8

    def _norm(X):
        f = X.reshape(len(X), -1)
        return ((f - mu) / sig).astype(np.float32).reshape(X.shape)

    X_tr = _norm(X_tr_raw)
    X_va = _norm(X_va_raw)
    X_te = _norm(X_te_raw)
    X_vr = _norm(X_val_rogue_b)
    X_tr_ = _norm(X_test_rogue_b)

    y_tr = (y_tr_raw == target_label).astype(np.int64)
    y_va = (y_va_raw == target_label).astype(np.int64)
    y_te = (y_te_raw == target_label).astype(np.int64)

    return {
        "X_tr": X_tr,  "y_tr": y_tr,
        "X_va": X_va,  "y_va": y_va,
        "X_te": X_te,  "y_te": y_te,
        "X_vr": X_vr,
        "X_tr_": X_tr_,
        "fold_idx":     int(fold_idx),
        "n_transients": int(n_transients),
        "n_train_pos":  int(y_tr.sum()),
        "n_val_pos":    int(y_va.sum()),
    }


def _fold_worker(fold_spec, model_name, epochs, patience, dropout, batch_size, lr, weight_decay, augment=None):
    """
    Train and evaluate one LOTO fold. Returns four scores:

      val_auth_score    P(class=1) for the val transient    → threshold selection
      val_rogue_scores  P(class=1) for all val_rogue samples → threshold selection
      test_auth_score   P(class=1) for the test transient   → final evaluation
      test_rogue_scores P(class=1) for all test_rogue samples → final evaluation
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from torch.utils.data import TensorDataset, DataLoader
    from binary_pla.models import build_binary_model
    from binary_pla.trainer import _class_weights, _run_epoch, _EarlyStopping

    def _loader(X, y, shuffle):
        X_c = np.ascontiguousarray(X).copy()
        y_c = y.astype(np.int64).copy()
        ds = TensorDataset(torch.from_numpy(X_c), torch.from_numpy(y_c))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    def _scores(model, X):
        """Return P(class=1) for every row of X."""
        X_t = torch.from_numpy(np.ascontiguousarray(X).copy())
        with torch.no_grad():
            return F.softmax(model(X_t), dim=1)[:, 1].numpy()

    train_loader = _loader(fold_spec["X_tr"], fold_spec["y_tr"], shuffle=True)
    val_loader   = _loader(fold_spec["X_va"], fold_spec["y_va"], shuffle=False)

    torch.manual_seed(fold_spec["fold_idx"])
    model     = build_binary_model(model_name, dropout=dropout)
    weights   = _class_weights(train_loader)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    stopper   = _EarlyStopping(patience)

    for _ in range(1, epochs + 1):
        _run_epoch(model, train_loader, criterion, optimizer, True, augment)
        vl_loss, _ = _run_epoch(model, val_loader, criterion, None, False)
        scheduler.step(vl_loss)
        if stopper.step(vl_loss, model):
            break
    stopper.restore(model)
    model.eval()

    val_auth_score    = float(_scores(model, fold_spec["X_va"][fold_spec["y_va"] == 1])[0])
    val_rogue_scores  = _scores(model, fold_spec["X_vr"]).tolist()
    test_auth_score   = float(_scores(model, fold_spec["X_te"])[0])
    test_rogue_scores = _scores(model, fold_spec["X_tr_"]).tolist()

    return {
        "fold_idx":         fold_spec["fold_idx"],
        "val_auth_score":   val_auth_score,
        "val_rogue_scores": val_rogue_scores,
        "test_auth_score":  test_auth_score,
        "test_rogue_scores": test_rogue_scores,
    }


def _find_optimal_threshold(val_auth_scores, val_rogue_scores):
    """Threshold that maximises ADR on the val set. Returns (threshold, fpr, tpr, thresholds, adr)."""
    scores = np.array(val_auth_scores + val_rogue_scores)
    labels = np.array([1] * len(val_auth_scores) + [0] * len(val_rogue_scores))

    # roc_curve: fpr = rogue FAR = 1 - rogue_tvr, tpr = auth_tvr
    fpr, tpr, thresholds = roc_curve(labels, scores)
    adr = (tpr + (1.0 - fpr)) / 2.0
    best_idx = int(np.argmax(adr))

    return float(thresholds[best_idx]), fpr, tpr, thresholds, adr


def _apply_threshold(auth_scores, rogue_scores, threshold):
    """Compute auth_tvr and rogue_tvr at a given threshold."""
    auth_tvr  = float(np.mean(np.array(auth_scores)  >= threshold))
    rogue_tvr = float(np.mean(np.array(rogue_scores) < threshold))
    return auth_tvr, rogue_tvr


def run_loto(
    trial_name: str,
    repr_type:  str   = "fv",
    model_name: str   = "mlp_fv",
    epochs:     int   = 50,
    patience:   int   = 10,
    dropout:      float = DROPOUT,
    batch_size:   int   = BATCH_SIZE,
    n_jobs:       int   = N_JOBS,
    seed:         int   = 42,
    lr:           float = LR,
    weight_decay: float = WEIGHT_DECAY,
    augment               = None,
    verbose:    bool  = True,
) -> dict:
    """
    LOTO cross-validation with val-based optimal threshold selection.

    Returns a dict keyed by device name, each with auth_tvr, rogue_tvr, adr at
    the default threshold (0.5) and at threshold_opt (chosen on val scores).
    Also includes val ROC data for plotting and a "_summary" key with trial means.
    """
    from binary_pla.data_loader import _load_cache, device_label, TRIALS

    authorized_names  = TRIALS[trial_name]["authorized"]
    rogue_names       = TRIALS[trial_name]["rogue"]
    authorized_labels = [device_label(n) for n in authorized_names]
    rogue_labels      = [device_label(n) for n in rogue_names]

    X_base, y_base, _     = _load_cache(repr_type, use_windowed=False)
    X_win,  y_win,  n_win = _load_cache(repr_type, use_windowed=True)

    auth_mask_b  = np.isin(y_base, authorized_labels)
    rogue_mask_b = np.isin(y_base, rogue_labels)
    auth_mask_w  = np.isin(y_win,  authorized_labels)

    X_auth_b, y_auth_b = X_base[auth_mask_b], y_base[auth_mask_b]
    X_rogue_b, y_rogue_b = X_base[rogue_mask_b], y_base[rogue_mask_b]
    X_auth_w, y_auth_w = X_win[auth_mask_w], y_win[auth_mask_w]

    val_rogue_mask, test_rogue_mask = _split_rogues(y_rogue_b, rogue_labels, seed)
    X_val_rogue,  y_val_rogue  = X_rogue_b[val_rogue_mask],  y_rogue_b[val_rogue_mask]
    X_test_rogue, y_test_rogue = X_rogue_b[test_rogue_mask], y_rogue_b[test_rogue_mask]

    fold_specs   = []
    fold_targets = []

    for target_name in authorized_names:
        target_label = device_label(target_name)
        n_transients = int((y_auth_b == target_label).sum())

        for fold_idx in range(n_transients):
            spec = _prepare_fold(
                X_auth_b, y_auth_b,
                X_auth_w, y_auth_w,
                X_val_rogue,  y_val_rogue,
                X_test_rogue, y_test_rogue,
                authorized_labels, target_label,
                fold_idx, n_win, model_name,
            )
            fold_specs.append(spec)
            fold_targets.append(target_name)

    n_total = len(fold_specs)

    if verbose:
        print(f"[run_loto] {trial_name} | {repr_type} | {model_name} | "
              f"seed={seed} | n_jobs={n_jobs}")
        for name in authorized_names:
            lbl = device_label(name)
            nf  = int((y_auth_b == lbl).sum())
            print(f"  {name}: {nf} folds")
        print(f"  Total folds : {n_total}")
        print(f"  Val rogues  : {val_rogue_mask.sum()} samples "
              f"({len(np.unique(y_rogue_b[val_rogue_mask]))} devices)")
        print(f"  Test rogues : {test_rogue_mask.sum()} samples "
              f"({len(np.unique(y_rogue_b[test_rogue_mask]))} devices)")

    fold_results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
        delayed(_fold_worker)(spec, model_name, epochs, patience, dropout, batch_size, lr, weight_decay, augment)
        for spec in fold_specs
    )

    for target_name, res in zip(fold_targets, fold_results):
        res["target_name"] = target_name

    per_device = {}

    for target_name in authorized_names:
        device_folds = [r for r, t in zip(fold_results, fold_targets)
                        if t == target_name]
        n_folds = len(device_folds)

        # Per-fold binary auth outcomes (0/1) and rogue_tvr at default threshold
        auth_correct  = [(r["test_auth_score"] >= 0.5) for r in device_folds]
        rogue_tvr_folds = [float(np.mean(np.array(r["test_rogue_scores"]) < 0.5))
                           for r in device_folds]

        auth_tvr  = float(np.mean(auth_correct))
        auth_std  = float(np.std(auth_correct))
        rogue_tvr = float(np.mean(rogue_tvr_folds))
        rogue_std = float(np.std(rogue_tvr_folds))

        # Pool val scores for threshold selection
        val_auth  = [r["val_auth_score"]   for r in device_folds]
        val_rogue = [s for r in device_folds for s in r["val_rogue_scores"]]

        threshold_opt, fpr, tpr, thresholds, adr_curve = \
            _find_optimal_threshold(val_auth, val_rogue)

        # Pool test scores and apply optimal threshold
        test_auth  = [r["test_auth_score"]    for r in device_folds]
        test_rogue = [s for r in device_folds for s in r["test_rogue_scores"]]
        auth_tvr_opt, rogue_tvr_opt = _apply_threshold(
            test_auth, test_rogue, threshold_opt
        )

        per_device[target_name] = {
            # Default threshold = 0.5
            "auth_tvr":      auth_tvr,
            "auth_tvr_std":  auth_std,
            "rogue_tvr":     rogue_tvr,
            "rogue_tvr_std": rogue_std,
            "adr":           (auth_tvr + rogue_tvr) / 2,
            "n_folds":       n_folds,
            # Optimal threshold (chosen on val, applied to test)
            "threshold_opt": threshold_opt,
            "auth_tvr_opt":  auth_tvr_opt,
            "rogue_tvr_opt": rogue_tvr_opt,
            "adr_opt":       (auth_tvr_opt + rogue_tvr_opt) / 2,
            # Val ROC data for plotting (safe to expose — no test data)
            "val_auth_scores":  val_auth,
            "val_rogue_scores": val_rogue,
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
        }

    def _mean(key):
        return float(np.mean([per_device[d][key] for d in authorized_names]))

    per_device["_summary"] = {
        "mean_auth_tvr":      _mean("auth_tvr"),
        "mean_auth_tvr_std":  _mean("auth_tvr_std"),
        "mean_rogue_tvr":     _mean("rogue_tvr"),
        "mean_rogue_tvr_std": _mean("rogue_tvr_std"),
        "mean_adr":           _mean("adr"),
        "mean_auth_tvr_opt":  _mean("auth_tvr_opt"),
        "mean_rogue_tvr_opt": _mean("rogue_tvr_opt"),
        "mean_adr_opt":       _mean("adr_opt"),
    }

    if verbose:
        s = per_device["_summary"]
        print(f"\n{'─'*70}")
        print(f"  {'Device':<12} {'Auth':>8} {'±':>6} {'Rogue':>8} {'±':>6} "
              f"{'ADR':>7} │ {'Auth*':>7} {'Rogue*':>8} {'ADR*':>7} {'thr*':>6}")
        print(f"  {'─'*66}")
        for name in authorized_names:
            d = per_device[name]
            print(f"  {name:<12} {d['auth_tvr']:>8.3f} {d['auth_tvr_std']:>6.3f} "
                  f"{d['rogue_tvr']:>8.3f} {d['rogue_tvr_std']:>6.3f} "
                  f"{d['adr']:>7.3f} │ "
                  f"{d['auth_tvr_opt']:>7.3f} {d['rogue_tvr_opt']:>8.3f} "
                  f"{d['adr_opt']:>7.3f} {d['threshold_opt']:>6.3f}")
        print(f"  {'─'*66}")
        print(f"  {'MEAN':<12} {s['mean_auth_tvr']:>8.3f} {s['mean_auth_tvr_std']:>6.3f} "
              f"{s['mean_rogue_tvr']:>8.3f} {s['mean_rogue_tvr_std']:>6.3f} "
              f"{s['mean_adr']:>7.3f} │ "
              f"{s['mean_auth_tvr_opt']:>7.3f} {s['mean_rogue_tvr_opt']:>8.3f} "
              f"{s['mean_adr_opt']:>7.3f}")
        print(f"  (* = optimal threshold from val)")
        print(f"{'─'*70}")

    return per_device
