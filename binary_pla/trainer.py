"""
Per-device binary classifier training.

One model per authorized device per trial. Checkpoints go to:
    results/checkpoints/{trial_name}/{model_name}/{windowed|non_windowed}/device_{N}.pt
"""

import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from binary_pla.models import build_binary_model, count_parameters
from binary_pla.config import LR, WEIGHT_DECAY, DROPOUT

_HERE     = os.path.dirname(os.path.abspath(__file__))
_ROOT     = os.path.dirname(_HERE)
_CKPT_DIR = os.path.join(_ROOT, "results", "checkpoints")


def _ckpt_path(trial_name, model_name, device_name, use_windowed):
    tag = "windowed" if use_windowed else "non_windowed"
    d   = os.path.join(_CKPT_DIR, trial_name, model_name, tag)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{device_name}.pt")


def _class_weights(train_loader):
    """Inverse-frequency weights — corrects for 1-vs-(N-1) class imbalance."""
    all_y = torch.cat([y for _, y in train_loader])
    n_pos = (all_y == 1).sum().float()
    n_neg = (all_y == 0).sum().float()
    total = n_pos + n_neg
    return torch.tensor([total / (2 * n_neg), total / (2 * n_pos)])


class _EarlyStopping:
    def __init__(self, patience):
        self.patience   = patience
        self.best       = float("inf")
        self.counter    = 0
        self.best_state = None

    def step(self, val_loss, model):
        if val_loss < self.best - 1e-4:  # require >1e-4 improvement to reset counter (ignores noise-level fluctuations)
            self.best    = val_loss
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore(self, model):
        if self.best_state:
            model.load_state_dict(self.best_state)


def _run_epoch(model, loader, criterion, optimizer, training, augment=None):
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for X, y in loader:
            if training and augment is not None:
                X = augment(X)
            logits = model(X)
            loss   = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct    += (logits.argmax(1) == y).sum().item()
            total      += len(y)
    return total_loss / total, correct / total


def train_device(
    device_name:  str,
    train_loader,
    val_loader,
    model_name:   str,
    trial_name:   str,
    use_windowed: bool,
    epochs:       int   = 50,
    lr:           float = LR,
    weight_decay: float = WEIGHT_DECAY,
    dropout:      float = DROPOUT,
    patience:     int   = 10,
    seed:         int   = 42,
    augment             = None,
    verbose:      bool  = True,
    save:         bool  = True,
):
    """Train one binary classifier. augment is applied per batch during training only."""
    torch.manual_seed(seed)

    model     = build_binary_model(model_name, dropout=dropout)
    n_params  = count_parameters(model)
    weights   = _class_weights(train_loader)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    stopper   = _EarlyStopping(patience=patience)

    best_val_acc  = 0.0
    best_val_loss = float("inf")

    if verbose:
        aug_str = f"  augment={augment.__class__.__name__}" if augment else ""
        print(f"  [{device_name}]  params={n_params:,}  "
              f"train={len(train_loader.dataset)}  val={len(val_loader.dataset)}{aug_str}")

    t0 = time.time()
    epoch_stopped = epochs  # updated on early stop; otherwise ran all epochs
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = _run_epoch(model, train_loader, criterion, optimizer, True,  augment)
        vl_loss, vl_acc = _run_epoch(model, val_loader,   criterion, None,      False, None)

        scheduler.step(vl_loss)
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss

        if stopper.step(vl_loss, model):
            epoch_stopped = epoch
            if verbose:
                print(f"    early stop @ epoch {epoch}  "
                      f"best_val_acc={best_val_acc:.3f}")
            break

    stopper.restore(model)
    elapsed = time.time() - t0

    if save:
        path = _ckpt_path(trial_name, model_name, device_name, use_windowed)
        torch.save(model.state_dict(), path)

    return {
        "model":      model,
        "val_acc":    best_val_acc,
        "val_loss":   best_val_loss,
        "epochs_run": epoch_stopped,
        "elapsed_s":  elapsed,
    }


def train_trial(
    trial_data,
    model_name:   str,
    epochs:       int   = 50,
    patience:     int   = 10,
    dropout:      float = 0.3,
    seed:         int   = 42,
    augment             = None,
    verbose:      bool  = True,
    save:         bool  = True,
):
    """Train one binary model per authorized device. Returns {device_name: result}."""
    trial_name   = trial_data["trial_name"]
    use_windowed = trial_data["use_windowed"]

    if verbose:
        tag = "windowed" if use_windowed else "non-windowed"
        print(f"\n{'='*60}")
        print(f"  {trial_name.upper()}  |  {model_name.upper()}  |  {tag}")
        print(f"  authorized devices: {trial_data['authorized']}")
        print(f"{'='*60}")

    results = {}
    for dev_name in trial_data["authorized"]:
        d = trial_data["per_device"][dev_name]
        results[dev_name] = train_device(
            device_name  = dev_name,
            train_loader = d["train_loader"],
            val_loader   = d["val_loader"],
            model_name   = model_name,
            trial_name   = trial_name,
            use_windowed = use_windowed,
            epochs       = epochs,
            patience     = patience,
            dropout      = dropout,
            seed         = seed,
            augment      = augment,
            verbose      = verbose,
            save         = save,
        )
    return results


