[← Back to root](../README.md)

# `binary_pla/` — Deep Learning Training Pipeline

This module contains the PyTorch training infrastructure for the per-device binary PLA classifiers. It is consumed by the experiment notebooks and depends on the HDF5 caches produced by [`dl_classification/`](../dl_classification/README.md) and [`raw_dgt/`](../raw_dgt/README.md).

---

## Module Map

| File | Role |
|------|------|
| `config.py` | Central configuration — paths, trial definitions, hyperparameters |
| `models.py` | PyTorch model definitions (MLP-FV, GRU-DGT, BiGRU) |
| `data_loader.py` | HDF5 loading, three-way split, normalization |
| `trainer.py` | Per-device training loop with early stopping |
| `augmentation.py` | SpecAugment data augmentation for DGT matrices |
| `loto.py` | Leave-One-Transient-Out cross-validation |
| `results_io.py` | JSON results serialization with metadata |

---

## Component Details

### `config.py`
Single source of truth for all experiment parameters. Defines:
- **HDF5 paths** — absolute paths to `fv_original.h5`, `fv_windowed.h5`, `dgt_original.h5`, `dgt_windowed.h5`.
- **Trial definitions** — 3 trials, each specifying which of the 12 devices are authorized vs. rogue. Each authorized device gets its own binary classifier.

  | Trial | Authorized | Rogue |
  |-------|-----------|-------|
  | trial_1 | device3, 2, 12, 9 | device8, 11, 5, 1, 6, 10, 7, 4 |
  | trial_2 | device10, 6, 12, 3, 11, 1 | device2, 5, 8, 7, 4, 9 |
  | trial_3 | device1, 10, 9, 8, 12, 11, 6, 7 | device4, 2, 5, 3 |

- **Random seeds** — `[42, 7, 13, 99, 2024]` for 5-seed uncertainty quantification.
- **Training hyperparameters** — batch size (64), learning rate (1e-3), dropout (0.3), weight decay (1e-4), train/val/test fractions (70/15/15).
- **SpecAugment config** — time mask max (30 bins), frequency mask max (30 bins), noise std (0.02).

### `models.py`
Three self-contained PyTorch binary classifiers:

**MLP-FV** — takes a 505-dim z-scored feature vector. Uses the same feature space as the SVM baseline, allowing direct comparison.
```
Linear(505→256) → BatchNorm → ReLU → Dropout
Linear(256→128) → BatchNorm → ReLU → Dropout
Linear(128→2)
```

**GRU-DGT** — takes a 150×150 DGT matrix, read row-by-row as a sequence of 150 time steps each with 150 frequency features. Learns its own representation directly from the time-frequency image.
```
GRU(input=150, hidden=64, layers=2, bidirectional=False) → last hidden state
Dropout → Linear(64→2)
```

**BiGRU-DGT** — same input as GRU-DGT, but processes the sequence in both temporal directions simultaneously. The forward and backward final hidden states are concatenated before classification.
```
BiGRU(input=150, hidden=64×2, layers=2) → concat(fwd, bwd) hidden
Dropout → Linear(128→2)
```

All three are instantiated via `build_binary_model("mlp_fv" | "gru_dgt" | "bigru_dgt")`.

### `data_loader.py`
Loads a device's samples from HDF5, then performs a **transient-aware three-way split** (70% train / 15% val / 15% test). "Transient-aware" means all augmented windows derived from the same original transient are kept together — they all go to the same partition. This prevents the common leakage bug where windows of the same transient end up in both train and test.

Normalization (z-score) is fitted on the training split only and applied to val and test, preventing any leakage from future data.

### `trainer.py`
Per-device binary training loop. For each (device, seed, model_type) combination:
- Uses `CrossEntropyLoss` with class weights inversely proportional to class frequency (handles the imbalanced auth vs. rogue ratio).
- Optimizes with `AdamW`.
- Reduces learning rate on validation loss plateau (`ReduceLROnPlateau`).
- Stops early if validation loss does not improve for `patience` epochs (default: 10).
- Saves the best checkpoint (lowest validation loss) and reloads it for final evaluation.

### `augmentation.py`
`SpecAugmentDGT`: on-the-fly augmentation applied to 150×150 DGT matrices **during training only**. Randomly masks contiguous bands along the time axis and frequency axis (setting them to zero), and adds small Gaussian noise. This regularizes the GRU models against overfitting on the small dataset. Applied with probability `p=0.5` per sample per batch.

### `loto.py`
**Leave-One-Transient-Out (LOTO)** cross-validation. For each fold, one original transient of the authorized device is held out as the test sample. The remaining transients (and their windows) form the training set. A separate validation set is used exclusively for threshold selection and early stopping — never for final evaluation.

After all folds, the optimal decision threshold is selected by maximizing ADR on the pooled validation scores. This threshold is then applied to the test scores to produce the final `auth_tvr` and `rogue_tvr`.

LOTO is an alternative to the held-out test set used in `notebooks/02_dl_honest.ipynb`. It uses all available data (important given the small dataset) at the cost of higher variance per fold.

### `results_io.py`
Saves and loads experiment results as JSON files in `results/`. Each file is wrapped in a metadata envelope recording: notebook ID, trial name, timestamp, and git hash — enabling reproducibility and preventing accidental overwriting of results from different runs.

---

## Data flow

```
fv_windowed.h5    →  data_loader.py  →  trainer.py  →  MLP-FV results
dgt_windowed.h5   →  data_loader.py  →  trainer.py  →  GRU / BiGRU results
                                            │
                                     augmentation.py (training only)
                                            │
                                     results_io.py → results/nb02_*.json
```

Used by: [`notebooks/02_dl_honest.ipynb`](../notebooks/README.md) (primary), [`notebooks/04_multiclass_classification.ipynb`](../notebooks/README.md).  
Depends on caches from: [`dl_classification/`](../dl_classification/README.md) and [`raw_dgt/`](../raw_dgt/README.md).
