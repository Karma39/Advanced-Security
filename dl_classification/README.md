[← Back to root](../README.md)

# `dl_classification/` — Feature-Vector Preprocessing Pipeline

This module bridges the original preprocessing code in [`src/`](../src/README.md) and the deep learning training pipeline in [`binary_pla/`](../binary_pla/README.md). It processes raw `.bin` IQ recordings into **505-dimensional RF-DNA feature vectors** and saves them as HDF5 caches ready for the MLP-FV model.

---

## Background

### What is the 505-dim feature vector?

Rather than feeding the raw 150×150 DGT matrix into a model, the RF-DNA approach first extracts compact statistical descriptors:

1. The DGT power matrix is divided into a **15×15 grid** of non-overlapping patches.
2. Four statistics are computed per patch: **mean, standard deviation, skewness, kurtosis**.
3. These are concatenated into a single vector. After NaN/zero-variance filtering (as defined in the original paper), the result is a **505-dimensional vector** per transient.

This representation was designed for classical ML (SVM) but is also used here as the input to the MLP-FV deep learning model, allowing a direct apples-to-apples comparison.

### Windowed augmentation

The dataset has only ~134 transients across 12 devices — far too few to train a neural network from scratch. To increase training data, a **sliding window** is applied to each transient before the DGT computation: each transient yields up to 10 overlapping sub-windows, each producing its own 505-dim vector. This gives 10× more training samples (1340 total) while keeping the original transients for validation and test.

---

## Module Contents

### `data_loader_bridge.py`
`RFPreprocessor` class: wraps the preprocessing chain from `src/preprocessing.py` and `src/features_generation.py`. For each device, it:
1. Loads raw IQ data from `original_dataset/`.
2. Detects and filters transients (Chebyshev filtering, duration/magnitude thresholds).
3. Applies the sliding-window strategy (configurable `max_windows` per transient).
4. Calls `fast_fingerprint.py` to compute the DGT, then extracts the 505-dim feature vector from each windowed segment.

Returns arrays of shape `(n_transients × max_windows, 505)` with a corresponding group index identifying which original transient each window came from (needed for leave-one-transient-out splits).

### `fast_fingerprint.py`
An optimized FFT-based implementation of the Discrete Gabor Transform. The original DGT in `src/features_generation.py` uses a triple-nested Python loop with complexity O(M × K_G × M×N), making it prohibitively slow when generating thousands of augmented windows. This module achieves ~100× speedup by restructuring the computation as a batched FFT convolution, enabling interactive experimentation with the full augmented dataset.

See [`raw_dgt/README.md`](../raw_dgt/README.md) for a conceptual explanation of what DGT computes.

---

## Outputs

`RFPreprocessor` (called from [`notebooks/00_setup.ipynb`](../notebooks/README.md)) writes two HDF5 files:

| File | Samples | Shape per sample | Description |
|------|---------|-----------------|-------------|
| `processed_data/fv_original.h5` | 134 | (505,) | One vector per transient, no augmentation |
| `processed_data/fv_windowed.h5` | 1340 | (505,) | 10 windows per transient |

Each file includes a `transient_id` dataset mapping each sample back to its source transient, used by the data loader for group-aware train/val/test splits.

---

## How it fits in the project

```
original_dataset/   →   data_loader_bridge.py + fast_fingerprint.py
                                │
                    fv_original.h5 / fv_windowed.h5
                                │
                    binary_pla/data_loader.py (train/val/test split)
                                │
                    MLP-FV (in notebooks/02_dl_baseline.ipynb)
```

The DGT matrix pipeline (without feature extraction) is in [`raw_dgt/`](../raw_dgt/README.md).
