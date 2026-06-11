[← Back to root](../README.md)

# `src/` — Original RF-DNA Preprocessing Pipeline

> **Attribution:** This module is the original work of Ildi Alla, Selma Yahia, Valeria Loscri, and Hossien Eldeeb, published as: *"Robust Device Authentication in Multi-Node Networks: ML-Assisted Hybrid PLA Exploiting Hardware Impairments"*, ACSAC 2024. It is included here unmodified and serves as the reference baseline. Original code: [github.com/PLA-AP/PLA](https://github.com/PLA-AP/PLA)

---

## Background

### What is RF-DNA fingerprinting?

Every physical radio transmitter has tiny hardware imperfections — tolerances in capacitors and inductors, non-linearities in amplifiers, slight frequency offsets in oscillators. These imperfections are repeatable: the same device always produces the same subtle distortions on its signal. RF-DNA fingerprinting captures this hardware signature from the **transient** (the brief power-on burst at the start of each transmission) and uses it to identify a specific device, without requiring any cryptographic key.

The fingerprint is a 505-dimensional vector of statistical descriptors extracted from the **Discrete Gabor Transform (DGT)** of the transient signal. The DGT is described in detail in [`raw_dgt/README.md`](../raw_dgt/README.md).

### What is Physical Layer Authentication (PLA)?

PLA is a binary classification problem: given a received transient, decide whether it came from the **authorized** device or from a **rogue** impersonator. A separate binary classifier is trained per authorized device. The system is evaluated using three metrics:

| Metric | Definition |
|--------|-----------|
| **TVR (True Verification Rate)** | Fraction of genuine transmissions correctly accepted. Also called TDR (True Detection Rate) in this codebase. |
| **FDR (False Detection Rate)** | Fraction of rogue transmissions incorrectly accepted as genuine. Lower is better. |
| **ADR (Average Detection Rate)** | Average of Auth TVR and Rogue TVR — the headline metric. ADR = 1.0 means perfect; ADR = 0.5 means random. |

---

## Module Contents

### `preprocessing.py`
Loads raw IQ (complex64) samples from `.bin` files recorded at 20 MHz on a BladeRF AX4 SDR, detects transients by amplitude thresholding, applies a 4th-order Chebyshev low-pass filter (cutoff 5 MHz, ripple 0.5 dB), and discards transients that are too short or too weak. Returns a list of cleaned transient arrays ready for feature extraction.

### `features_generation.py`
Implements the full RF-DNA fingerprint pipeline:
1. Computes the 150×150 DGT matrix for each transient (see [`raw_dgt/README.md`](../raw_dgt/README.md) for what DGT is).
2. Divides the matrix into a 15×15 grid of non-overlapping patches.
3. Extracts four statistics from each patch (mean, standard deviation, skewness, kurtosis).
4. Concatenates all patch statistics → **505-dimensional feature vector** (15 × 15 × 4 = 900, minus 395 dropped due to NaN/variance filtering in the original paper → 505).

This 505-dim vector is the input to both the SVM baseline and the MLP-FV deep learning model.

### `features_selections.py`
Dimensionality reduction and feature selection methods applied before training the SVM:
- **ANOVA** — univariate F-test; selects the top-k most discriminative features per fold.
- **Mutual Information** — non-linear feature ranking.
- **RFE** — Recursive Feature Elimination using a linear SVM as the estimator.
- **PCA** — principal component analysis for unsupervised dimensionality reduction.

> **Evaluation note:** In the original pipeline, ANOVA selection is fitted on the full dataset before cross-validation splits. This is Protocol Variant A documented in [`notebooks/01_ml_baseline.ipynb`](../notebooks/README.md).

### `evaluation.py`
Computes TVR, FDR, and the closeness score for a trained binary classifier. Also identifies which samples were misclassified and which physical device they belonged to.

### `dataloader.py`
End-to-end data loading for the ML pipeline: reads raw `.bin` files, runs the full preprocessing and feature extraction chain from `preprocessing.py` and `features_generation.py`, applies feature selection, and returns `(X_train, X_test, y_train, y_test)` arrays ready for scikit-learn models. Can cache the processed fingerprints to an HDF5 file to avoid recomputation.

---

## How it fits in the project

```
original_dataset/  →  preprocessing.py  →  features_generation.py  →  505-dim vectors
                                                                              │
                                                                     features_selections.py
                                                                              │
                                                                         RF_Fingerprint.py (SVM)
                                                                         dl_classification/ (MLP-FV cache)
```

The evaluation protocol variants for this pipeline are documented in [`notebooks/01_ml_baseline.ipynb`](../notebooks/README.md). The DL alternatives that bypass the feature-selection step are in [`binary_pla/`](../binary_pla/README.md).
