[← Back to root](../README.md)

# `raw_dgt/` — DGT Matrix Preprocessing Pipeline

This module converts raw `.bin` IQ recordings into **150×150 DGT power matrices** and saves them as HDF5 caches. Unlike [`dl_classification/`](../dl_classification/README.md), which goes all the way to a 505-dim feature vector, this pipeline stops at the 2D matrix stage — feeding directly into GRU and BiGRU models that learn their own features from the time-frequency representation.

---

## Background

### What is the Discrete Gabor Transform (DGT)?

An RF transient is a **non-stationary signal**: its frequency content changes over time as the transmitter powers up. A plain FFT collapses all time information into a single spectrum, losing the temporal evolution of the signal — which is precisely where the hardware fingerprint lives.

The **Discrete Gabor Transform** solves this by decomposing the signal into overlapping, Gaussian-windowed short-time Fourier slices. Concretely:

1. A sliding Gaussian window of width M is applied to the transient at K_G overlapping positions along the time axis.
2. At each position a DFT of length M is computed.
3. The result is a 2D complex matrix of shape (K_G × M) — here **150 × 150**.
   - **Rows (150):** time positions (each row = one windowed slice of the transient).
   - **Columns (150):** frequency bins within that slice.

Taking the squared magnitude of this complex matrix gives the **DGT power matrix**: a spectrogram-like image where bright regions indicate strong energy at a particular time and frequency. The GRU reads this matrix row-by-row (150 time steps, each with 150 frequency features).

### Why not just use FFT?

The DGT preserves the joint time-frequency structure of the transient. This matters because different hardware imperfections (e.g., oscillator warm-up vs. amplifier non-linearity) manifest at different moments during the power-on burst. An FFT cannot distinguish "distortion at the start" from "distortion at the end."

### Complexity and the fast implementation

The naive DGT is O(M × K_G × M×N) — a triple-nested loop that is extremely slow for long signals. [`dl_classification/fast_fingerprint.py`](../dl_classification/README.md) replaces this with an FFT-based algorithm achieving ~100× speedup, and is called by both this pipeline and the FV pipeline.

---

## Module Contents

### `dgt_data_loader.py`
`DGTPreprocessor` class: reads `.bin` IQ files from `original_dataset/`, runs the preprocessing chain (normalization, transient detection, Chebyshev filtering from `src/preprocessing.py`), computes the DGT via `fast_fingerprint.py`, and returns the **150×150 power matrix** (magnitude² of the complex Gabor coefficients) for each transient.

In **windowed mode** (used for training), a sliding window is applied to each transient before computing the DGT, generating up to 10 matrices per transient. This 10× data augmentation is applied to the training split only — the original transients (one matrix each) are used for validation and test, keeping those splits uncontaminated.

### `run_pipeline_dgt.py`
Orchestrates cache building. Calls `DGTPreprocessor` for all 12 devices and writes two HDF5 files:

| File | Samples | Shape per sample | Description |
|------|---------|-----------------|-------------|
| `processed_data/dgt_original.h5` | 134 | (150, 150) | One matrix per transient, no augmentation |
| `processed_data/dgt_windowed.h5` | 1340 | (150, 150) | 10 windows per transient |

Each HDF5 file stores a `windows_per_transient` attribute used by the data loader to correctly reconstruct transient-level groupings for the leave-one-transient-out split.

---

## How it fits in the project

```
original_dataset/   →   dgt_data_loader.py (DGT computation)
                                │
                    dgt_original.h5 / dgt_windowed.h5
                                │
                    binary_pla/data_loader.py (train/val/test split)
                                │
                    GRU-DGT / BiGRU+SpecAug (in notebooks/02_dl_honest.ipynb)
```

The DL training pipeline that consumes these caches is in [`binary_pla/`](../binary_pla/README.md). Caches are built by running [`notebooks/00_setup.ipynb`](../notebooks/README.md).
