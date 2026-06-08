[← Back to root](../README.md)

# `notebooks/` — Experiment Notebooks

These notebooks implement the full experimental workflow, from building data caches through the final ML vs. DL comparison. **Run them in order.** Each notebook saves its results to `results/` as a JSON file, so later notebooks can load results from earlier ones without re-running training.

---

## Execution Order

| # | Notebook | What it does | Outputs |
|---|----------|-------------|---------|
| 0 | `00_setup.ipynb` | Build all HDF5 caches from raw data | `processed_data/*.h5` (4 files) |
| 1 | `01_ml_baseline.ipynb` | Replicate SVM pipeline + expose 3 biases | `results/nb01_ml_baseline_trial_*.json` |
| 2 | `02_dl_honest.ipynb` | Honest DL evaluation, 5 seeds | `results/nb02_dl_honest_trial_*.json` |
| 3 | `03_comparison.ipynb` | ML vs. DL side-by-side | Figures (displayed inline) |
| 4 | `04_multiclass_classification.ipynb` | 12-class device identification | `results/nb04_multiclass_all_devices.json` |

---

## Notebook Descriptions

### `00_setup.ipynb` — Cache Builder
**Run this first, once.** Calls `dl_classification/run_pipeline.py` and `raw_dgt/run_pipeline_dgt.py` to process the raw `.bin` IQ recordings for all 12 devices and write four HDF5 files to `processed_data/`:

| Cache | Samples | Feature type |
|-------|---------|-------------|
| `fv_original.h5` | 134 | 505-dim feature vectors, 1 per transient |
| `fv_windowed.h5` | 1340 | 505-dim feature vectors, 10 windows/transient |
| `dgt_original.h5` | 134 | 150×150 DGT matrices, 1 per transient |
| `dgt_windowed.h5` | 1340 | 150×150 DGT matrices, 10 windows/transient |

All subsequent notebooks load from these caches — never from raw `.bin` files directly.

---

### `01_ml_baseline.ipynb` — SVM Bias Audit
Replicates the original `RF_Fingerprint.py` pipeline and progressively corrects three evaluation biases. The goal is to understand what the original ADR of ~0.97 actually measures, and what a fair ADR estimate looks like.

**Bias 1 — ANOVA leakage:** In the original pipeline, feature selection (ANOVA top-k) is fitted on the full dataset before the cross-validation loop, meaning the model indirectly "sees" all test samples when selecting features. Fix: fit ANOVA inside each fold, on training data only.

**Bias 2 — Lucky fold selection:** The original pipeline reports the best fold (maximum ADR) rather than the average. Fix: average ADR across all folds.

**Bias 3 — Test-set k sweep:** The number of features k is chosen by sweeping over values and picking the one that maximizes test performance. Fix: fix k=50 (chosen on validation, never touching test).

| State | ADR |
|-------|-----|
| Original (all biases) | **0.973** |
| Fix 1 only | 0.786 |
| Fix 1 + 2 | 0.554 |
| Fix 1 + 2 + 3 | **0.458** |

Results are saved per trial and loaded by `03_comparison.ipynb`.

---

### `02_dl_honest.ipynb` — Honest DL Evaluation
Trains three deep learning models using a strict three-way split (70% train / 15% val / 15% test) with no leakage between splits. Each model is trained over 5 random seeds to quantify uncertainty.

**Models evaluated:**
- **MLP-FV** — 3-layer MLP on 505-dim feature vectors (same features as the SVM).
- **GRU-DGT** — 2-layer unidirectional GRU on 150×150 DGT matrices.
- **BiGRU+SpecAug** — 2-layer bidirectional GRU on DGT matrices, with SpecAugment regularization during training.

**Results (5 seeds, held-out test set):**

| Model | Auth TVR | Rogue TVR | ADR |
|-------|----------|-----------|-----|
| MLP-FV | 0.624 ± 0.050 | 0.680 ± 0.015 | **0.660 ± 0.019** |
| GRU-DGT | 0.648 ± 0.131 | 0.567 ± 0.059 | **0.604 ± 0.055** |
| BiGRU+SpecAug | 0.641 ± 0.069 | 0.616 ± 0.051 | **0.627 ± 0.025** |

All three are statistically indistinguishable (Wilcoxon p > 0.0625 for all pairs; n=5 is the structural minimum for p < 0.05).

---

### `03_comparison.ipynb` — ML vs. DL Comparison
Loads the saved results from notebooks 01 and 02 and displays them side by side. Examines whether the DL models offer any advantage over the corrected SVM baseline (ADR = 0.458), and discusses whether the gap is architectural or a consequence of the small dataset size (134 transients total).

---

### `04_multiclass_classification.ipynb` — 12-Class Device Identification
Extends the experiment to **multi-class classification**: instead of one binary classifier per device, a single model identifies which of the 12 devices transmitted a given transient. This is a harder but more scalable formulation.

Trains MLP-FV and BiGRU on all 12 devices and compares per-class accuracy. The goal is to test whether the low ADR in the binary experiments is due to limited data (the fundamental constraint) or to the binary classification formulation.

---

## Modules used by these notebooks

- [`binary_pla/`](../binary_pla/README.md) — training loop, models, data loader (used by notebooks 02, 04)
- [`dl_classification/`](../dl_classification/README.md) — FV cache builder (used by notebook 00)
- [`raw_dgt/`](../raw_dgt/README.md) — DGT cache builder (used by notebook 00)
- [`src/`](../src/README.md) — original preprocessing utilities (used indirectly via dl_classification)
